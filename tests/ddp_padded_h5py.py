import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensortrace import ModelTracer, stack_padded_tensors, H5PYSaverCallback
from tensortrace.h5py_tracer import H5PYSaverCallback

from .utils import DummyDataset, SimpleModel, setup, cleanup


def train(rank, world_size):
    setup(rank, world_size)
    print(f"Running DDP process on rank {rank}.")
    
    # Create model and move it to GPU
    device_id = rank % torch.cuda.device_count()
    model = SimpleModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    
    # Create dataset and dataloader
    dataset = DummyDataset(size=1000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    # Optimizer and loss function
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Save values as they are gathered into the H5Py file
    saver_callback = H5PYSaverCallback(dataset_file="test.h5")
    tracer = ModelTracer(
        model, 
        [
            "forward", # Save output of forward function
            "forward.x", # Save input of forward function
        ],
        post_gather_callbacks=[stack_padded_tensors, saver_callback],
        gathering_interval=2
    )

    # Training loop
    with tracer:
        for epoch in range(4):
            sampler.set_epoch(epoch)
            ddp_model.train()
            
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device_id), target.to(device_id)
                
                optimizer.zero_grad()
                output = ddp_model(data)
                loss = loss_fn(output, target.long())
                loss.backward()
                optimizer.step()

    # Example accessing values:
    if rank == 0:
        # Dataset of shape (N//G,G*W,B,C), where
        # - N is the number of iterations
        # - G is the gathering interval
        # - W is the world size.
        # - B is the batch size
        print(saver_callback.datasets['forward.x'])

    saver_callback.close()
    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)