import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensortrace import ModelTracer, stack_padded_tensors, GlobalVariableResult, Variable

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

    # Take the mean of all GPUs and all iteration collected since the last gathering.
    def mean_post_gather(name, results: GlobalVariableResult):
        results.values = [results.values.mean(dim=(0,1))] # (G,B,C) -> (C)
        results.iterations = results.iterations[:1] # (G,) -> (1,)
        results.ranks = results.ranks[:1] # (G,) -> (1,)

    variables = Variable(
        [
            "forward", # Save output of forward function
            "forward.x", # Save input of forward function"
        ], 
        post_gather_callbacks=[stack_padded_tensors, mean_post_gather],
        post_trace_callbacks=[stack_padded_tensors],
        gather_interval=4
    )

    # Training loop
    with ModelTracer(model, variables):
        for epoch in range(100):
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
        # Tensor of shape (N//G,C), where 
        # - N is the number of iterations
        # - G is the gathering interval.
        x = variables.results['forward.x'].values # (N//G,C)
        print(x.shape)

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)