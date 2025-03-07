import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tensortrace import ModelTracer, stack_padded_tensors

from .utils import DummyDataset, SimpleModel, setup, cleanup


def train():
    # Create model and move it to GPU
    device_id = 0
    model = SimpleModel().to(device_id)
    
    # Create dataset and dataloader
    dataset = DummyDataset(size=1000)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    # Optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Take the mean of all GPUs and all iteration collected since the last gathering.
    def mean_post_gather(name, values, iterations, ranks):
        values[0] = values[0].mean(dim=(0,1)) # (C)
        pass

    tracer = ModelTracer(
        model, 
        [
            "forward", # Save output of forward function
            "forward.x", # Save input of forward function
        ],
        post_gather_callbacks=[stack_padded_tensors, mean_post_gather],
        post_trace_callbacks=[stack_padded_tensors],
        gathering_interval=4
    )

    # Training loop
    with tracer:
        for epoch in range(100):
            model.train()
            
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device_id), target.to(device_id)
                
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target.long())
                loss.backward()
                optimizer.step()

    # Tensor of shape (N//G,C), where 
    # - N is the number of iterations
    # - G is the gathering interval.
    x = tracer.results['forward.x'].values[0] # (N//G,C)
    print(x.shape)


if __name__ == "__main__":
    train()