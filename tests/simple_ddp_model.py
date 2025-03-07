import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tensortrace import ModelTracer

# Example dataset
class DummyDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.data = torch.randn(size, 20)
        self.targets = torch.randint(0, 2, (size,))
    
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    
    def __len__(self):
        return self.size

# Example Model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        return self.layers(x)

def setup(rank, world_size):
    # Initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    print(f"Running DDP process on rank {rank}.")
    
    # Create model and move it to GPU
    device_id = rank % torch.cuda.device_count()
    model = SimpleModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    
    # Create dataset and dataloader
    dataset = DummyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    # Optimizer and loss function
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    # Trace variables within the forward pass
    with ModelTracer(model, [
        "forward", # Save output of forward function
        "forward.x", # Save input of forward function
    ]) as tracer:
        for epoch in range(3):
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
    print(tracer.results['forward'])

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)