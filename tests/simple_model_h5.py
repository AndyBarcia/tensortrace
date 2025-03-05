import torch
import torch.nn as nn
import numpy as np
import h5py

from tensortrace import ModelTracerH5PY

# Example Model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30)
        )
        self.decoder = nn.Linear(30, 5)
        self.variable = 0

    def forward(self, x):
        y = self.encoder(x)
        z = self.decoder(y)
        self.variable += 1
        return z

model = SimpleModel()

h5py_filename = "/tmp/my_trace.h5"
with ModelTracerH5PY(
    model, 
    [
        "forward", # Save output of forward function
        "forward.x", # Save input of forward function
        "forward.y", # Save local variable of forward
        "forward.self.variable", # Save class attribute
        "encoder.1", # Save output of first layer of encoder
    ], 
    h5py_filename,
    float_dtype=np.float16,
    compression="gzip",
    shuffle=True
) as traced_model:
    for _ in range(10):
        input_tensor = torch.randn(1, 10)
        output = traced_model(input_tensor)

# Load data from the HDF5 file
with h5py.File(h5py_filename, 'r') as f:
    variable_data = f['forward.self.variable'][:]
    iterations = f['forward.self.variable_iterations'][:]
    print(f"Variable Data: {variable_data}")
    print(f"Iterations: {iterations}")

    encoder_weight = f["encoder.1"][:]
    encoder_weight_iterations = f["encoder.1_iterations"]
    print(f"Encoder Weight: {encoder_weight}")
    print(f"Encoder Iterations: {encoder_weight_iterations}")

    forward = f["forward"][:]
    forward_iterations = f["forward_iterations"]
    print(f"forward: {forward}")
    print(f"forward iterations: {forward_iterations}")