import torch
import torch.nn as nn
from tensortrace import ModelTracer

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

# Trace variables within the forward pass
with ModelTracer(model, [
    "forward", # Save output of forward function
    "forward.x", # Save input of forward function
    "forward.y", # Save local variable of forward
    "forward.self.variable", # Save class attribute
    "encoder.1", # Save output of first layer of encoder
]) as tracer:
    for _ in range(10):
        input_tensor = torch.randn(1, 10)
        output = tracer(input_tensor)

# Example accessing values:
variable_values = tracer.results['forward.self.variable']
variable_iterations = tracer.results_iterations['forward.self.variable']