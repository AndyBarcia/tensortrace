# TensorTracer

[![PyPI version](https://badge.fury.io/py/tensortracer.svg)](https://badge.fury.io/py/tensortracer)

TensorTracer is a PyTorch utility for tracing and saving model variables during execution. It allows you to easily monitor the values of tensors, scalars, and other Python objects within your model, and optionally save them to an HDF5 file for later analysis.

## Features

*   **Flexible Variable Tracking:** Trace any combination of tensors, scalars, and other Python objects within your model's methods and local variables, including in nested the dictionary.
*   **Distributed Model Support:** Transparently trace distributed models; values are automatically gathered in a single process, and the rank of the process each value was in is saved. 
*   **Wildcard Support:** Use wildcards (`*`) in variable paths to match multiple variables or nested structures.
*   **Callback Support:** Allows manipulating captured values as they are traced and saved.
*   **HDF5 Storage:**  Optionally save traced variables to an HDF5 file for efficient storage and later analysis, with configurable data types and compression.
*   **Change Detection:**  Only saves variables that have changed since the last iteration, reducing storage overhead.
*   **Iteration Tracking**: Save the iteration in which a variable changed.

## Installation

```bash
pip install tensortrace
```

## Usage

Example of basic usage with a distributed model. The results of each GPU is gathered every 4 iterations.

```python
tracer = ModelTracer(
    model, 
    [
        "forward", # Save output of forward function
        "forward.x", # Save input of forward function
    ],
    pre_gather_callbacks=[stack_padded_tensors],
    post_gather_callbacks=[stack_padded_tensors],
    post_trace_callbacks=[stack_padded_tensors],
    gathering_interval=4
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
    # Tensor of shape (N//G,W,G,B,C), where 
    # - N is the number of iterations
    # - G is the gathering interval.
    # - W is the world size.
    # - B is the batch size
    x = tracer.results['forward.x'].values[0] # (N*2,B,C)
    print(x.shape)
```

This is a similar example, but using `H5PYSaverCallback` to move the traced values
to a file dataset as they are gathered in the main process. 

```python
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
```

This is another similar example, but the mean of the tensor is taken across all GPUs and batches every 4 iterations. This can be used to reduce the memory usage of traced tensor.

```python
# Take the mean of all iteration collected since the last gathering.
def mean_post_gather(name, values, iterations, ranks):
    values[0] = values[0].mean(dim=(0,1)) # (G,B,C) -> (C)
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
```