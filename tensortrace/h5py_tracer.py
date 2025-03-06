from .tracer import ModelTracer

import sys
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from typing import List, Any, Type,Optional
import h5py
from pathlib import Path


class ModelTracerH5PY(ModelTracer):
    def __init__(
        self, 
        model: nn.Module, 
        variables: List[str],
        dataset_file: str,
        trace_interval = 1,
        saving_interval = 1,
        eval_only = False,
        train_only = False,
        iteration_dtype: Type = np.int64,
        float_dtype: Type = np.float32,
        int_dtype: Type = np.int32,
        compression: Optional[str] = None,
        compression_opts: Optional[int] = None,
        shuffle: bool = False,
    ):
        super().__init__(model, variables, trace_interval, eval_only, train_only)
        self.saving_interval = saving_interval

        # Dataset type configurations
        self.iteration_dtype = iteration_dtype
        self.float_dtype = float_dtype
        self.int_dtype = int_dtype

        # Compression configurations
        self.compression = compression
        self.compression_opts = compression_opts
        self.shuffle = shuffle

        self.dataset_file = Path(dataset_file)

        # Remove embedding file if it already exists.
        if self.dataset_file.exists():
            self.dataset_file.unlink()
            print(f"{dataset_file} has been removed.")

        # Make sure the output directory already exists
        self.dataset_file.parent.mkdir(parents=True, exist_ok=True)
        # Create the specified h5py file. The datasets
        # will be created dynamically as we trace the model.
        self.h5_file = h5py.File(self.dataset_file, 'w')
        self.datasets = {}
    
    def _get_normalized_value(self, value: Any) -> Any:
        """Normalize input to numpy arrays with configured dtypes or scalar types."""
        if isinstance(value, torch.Tensor):
            np_value = value.detach().cpu().numpy()
            if value.is_floating_point():
                np_value = np_value.astype(self.float_dtype)
            else:
                np_value = np_value.astype(self.int_dtype)
            return np_value
        elif isinstance(value, (float, np.floating)):
            return self.float_dtype(value)
        elif isinstance(value, (int, np.integer)):
            return self.int_dtype(value)
        else:
            return value  # Handle other types as-is

    def _write_simple_type_dataset(self, name: str, variables: List[Any], iterations: List[int]):
        """Write scalar or non-tensor data to HDF5 dataset with configured settings."""
        if not variables:
            return

        # Determine dtype from the first variable (all assumed to be same type after normalization)
        content_dtype = variables[0].dtype if hasattr(variables[0], 'dtype') else type(variables[0])
        
        if name not in self.datasets:
            # Create datasets with inferred dtype and compression settings
            content_dataset = self.h5_file.create_dataset(
                name, 
                shape=(0,),
                maxshape=(None,),
                dtype=content_dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
                shuffle=self.shuffle
            )
            iteration_dataset = self.h5_file.create_dataset(
                f"{name}_iterations",
                shape=(0,),
                maxshape=(None,),
                dtype=self.iteration_dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
                shuffle=self.shuffle
            )
            self.datasets[name] = (content_dataset, iteration_dataset)
        else:
            content_dataset, iteration_dataset = self.datasets[name]
        
        # Append data
        content_dataset.resize(content_dataset.shape[0] + len(variables), axis=0)
        content_dataset[-len(variables):] = variables

        iteration_dataset.resize(iteration_dataset.shape[0] + len(iterations), axis=0)
        iteration_dataset[-len(iterations):] = iterations

    def _write_tensor_dataset(self, name: str, variables: List[np.ndarray], iterations: List[int]):
        """Write tensor data to HDF5 dataset with padding and configured settings."""
        tensor_dims = {len(x.shape) for x in variables}
        if len(tensor_dims) != 1:
            raise ValueError(f"Variable {name} has inconsistent tensor dimensions: {tensor_dims}")
        tensor_dims = tensor_dims.pop()
        

        # Calculate padding and stack
        tensor_sizes = [x.shape for x in variables]
        tensor_max_shape = [max(sizes) for sizes in zip(*tensor_sizes)]
        padded_vars = []
        for tensor in variables:
            pad = [(0, max_dim - dim) for max_dim, dim in zip(tensor_max_shape, tensor.shape)]
            padded_vars.append(np.pad(tensor, pad) if any(pad) else tensor)
        variables = np.stack(padded_vars)

        if name not in self.datasets:
            # Create datasets with inferred dtype and compression settings
            content_dataset = self.h5_file.create_dataset(
                name, 
                shape=(0, *tensor_max_shape),
                maxshape=(None, *tensor_max_shape),
                dtype=variables.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
                shuffle=self.shuffle
            )
            iteration_dataset = self.h5_file.create_dataset(
                f"{name}_iterations",
                shape=(0,),
                maxshape=(None,),
                dtype=self.iteration_dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
                shuffle=self.shuffle
            )
            self.datasets[name] = (content_dataset, iteration_dataset)
        else:
            content_dataset, iteration_dataset = self.datasets[name]
            if any(tensor_max_shape[i] > content_dataset.shape[i+1] for i in range(tensor_dims)):
                raise ValueError(f"Variable {name} exceeds existing dataset dimensions")

        # Append data
        content_dataset.resize(content_dataset.shape[0] + len(variables), axis=0)
        content_dataset[-len(variables):] = variables
        iteration_dataset.resize(iteration_dataset.shape[0] + len(iterations), axis=0)
        iteration_dataset[-len(iterations):] = iterations
    
    def write_results(self):
        """ Write traced results to H5PY file, supporting multiple data types. """
        for name, variables in self.results.items():
            # Normalize all variables to numpy-compatible format
            normalized_variables = [self._get_normalized_value(x) for x in variables]
            
            # Determine type for writing
            if isinstance(normalized_variables[0], (np.ndarray)):
                self._write_tensor_dataset(name, normalized_variables, self.results_iterations[name])
            else:
                self._write_simple_type_dataset(name, normalized_variables, self.results_iterations[name])

    def __enter__(self):
        # When entering the context, we need to wrap the model's forward function
        # to enable tracing. After exiting the context, we will restore the original 
        # function.
        self.original_forward = self.model.forward

        def wrapped_forward(*args, **kwargs):
            self.current_iteration += 1

            trace_needed = (
                self.current_iteration % self.trace_interval == 0 and
                (not self.eval_only or not self.model.training) and
                (not self.train_only or self.model.training)
            )

            if trace_needed:
                original_trace = sys.gettrace()
                sys.settrace(partial(self._trace_calls, target_paths=self.target_paths))

            try:
                output = self.original_forward(*args, **kwargs)
            finally:
                if trace_needed:
                    sys.settrace(original_trace)

            # If not saving this iteration, just return the output
            if self.current_iteration % self.saving_interval != 0:
                return output

            # Write results to the h5py file
            self.write_results()

            # Reset tracked variables for this iteration
            self.results = {}
            self.results_iterations = {}

            return output

        self.model.forward = wrapped_forward
        return self


class H5PYTracedModel(nn.Module):
    def __init__(
        self, 
        model: nn.Module, 
        variables: List[str],
        dataset_file: str,
        trace_interval = 1,
        saving_interval = 1,
        eval_only = False,
        train_only = False,
        iteration_dtype: Type = np.int64,
        float_dtype: Type = np.float32,
        int_dtype: Type = np.int32,
        compression: Optional[str] = None,
        compression_opts: Optional[int] = None,
        shuffle: bool = False,
    ):
        super()
        self.model = model
        self.tracer = ModelTracerH5PY(
            model, 
            variables, 
            dataset_file, 
            trace_interval, 
            saving_interval, 
            eval_only, 
            train_only, 
            iteration_dtype, 
            float_dtype, 
            int_dtype, 
            compression, 
            compression_opts, 
            shuffle
        )

    def forward(self, *args, **kwargs):
        with self.tracer:
            return self.model(*args, **kwargs)