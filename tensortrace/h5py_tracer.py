import torch
import numpy as np
from typing import List, Any, Type,Optional
import h5py
from pathlib import Path
import torch.distributed as dist


class H5PYSaverCallback():
    def __init__(
        self, 
        dataset_file: str,
        iteration_dtype: Type = np.int64,
        float_dtype: Type = np.float32,
        int_dtype: Type = np.int32,
        compression: Optional[str] = None,
        compression_opts: Optional[int] = None,
        shuffle: bool = False,
        existing_ok: bool = True,
    ):
        # Dataset type configurations
        self.iteration_dtype = iteration_dtype
        self.float_dtype = float_dtype
        self.int_dtype = int_dtype

        # Compression configurations
        self.compression = compression
        self.compression_opts = compression_opts
        self.shuffle = shuffle

        self.dataset_file = Path(dataset_file)

        # Remove embedding file if it already exists. Make sure
        # to only do it in the main process.
        if not dist.is_initialized() or dist.get_rank() == 0:  
            if self.dataset_file.exists():
                if existing_ok:
                    self.dataset_file.unlink(missing_ok=True)
                else:
                    raise ValueError(f"File {self.dataset_file} already exists")

            # Make sure the output directory already exists
            self.dataset_file.parent.mkdir(parents=True, exist_ok=True)
            # Create the specified h5py file. The datasets
            # will be created dynamically as we trace the model.
            self.h5_file = h5py.File(self.dataset_file, 'w')
            self.datasets = {}
        else:
            self.h5_file = None
            self.datasets = None
    
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

    def _write_simple_type_dataset(self, name: str, variables: List[Any], iterations: List[int], ranks: List[int]):
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

    def _write_tensor_dataset(self, name: str, variables: List[np.ndarray], iterations: List[int], ranks: List[int]):
        """Write tensor data to HDF5 dataset with padding and configured settings."""
        tensor_dims = {len(x.shape) for x in variables}
        if len(tensor_dims) != 1:
            raise ValueError(f"Variable {name} has inconsistent tensor dimensions: {tensor_dims}")
        tensor_dims = tensor_dims.pop()
        
        # Calculate padding and stack
        tensor_sizes = [x.shape for x in variables]
        tensor_max_shape = [max(sizes) for sizes in zip(*tensor_sizes)]
        variables = np.stack(variables)

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
    
    def __call__(self, name, values, iterations, ranks):
        # Normalize all variables to numpy-compatible format
        normalized_values = [self._get_normalized_value(x) for x in values]

        # Determine type for writing
        if isinstance(normalized_values[0], (np.ndarray)):
            self._write_tensor_dataset(name, normalized_values, iterations, ranks)
        else:
            self._write_simple_type_dataset(name, normalized_values, iterations, ranks)
        
        values.clear()
        iterations.clear()
        ranks.clear()

    def close(self):
        if self.h5_file is not None:
            self.h5_file.flush()
            self.h5_file.close()