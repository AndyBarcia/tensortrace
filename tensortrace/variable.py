from typing import List, Callable, Any, Dict, Tuple, Literal, Union
from collections import defaultdict, namedtuple
import numpy as np
import torch.distributed as dist
from dataclasses import dataclass
from torch import Tensor
from copy import deepcopy


@dataclass
class LocalVariableResult:
    iterations: Union[list, np.ndarray]
    values: Union[list, np.ndarray]


@dataclass
class GlobalVariableResult:
    iterations: Union[list, np.ndarray]
    ranks: Union[list, np.ndarray]
    values: Union[list, np.ndarray]


def partial_path_matching(target_path: List[str], path: List[str]):
    if len(target_path) < len(path):
        return False
    for a,b in zip(target_path, path):
        if a != '*' and b != '' and a != b:
            return False
    return True


def exact_path_matching(target_path: List[str], path: List[str]):
    if len(target_path) != len(path):
        return False
    for a,b in zip(target_path, path):
        if a != '*' and b != '' and a != b:
            return False
    return True


class Variable:
    def __init__(
        self,
        paths: List[str],
        trace_interval: int = 1,
        gather_interval: int = 1,
        pre_save_callbacks: List[Callable[[str, Any, int], Any]] = [],
        pre_gather_callbacks: List[Callable[[str, LocalVariableResult], None]] = [],
        post_gather_callbacks: List[Callable[[str, GlobalVariableResult], None]] = [],
        post_trace_callbacks: List[Callable[[str, GlobalVariableResult], None]] = [],
        mode: Literal['eval', 'train'] = 'train'
    ):
        self.paths = paths
        self.trace_interval = trace_interval
        self.gather_interval = gather_interval
        self.pre_save_callbacks = pre_save_callbacks
        self.pre_gather_callbacks = pre_gather_callbacks
        self.post_gather_callbacks = post_gather_callbacks
        self.post_trace_callbacks = post_trace_callbacks
        self.mode = mode

        self.target_paths = [path.split('.') for path in self.paths]

        # Local storage for this process: {var_name: (iterations, values)}
        self.local_results: Dict[str, LocalVariableResult] = defaultdict(lambda: LocalVariableResult([], []))

        # Global aggregated storage: {var_name: ResultEntry}
        self.results: Dict[str, GlobalVariableResult] = defaultdict(lambda: GlobalVariableResult([], [], []))

        # Tracking data for change detection. This is only relevant to
        # this process' rank.
        self.last_result_data_ptr = {}
        self.last_result_version_counter = {}

    def save_variable_result(self, var_name: str, value: Any, current_iteration: int):
        # Callback tha can be used to pre-process values before they are saved.
        # For example, computing mean of batch dimensions to reduce memory.
        for callback in self.pre_save_callbacks:
            value = callback(var_name, value, current_iteration)

        if isinstance(value, Tensor):
            # If it's a tensor, save it only if it changed with respect to the
            # previous version of the tensor, using its version counter to
            # easily check for changes.
            # TODO: version counter doesn't appear to work with buffers?
            # Do something about it instead of saving every iteration.
            if (
                self.last_result_data_ptr.get(var_name) != id(value) or
                self.last_result_version_counter.get(var_name) != value._version or
                var_name not in self.local_results or
                current_iteration != self.local_results.get(var_name).iterations[-1]
            ):
                res = self.local_results[var_name]
                res.iterations.append(current_iteration)
                res.values.append(value.clone().detach().cpu())
                
                # Save tensor data pointer and version counter to keep track of future changes.
                self.last_result_data_ptr[var_name] = id(value)
                self.last_result_version_counter[var_name] = value._version
        elif isinstance(value, (int, float, complex, str, bytes, bool, type(None))):
            # If it's a simple type, just compare it to check if it changed
            if type(value) != type(self.last_result_data_ptr.get(var_name)) or value != self.last_result_data_ptr.get(var_name):
                res = self.local_results[var_name]
                res.iterations.append(current_iteration)
                res.values.append(deepcopy(value))
                
                # Save value to check later if it changed
                self.last_result_data_ptr[var_name] = value
        else:
            # If it's a complex type that is not a tensor, just use its ID to check
            # for changes in the object, and save it with deepcopy.
            if id(value) != self.last_result_data_ptr.get(var_name):
                res = self.local_results[var_name]
                res.iterations.append(current_iteration)
                res.values.append(deepcopy(value))
                    
                self.last_result_data_ptr[var_name] = id(value)

    def gather_results(self, rank: int, world_size: int):
        # Pre-gather processing
        for var_name, results in self.local_results.items():
            for callback in self.pre_gather_callbacks:
                callback(var_name, results)

        if world_size > 1:
            # Gather results from all processes to rank 0
            gathered_data = [{} for _ in range(world_size)]
            dist.gather_object(dict(self.local_results), gathered_data if rank == 0 else None, dst=0)

            if rank == 0:
                # Merge results from all ranks
                merged_data = defaultdict(lambda: GlobalVariableResult([], [], []))
                for rank_idx, rank_data in enumerate(gathered_data):
                    for var_name, results in rank_data.items():
                        res = merged_data[var_name]
                        res.iterations.extend(results.iterations)
                        res.values.extend(results.values)
                        res.ranks.extend([rank_idx] * len(results.iterations))

                # Post-gather processing
                for var_name, results in merged_data.items():
                    for callback in self.post_gather_callbacks:
                        callback(var_name, results)

                    # Append to global results
                    global_entry = self.results[var_name]
                    global_entry.iterations.extend(results.iterations)
                    global_entry.ranks.extend(results.ranks)
                    global_entry.values.extend(results.values)
        else:
            # Single-process case
            for var_name, results in self.local_results.items():
                result = GlobalVariableResult(
                    iterations = results.iterations, 
                    ranks = [0] * len(results.iterations), 
                    values = results.values
                )
                for callback in self.post_gather_callbacks:
                    callback(var_name, result)

                # Append to global results
                global_entry = self.results[var_name]
                global_entry.iterations.extend(result.iterations)
                global_entry.ranks.extend(result.iterations)
                global_entry.values.extend(result.values)

        # Clear local storage
        self.local_results.clear()