import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel, DataParallel
from typing import List, Dict, DefaultDict, Any, Tuple
import sys
from copy import deepcopy
from functools import partial
from collections import defaultdict
import torch.distributed as dist

from .utils import extract_nested_values


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


class ModelTracer:
    def __init__(
        self, 
        model: nn.Module, 
        variables: List[str], 
        trace_interval = 1,
        gathering_interval = 1,
        eval_only = False,
        train_only = False,
    ):
        """
        Initialize ModelTracer as a context manager for tracing model variables
        
        Args:
            model (nn.Module): PyTorch model to trace
            variables (List[str]): List of variable paths to track
        """
        
        # If this model is DDP or DP get the underlying model so that all
        # variable paths still make sense without appending "module." to them.
        if isinstance(model, (DistributedDataParallel, DataParallel)):
            self.model = model.module
        else:
            self.model = model

        self.variables = variables
        self.trace_interval = trace_interval
        self.gathering_interval = gathering_interval
        self.eval_only = eval_only
        self.train_only = train_only

        assert not (self.eval_only and self.train_only), "You can't set both 'eval_only' and 'train_only' to True."

        # Paths of variables or modules we are interested in
        self.target_paths = [name.split('.') for name in self.variables] 
        
        # Map to easily convert modules to paths like [decoder, layers, 0]
        self.submodule_to_path = {
            mod: (name.split('.') if name != '' else []) 
            for name, mod in self.model.named_modules()
        }

        # Local process storage (reset after gathering), with tuples (iteration, value)
        self.local_results: Dict[str, List[Tuple[int, Any]]] = {}

        # Global aggregated storage (on rank 0), with tuples (iteration, rank, value) 
        self.results: DefaultDict[str, List[Tuple[int, int, Any]]] = defaultdict(list)

        # Determine rank of this tracer.
        self.rank = 0
        if dist.is_initialized():
            self.rank = dist.get_rank()

        # Tracking data for change detection. This is only relevant to
        # this process' rank.
        self.last_result_data_ptr = {}
        self.last_result_version_counter = {}

        # Keep track of the number of times the model was executed.
        self.current_iteration = 0

        # State tracking attributes
        self.original_trace = None
    
    def save_variable_result(self, var_name, value):
        if isinstance(value, torch.Tensor):
            # If it's a tensor, save it only if it changed with respect to the
            # previous version of the tensor, using its version counter to
            # easily check for changes.
            # TODO: version counter doesn't appear to work with buffers?
            # Do something about it instead of saving every iteration.
            if (
                self.last_result_data_ptr.get(var_name) != id(value) or
                self.last_result_version_counter.get(var_name) != value._version or
                self.current_iteration != self.local_results.get(var_name, [(0,None)])[-1][0]
            ):
                if var_name not in self.local_results:
                    self.local_results[var_name] = [(self.current_iteration, value.clone().detach().cpu())]
                else:
                    self.local_results[var_name].append((self.current_iteration, value.clone().detach().cpu()))
                
                # Save tensor data pointer and version counter to keep track of future changes.
                self.last_result_data_ptr[var_name] = id(value)
                self.last_result_version_counter[var_name] = value._version
        elif isinstance(value, (int, float, complex, str, bytes, bool, type(None))):
            # If it's a simple type, just compare it to check if it changed
            if type(value) != type(self.last_result_data_ptr.get(var_name)) or value != self.last_result_data_ptr.get(var_name):
                if var_name not in self.local_results:
                    self.local_results[var_name] = [(self.current_iteration, deepcopy(value))]
                else:
                    self.local_results[var_name].append((self.current_iteration, deepcopy(value)))
                
                # Save value to check later if it changed
                self.last_result_data_ptr[var_name] = value
        else:
            # If it's a complex type that is not a tensor, just use its ID to check
            # for changes in the object, and save it with deepcopy.
            if id(value) != self.last_result_data_ptr.get(var_name):
                if var_name not in self.local_results:
                    self.local_results[var_name] = [(self.current_iteration, deepcopy(value))]
                elif id(value) != self.last_result_data_ptr[var_name]:
                    self.local_results[var_name].append((self.current_iteration, deepcopy(value)))
                    
                self.last_result_data_ptr[var_name] = id(value)

    def _trace_calls(self, frame, event, arg, target_paths):
        """Trace function to capture local variables"""
        if event == "call":            
            # If this function doesn't accept self, we can't track it, so don't bother 
            # tracing its body.
            if 'self' not in frame.f_locals:
                return self.original_trace    
            
            # If 'self' is not a tracked module of our model, or we can't
            # even get the hash of the module, we shouldn't track it, nor its body.
            self_obj = frame.f_locals['self']
            try:
                module_path = self.submodule_to_path[self_obj]
            except Exception:
                return self.original_trace

            # Get all target paths that are subpaths of the current module.
            sub_target_paths = [
                target_path for target_path in target_paths 
                if partial_path_matching(target_path, module_path)
            ]

            # If we are not interesting in any variable of this function, don't keep tracing.
            # Otherwise, keep tracing the variables we are interested in.
            if not sub_target_paths:
                return self.original_trace
            return partial(self._trace_calls, target_paths=sub_target_paths)
        elif event == "line":
            # If this is a line of a function we are interested in, we should check
            # all its local variables to see if they match any of the target variables.
            func_name = frame.f_code.co_name
            self_obj = frame.f_locals['self']
            module_path = self.submodule_to_path[self_obj]

            sub_target_paths = [
                target_path[len(module_path)+1:] 
                for target_path in target_paths
                if len(module_path) < len(target_path) and target_path[len(module_path)] == func_name
            ]

            for var_name, var_value in extract_nested_values(frame.f_locals, sub_target_paths).items():
                qualified_path = module_path + [func_name, var_name]
                output_name = '.'.join(qualified_path)
                self.save_variable_result(output_name, var_value)
            
            # Keep on tracing this function.
            return partial(self._trace_calls, target_paths=target_paths)
        elif event == "return":
            # If this is the return of a function we are interested in, check if we should
            # save its output.
            func_name = frame.f_code.co_name
            self_obj = frame.f_locals['self']
            module_path = self.submodule_to_path[self_obj]

            qualified_path = module_path + [func_name]
            if any([exact_path_matching(target_path, qualified_path) for target_path in target_paths]):
                output_name = '.'.join(qualified_path)
                self.save_variable_result(output_name, arg)
            elif func_name == "forward" and any([exact_path_matching(target_path, module_path) for target_path in target_paths]):
                output_name = '.'.join(module_path)
                self.save_variable_result(output_name, arg)

            return
        return partial(self._trace_calls, target_paths=target_paths)

    def _gather_results(self):
        if dist.is_initialized():
            world_size = dist.get_world_size()
            # Gather results from all processes to rank 0
            gathered_results = [{} for _ in range(world_size)] # List[R; Dict[str, List[N; Tuple[iteration, value]]]]
            dist.gather_object(self.local_results, gathered_results if self.rank == 0 else None, dst=0)

            if self.rank == 0:
                # Temporary storage for all new (iteration, rank, value) entries per variable
                merged_entries = defaultdict(list) # Dict[str, List[N; tuple]]
                
                # Collect all entries across ranks
                for rank, rank_results in enumerate(gathered_results):
                    for var_name, var_entries in rank_results.items():
                        for it, val in var_entries:
                            merged_entries[var_name].append((it, rank, val))
                
                # Sort and append entries to global results
                for var_name, entries in merged_entries.items():
                    # Add the sorted entries to the global results.
                    self.results[var_name].extend(entries)
        else:
            # If this is not a distributed setup, just add the rank 0 identifier.
            for var_name, entries in self.local_results.items():
                for iteration, value in entries:
                    self.results[var_name].append((iteration, 0, value))
        
        # Reset local storage after gathering
        self.local_results.clear()

    def __enter__(self):
        # When entering the context, we need to wrap the model's forward function
        # to enable tracing. After exiting the context, we will restore the original 
        # function.
        self.original_forward = self.model.forward

        def wrapped_forward(*args, **kwargs):
            if (
                (self.eval_only and self.model.training) or
                (self.train_only and not self.model.training)
            ):
                # If we are not in a mode we are interested in, just call the original forward.
                # This is useful for example when we are training a model and we only want to trace
                # the evaluation phase.
                return self.original_forward(*args, **kwargs)

            self.current_iteration += 1

            # Determine if it is needed to trace in this iteration. If it is,
            # setup the tracing function.
            trace_needed = self.current_iteration % self.trace_interval == 0
            if trace_needed:
                original_trace = sys.gettrace()
                sys.settrace(partial(self._trace_calls, target_paths=self.target_paths))

            # Actually call the model, remembering to always reset the tracing 
            # function after finishing.
            try:
                output = self.original_forward(*args, **kwargs)
            finally:
                if trace_needed:
                    sys.settrace(original_trace)
            
            # Determine if it is time to gather the results of all ranks in rank0.
            gathering_needed = self.current_iteration % self.gathering_interval == 0
            if gathering_needed:
                self._gather_results()

            return output

        self.model.forward = wrapped_forward
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.forward = self.original_forward


def trace_model(
    model: nn.Module, 
    variables: List[str],
    *args, 
    **kwargs
) -> Dict[str, List[Any]]:
    with ModelTracer(model, variables) as tracer:
        output = model(*args, **kwargs)

    return output, tracer


class TracedModel(nn.Module):
    def __init__(
        self, 
        model: nn.Module,
        variables: List[str], 
        trace_interval = 1,
        eval_only = False,
        train_only = False,
    ):
        super().__init__()
        self.model = model
        self.tracer = ModelTracer(model, variables, trace_interval, eval_only, train_only)
    
    def forward(self, *args, **kwargs):
        with self.tracer:
            return self.model(*args, **kwargs)