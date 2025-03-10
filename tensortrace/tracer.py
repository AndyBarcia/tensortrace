import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel, DataParallel
from typing import List, Dict, DefaultDict, Any, Tuple, Callable, Optional, Union
import torch.nn.functional as F
import sys
from copy import deepcopy
from functools import partial
from collections import defaultdict, namedtuple
import torch.distributed as dist

from .utils import extract_nested_values
from .variable import Variable, LocalVariableResult, GlobalVariableResult


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
        variables: Union[List[Variable], Variable], 
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

        if isinstance(variables, Variable):
            self.variables = [variables]
        else:
            self.variables = variables
        
        # Map to easily convert modules to paths like [decoder, layers, 0]
        self.submodule_to_path = {
            mod: (name.split('.') if name != '' else []) 
            for name, mod in self.model.named_modules()
        }

        # Determine rank of this tracer.
        self.rank = 0
        if dist.is_initialized():
            self.rank = dist.get_rank()

        # Keep track of the number of times the model was executed separately
        # for training and evaluation.
        self.current_training_iteration = 0
        self.current_evaluation_iteration = 0

        # State tracking attributes
        self.original_trace = None

    def _trace_calls(self, frame, event, arg, target_paths: Dict[Variable, List[str]]):
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
            sub_paths = {
                var: [
                    path for path in paths if partial_path_matching(path, module_path)
                ] for var,paths in target_paths.items()
            }
            # Ignore whole variables that can't match this call.
            sub_paths = { var: paths for var,paths in sub_paths.items() if paths }

            # If we are not interesting in any variable of this function, don't keep tracing.
            # Otherwise, keep tracing the variables we are interested in.
            if not sub_paths:
                return self.original_trace
            return partial(self._trace_calls, target_paths=sub_paths)
        elif event == "line":
            # If this is a line of a function we are interested in, we should check
            # all its local variables to see if they match any of the target variables.
            func_name = frame.f_code.co_name
            self_obj = frame.f_locals['self']
            module_path = self.submodule_to_path[self_obj]

            # Iterate each variable
            for var, paths in target_paths.items():
                sub_paths = [
                    path[len(module_path)+1:] 
                    for path in paths
                    if len(module_path) < len(path) and path[len(module_path)] == func_name
                ]

                for var_name, var_value in extract_nested_values(frame.f_locals, sub_paths).items():
                    qualified_path = module_path + [func_name, var_name]
                    output_name = '.'.join(qualified_path)
                    iteration = self.current_training_iteration if var.mode == 'train' else self.current_evaluation_iteration
                    var.save_variable_result(output_name, var_value, iteration)
            
            # Keep on tracing this function.
            return partial(self._trace_calls, target_paths=target_paths)
        elif event == "return":
            # If this is the return of a function we are interested in, check if we should
            # save its output.
            func_name = frame.f_code.co_name
            self_obj = frame.f_locals['self']
            module_path = self.submodule_to_path[self_obj]
            
            qualified_path = module_path + [func_name]
            
            # Iterate each variable
            for var, paths in target_paths.items():
                if any([exact_path_matching(path, qualified_path) for path in paths]):
                    output_name = '.'.join(qualified_path)
                    iteration = self.current_training_iteration if var.mode == 'train' else self.current_evaluation_iteration
                    var.save_variable_result(output_name, arg, iteration)
                elif func_name == "forward" and any([exact_path_matching(path, module_path) for path in paths]):
                    output_name = '.'.join(module_path)
                    iteration = self.current_training_iteration if var.mode == 'train' else self.current_evaluation_iteration
                    var.save_variable_result(output_name, arg, iteration)

            return
        return partial(self._trace_calls, target_paths=target_paths)

    def __enter__(self):
        # When entering the context, we need to wrap the model's forward function
        # to enable tracing. After exiting the context, we will restore the original 
        # function.
        self.original_forward = self.model.forward

        def wrapped_forward(*args, **kwargs):
            # Get the variables we are interested in based on the model training mode,
            # and on the iteration we are currently in.
            traced_vars: List[Variable] = []
            gathered_vars: List[Variable] = []
            if self.model.training:
                self.current_training_iteration += 1
                for var in self.variables:
                    if var.mode == 'train':
                        if self.current_training_iteration % var.trace_interval == 0:
                            traced_vars.append(var)
                        if self.current_training_iteration % var.gather_interval == 0:
                            gathered_vars.append(var)
            else:
                self.current_evaluation_iteration += 1
                for var in self.variables:
                    if var.mode == 'eval':
                        if self.current_evaluation_iteration % var.trace_interval == 0:
                            traced_vars.append(var)
                        if self.current_evaluation_iteration % var.gather_interval == 0:
                            gathered_vars.append(var)
            
            if traced_vars:
                original_trace = sys.gettrace()

                target_paths = { var: var.target_paths for var in self.variables }
                sys.settrace(partial(self._trace_calls, target_paths=target_paths))

                # Actually call the model, remembering to always reset the tracing 
                # function after finishing.
                try:
                    output = self.original_forward(*args, **kwargs)
                finally:
                    sys.settrace(original_trace)
            else:
                # If we are not in a mode we are interested in, just call the original forward.
                # This is useful for example when we are training a model and we only want to trace
                # the evaluation phase.
                return self.original_forward(*args, **kwargs)

            if dist.is_initialized():
                world_size = dist.get_world_size()
            else:
                world_size = 1
            for var in gathered_vars:
                var.gather_results(self.rank, world_size)

            return output

        self.model.forward = wrapped_forward
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.forward = self.original_forward

        # Final callbacks to, for example, stack tensors.
        for var in self.variables:
            for callback in var.post_trace_callbacks:
                for name, res in var.results.items():
                    callback(name, res)


def trace_model(
    model: nn.Module, 
    variables: List[Variable],
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
        variables: List[Variable],
    ):
        super().__init__()
        self.model = model
        self.tracer = ModelTracer(model, variables)
    
    def forward(self, *args, **kwargs):
        with self.tracer:
            return self.model(*args, **kwargs)


def stack_tensors(name, results: Union[LocalVariableResult, GlobalVariableResult]):
    if results.values:
        results.values = torch.stack(results.values)
    else:
        results.values = torch.tensor([])

    """
    if not isinstance(iterations[0], torch.Tensor):
        stacked_iterations = torch.tensor(iterations)
        iterations.clear()
        iterations.append(stacked_iterations)
    if ranks and not isinstance(ranks[0], torch.Tensor):
        stacked_ranks = torch.tensor(ranks)
        ranks.clear()
        ranks.append(stacked_ranks)
    """


def stack_padded_tensors(name, results: Union[LocalVariableResult, GlobalVariableResult]):
    # Process 'values' to ensure all tensors are padded to the same shape
    if results.values:
        max_dims = max(t.ndim for t in results.values)
        padded_tensors = []
        # Expand each tensor with leading dimensions to match max_dims
        for t in results.values:
            while t.ndim < max_dims:
                t = t.unsqueeze(0)
            padded_tensors.append(t)
        # Determine the maximum size for each dimension
        max_sizes = [max(t.shape[i] for t in padded_tensors) for i in range(max_dims)]
        # Pad each tensor to the max sizes
        padded_values = []
        for t in padded_tensors:
            pad = []
            # Iterate dimensions from last to first to build padding tuple
            for i in reversed(range(max_dims)):
                pad_needed = max_sizes[i] - t.shape[i]
                pad.extend([0, pad_needed])
            padded_t = F.pad(t, pad)
            padded_values.append(padded_t)
        results.values = torch.stack(padded_values)
    else:
        results.values = torch.tensor([])  # Handle empty 'values' if necessary

    # Process 'iterations' if not already tensors
    """
    if iterations and not isinstance(iterations[0], torch.Tensor):
        stacked_iterations = torch.tensor(iterations)
        iterations.clear()
        iterations.append(stacked_iterations)

    # Process 'ranks' if provided and not already tensors
    if ranks and not isinstance(ranks[0], torch.Tensor):
        stacked_ranks = torch.tensor(ranks)
        ranks.clear()
        ranks.append(stacked_ranks)
    """