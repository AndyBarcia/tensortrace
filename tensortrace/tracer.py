import torch
import torch.nn as nn
from typing import List, Dict, Any
import sys
from copy import deepcopy
from functools import partial

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
        eval_only = False,
        train_only = False,
    ):
        """
        Initialize ModelTracer as a context manager for tracing model variables
        
        Args:
            model (nn.Module): PyTorch model to trace
            variables (List[str]): List of variable paths to track
        """
        self.model = model
        self.variables = variables
        self.trace_interval = trace_interval
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

        # Results of saved variables, including tensors and other values
        self.results = {}
        self.results_iterations = {}

        # Tracking data for change detection
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
                self.current_iteration != self.results_iterations.get(var_name, [0])[-1]
            ):
                if var_name not in self.results:
                    self.results[var_name] = [value.clone().detach()]
                    self.results_iterations[var_name] = [self.current_iteration]
                else:
                    self.results[var_name].append(value.clone().detach())
                    self.results_iterations[var_name].append(self.current_iteration)
                
                # Save tensor data pointer and version counter to keep track of future changes.
                self.last_result_data_ptr[var_name] = id(value)
                self.last_result_version_counter[var_name] = value._version
        elif isinstance(value, (int, float, complex, str, bytes, bool, type(None))):
            # If it's a simple type, just compare it to check if it changed
            if type(value) != type(self.last_result_data_ptr.get(var_name)) or value != self.last_result_data_ptr.get(var_name):
                if var_name not in self.results:
                    self.results[var_name] = [deepcopy(value)]
                    self.results_iterations[var_name] = [self.current_iteration]
                else:
                    self.results[var_name].append(deepcopy(value))
                    self.results_iterations[var_name].append(self.current_iteration)
                
                # Save value to check later if it changed
                self.last_result_data_ptr[var_name] = value
        else:
            # If it's a complex type that is not a tensor, just use its ID to check
            # for changes in the object, and save it with deepcopy.
            if id(value) != self.last_result_data_ptr.get(var_name):
                if var_name not in self.results:
                    self.results[var_name] = [deepcopy(value)]
                    self.results_iterations[var_name] = [self.current_iteration]
                elif id(value) != self.last_result_data_ptr[var_name]:
                    self.results[var_name].append(deepcopy(value))
                    self.results_iterations[var_name].append(self.current_iteration)
                    
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
            trace_needed = self.current_iteration % self.trace_interval == 0
            if trace_needed:
                original_trace = sys.gettrace()
                sys.settrace(partial(self._trace_calls, target_paths=self.target_paths))

            try:
                output = self.original_forward(*args, **kwargs)
            finally:
                if trace_needed:
                    sys.settrace(original_trace)

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