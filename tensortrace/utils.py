from typing import List, Any
import collections.abc


def get_object_items(obj):
    """
    Extract keys/indices for different types of objects
    """
    # Dict-like objects
    if isinstance(obj, collections.abc.Mapping):
        return obj.items()
    
    # Indexable objects (lists, tuples, numpy arrays)
    if isinstance(obj, collections.abc.Iterable):
        return enumerate(iter(obj))
    
    # Objects with __dict__
    if hasattr(obj, '__dict__'):
        return obj.__dict__.items()
        
    # Named tuples
    if hasattr(obj, '_fields'):
        return obj._asdict().items()
    
    # Default to empty list if no keys found
    return []


def get_object_value(obj, key):
    """
    Safely retrieve value for a given key/index
    """
    try:
        # Try to index dict or list-like objects
        return obj[key]
    except Exception:        
        # Objects with __dict__ or _modules or _parameters
        if hasattr(obj, '__dict__'):
            if key in obj.__dict__:
                return obj.__dict__[key]
            if '_modules' in obj.__dict__ and key in obj.__dict__['_modules']:
                return obj.__dict__['_modules'][key]
            if '_parameters' in obj.__dict__ and key in obj.__dict__['_parameters']:
                return obj.__dict__['_parameters'][key]
        
        # Objects with __slots__
        if hasattr(obj, '__slots__') and key in obj.__slots__:
            return getattr(obj, key)
        
        # Named tuples
        if hasattr(obj, '_fields') and key in obj._fields:
            return getattr(obj, key)
        
        # Fallback
        return None


def extract_nested_values(data: Any, paths: List[List[str]]) -> dict:
    """
    Recursively extract values from a nested dictionary using a path with wildcard support.
    
    Args:
    data (dict): The nested dictionary to extract values from
    path (list): A list of keys/indices to navigate through the dictionary
               Use '*' as a wildcard to match any key
    
    Returns:
    dict: A dictionary with flattened paths as keys and corresponding values
    """
    paths = [key for key in paths if key]
    if not paths:
        return {}

    def recursive_extract(current_data, current_paths, result, full_path):   
        # Ignore paths that are already empty
        current_paths = [key for key in current_paths if key]
        
        # Base case: if we've reached the end of the path
        if not current_paths:
            result['.'.join(full_path)] = current_data
            return result
        
        # Current key/index to match
        for current_path in current_paths:
            current_key = current_path[0]
            
            # If current key is a wildcard, get everything
            if current_key == '*':
                for key,value in get_object_items(current_data):
                    sub_paths = [
                        path[1:] for path in current_paths 
                        if path[0] == '*' or path[0] == str(key)
                    ]
                    recursive_extract(
                        value, 
                        sub_paths, 
                        result, 
                        full_path + [str(key)]
                    )
                continue
            # Otherwise, try to get object.
            next_data = get_object_value(current_data, current_key)
            if next_data is not None:
                sub_paths = [
                    path[1:] for path in current_paths 
                    if path[0] == '*' or path[0] == current_key
                ]
                recursive_extract(
                    next_data, 
                    sub_paths, 
                    result, 
                    full_path + [current_key]
                )
                continue

        return result
    
    # Initialize and return the result
    return recursive_extract(data, paths, {}, [])
