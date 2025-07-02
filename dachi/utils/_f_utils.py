import inspect
import typing
from typing import Iterator, AsyncIterator
from typing import Any, get_type_hints


def is_async_function(func) -> bool:
    """Check if a function is asynchronous."""
    return inspect.iscoroutinefunction(func)


def get_return_type(func) -> Any:
    """Get the return type of a function."""
    type_hints = get_type_hints(func)
    return type_hints.get('return', None)


def is_generator_function(func) -> bool:
    """Check if a function is a generator."""
    return inspect.isasyncgenfunction(func) or inspect.isgeneratorfunction(func)


def is_iterator(func) -> bool:
    """Check if a function is a generator."""
    return isinstance(func, typing.Iterator)


def is_async_iterator(func) -> bool:
    """Check if a function is a generator."""
    return isinstance(func, typing.AsyncIterator)


def is_async_generator_function(func) -> bool:
    """Check if a function is a generator."""
    return inspect.isasyncgenfunction(func)


def get_iterator_type(func) -> Any:
    """
    Get the type of items yielded by an iterator function.
    Works for Iterator or AsyncIterator.
    """
    return_type = get_return_type(func)
    if return_type and hasattr(return_type, '__origin__'):
        if issubclass(return_type.__origin__, Iterator):
            return return_type.__args__[0]  # Type of the iterator
        elif issubclass(return_type.__origin__, AsyncIterator):
            return return_type.__args__[0]  # Type of the async iterator
    return None


def get_function_info(func: Any) -> typing.Dict:
    """_summary_

    Args:
        func (Any): 

    Returns:
        typing.Dict: Get information about the function
    """
    if not callable(func):
        raise ValueError("Provided argument is not callable.")
    
    # Get the function name
    name = func.__name__

    # Get the docstring
    docstring = inspect.getdoc(func)

    # Get the signature
    signature = inspect.signature(func)
    parameters = []
    for name, param in signature.parameters.items():
        parameter_info = {
            "name": name,
            "type": param.annotation if param.annotation is not inspect.Parameter.empty else None,
            "default": param.default if param.default is not inspect.Parameter.empty else None,
            "keyword_only": param.kind == inspect.Parameter.KEYWORD_ONLY
        }
        parameters.append(parameter_info)

    # Get the return type
    return_type = signature.return_annotation if signature.return_annotation is not inspect.Parameter.empty else None

    return {
        "name": name,
        "docstring": docstring,
        "parameters": parameters,
        "return_type": return_type
    }
