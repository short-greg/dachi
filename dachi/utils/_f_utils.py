import inspect
from typing import Iterator, AsyncIterator


def to_async_function(func) -> bool:
    """Check if a function is asynchronous."""
    return inspect.iscoroutinefunction(func)
from typing import Any, get_type_hints


def get_return_type(func) -> Any:
    """Get the return type of a function."""
    type_hints = get_type_hints(func)
    return type_hints.get('return', None)


def is_generator_function(func) -> bool:
    """Check if a function is a generator."""
    return inspect.isgeneratorfunction(func)


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
