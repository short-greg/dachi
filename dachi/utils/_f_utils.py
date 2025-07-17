import inspect
import typing
from typing import Iterator, AsyncIterator
from typing import Any, get_type_hints
import importlib, sys
from typing import Mapping, Any


import ast
import inspect


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
    """

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


def get_literal_return_values(func):
    """
    Retrieve all possible literal return values from a function/method.
    Raises ValueError if any return is not a literal.
    """
    source = inspect.getsource(func)
    tree = ast.parse(source)
    literals = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            value = node.value
            if isinstance(value, ast.Constant):
                literals.add(value.value)
            else:
                raise ValueError("Non-literal return detected")
    return literals


class NameResolutionError(RuntimeError):
    ...

def resolve_name(name: str,
                 namespace: Mapping[str, Any] | None = None,
                 search_sys_modules: bool = False) -> Any:
    """
    Resolve 'pkg.mod.Class', 'core.Class', or just 'Class'.

    Parameters
    ----------
    namespace
        Where to look first (e.g. globals()).
    search_sys_modules
        If True, and `name` has no dots **and** isn't found in `namespace`,
        search all currently‑loaded modules for an attribute of that name.
        Raises if multiple matches are found.

    Returns
    -------
    The live object, or raises NameResolutionError.
    """
    if namespace is None:
        namespace = globals()

    # 1) Fast path: already in the supplied namespace
    if name in namespace:
        return namespace[name]

    # 2) Dotted path?  Split and resolve left‑to‑right
    if '.' in name:
        head, *tail = name.split('.')

        obj = namespace.get(head)
        if obj is None:                          # not an alias; import it
            try:
                obj = importlib.import_module(head)
            except ModuleNotFoundError as e:
                raise NameResolutionError(f"Can't import '{head}'") from e

        for attr in tail:
            try:
                obj = getattr(obj, attr)
            except AttributeError as e:
                raise NameResolutionError(
                    f"'{'.'.join([head]+tail[:tail.index(attr)+1])}' "
                    f"has no attribute '{attr}'") from e
        return obj

    # 3) Undotted name and not in namespace
    if search_sys_modules:
        matches = [
            getattr(m, name) for m in sys.modules.values()
            if m and hasattr(m, name)
        ]
        if not matches:
            raise NameResolutionError(f"'{name}' not found anywhere")
        if len(matches) > 1:
            raise NameResolutionError(
                f"Ambiguous: '{name}' found in {len(matches)} modules")
        return matches[0]

    raise NameResolutionError(f"'{name}' not found in namespace")
