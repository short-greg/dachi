# 1st party
import typing
import string
import re
import pydantic
from enum import Enum


class _Types(Enum):

    UNDEFINED = 'UNDEFINED'
    WAITING = 'WAITING'


UNDEFINED = _Types.UNDEFINED
"""Constant for UNDEFINED. usage: value is UNDEFINED"""
WAITING = _Types.WAITING
"""Constant for WAITING when streaming. usage: value is WAITING"""


class _PartialFormatter(string.Formatter):
    """A partial formatter that does not require all keys to be specified"""

    def __init__(self):
        super().__init__()

    def format(self, format_string, *args, required: bool=True, **kwargs):
        """Format the string

        Args:
            format_string : The string to format
            required (bool, optional): Whether the key is required. Defaults to True.

        Returns:
            str: the formatted string
        """
        if args and kwargs:
            raise ValueError("Cannot mix positional and keyword arguments")

        if kwargs and required:
            difference = set(kwargs.keys()).difference(set(get_str_variables(format_string)))
            if difference:
                raise ValueError(f'Variables specified that are not in the string {difference}')
        self.args = args
        self.kwargs = kwargs
        return super().format(format_string)

    def get_value(self, key, args, kwargs):
        if isinstance(key, str):
            return self.kwargs.get(key, '{' + key + '}')
        if isinstance(key, int):
            return self.args[key] if key < len(self.args) else '{' + str(key) + '}'
        return super().get_value(key, args, kwargs)

    def __call__(self, format_string, *args, **kwargs):
        return self.format(format_string, *args, **kwargs)


def get_str_variables(format_string: str) -> typing.List[str]:
    """Get the variables in a string to format

    Args:
        format_string (str): The string to get variables for 

    Raises:
        ValueError: If the string has both positional and named
        variables

    Returns:
        typing.List[str]: The list of variables
    """
    has_positional = re.search(r'\{\d*\}', format_string)
    has_named = re.search(r'\{[a-zA-Z_]\w*\}', format_string)
    
    if has_positional and has_named:
        raise ValueError("Cannot mix positional and named variables")

    # Extract variables
    if has_positional:
        variables = [int(var) if var.isdigit() else None for var in re.findall(r'\{(\d*)\}', format_string)]
        if None in variables:
            variables = list(range(len(variables)))
    else:
        variables = re.findall(r'\{([a-zA-Z_]\w*)\}', format_string)
    
    return variables


str_formatter = _PartialFormatter() # Use to format a string with keys and values, does not require all keys specified


def escape_curly_braces(value: typing.Any, render: bool=False) -> str:
    """Escape curly braces for dictionary-like structures."""

    if isinstance(value, str):
        result = f'"{value}"'
        return result
    if isinstance(value, typing.Dict):
        items = ', '.join(f'"{k}": {escape_curly_braces(v)}' for k, v in value.items())
        return f"{{{{{items}}}}}"
    if isinstance(value, typing.List):
        return '[{}]'.format(', '.join(escape_curly_braces(v) for v in value))
    if render:
        return render(value)
    return str(value)


def unescape_curly_braces(value: typing.Any) -> str:
    """Invert the escaping of curly braces."""
    if isinstance(value, str):
        return value.replace('{{', '{').replace('}}', '}')
    return value


primitives = (bool, str, int, float, type(None))
"""a list of primitive types"""


def is_primitive(obj) -> bool:
    """Utility to check if a value is a primitive

    Args:
        obj: Value to check

    Returns:
        bool: If it is a "primitive"
    """
    return type(obj) in primitives


def generic_class(t: typing.TypeVar, idx: int=0) -> typing.Type:
    """Gets the generic type for a class assuming that it only has 
    one.

    Args:
        t (typing.TypeVar): The class to get the generic type for
        idx (int, optional): . Defaults to 0.

    Returns:
        typing.Type: the type specified by the generic class
    """
    return t.__orig_class__.__args__[idx]


def is_nested_model(
    pydantic_model_cls: typing.Type[pydantic.BaseModel]
) -> bool:
    """Helper function to check if it is a nested model

    Args:
        pydantic_model_cls (typing.Type[pydantic.BaseModel]): The class to check if it is a nested model

    Returns:
        bool: If it is a nested model
    """
    for field in pydantic_model_cls.model_fields.values():
        
        if isinstance(field.annotation, type) and issubclass(field.annotation, pydantic.BaseModel):
            return True
    return False


def get_member(obj, loc: str):
    """Get a member from an object recursively

    Args:
        obj : the object
        loc (str): the location as a string, use '.' to indicate sub objects

    Returns:
        Any: The member
    """
    locs = loc.split('.')
    for loc in locs:
        obj = getattr(obj, loc)
    return obj


def is_undefined(val) -> bool:
    """Returns true if the vlaue is undefined

    Args:
        val : The value to check

    Returns:
        bool: Whether the value is undefined or not
    """
    return val == UNDEFINED or val == WAITING


def coalesce(val1, default) -> typing.Any:
    """
    Returns the first value if it is not UNDEFINED, otherwise returns the default value.
    This function is useful for spawning new versions of a class and defining which parameters in the class to update.
    Args:
        val1: The primary value to check.
        default: The default value to return if val1 is UNDEFINED.
    Returns:
        The first value if it is not UNDEFINED, otherwise the default value.
    """
    return val1 if val1 is not UNDEFINED else default


def doc(obj) -> str:
    """utility to get the docstring for the object

    Args:
        obj: the object to get the docstring for

    Returns:
        str: The docstring for the object
    """
    d = obj.__doc__
    return d if d is not None else ''


class Args:
    """A class that wraps args"""
    
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def pydantic_v2():
    return int(pydantic.__version__.split(".")[0]) >= 2



"""
Decorator that turns a class into a singleton accessed via `Cls.obj`.

Compatible with normal classes *and* Pydantic BaseModel (v1 & v2).

Key points
----------
* `Cls()` raises TypeError with a helpful message.
* `Cls.obj` lazily constructs & returns the single instance.
* Each subclass automatically becomes its own singleton.
* Original validation / __init__ code *still runs* (important for Pydantic).
* Clear error chaining if the first construction fails.
"""

def singleton(cls):
    # already done?
    if getattr(cls, "__is_singleton__", False):
        return cls

    orig_meta = type(cls)                     # e.g. ModelMetaclass for Pydantic

    class _SingletonMeta(orig_meta):
        # block direct instantiation
        def __call__(self, *a, **kw):
            raise TypeError(
                f"{self.__name__} is a singleton. "
                f"Use {self.__name__}.obj instead of instantiating it."
            )

        # runs for every subclass
        def __init__(self, name, bases, ns, **kw):
            super().__init__(name, bases, ns, **kw)
            self._instance = None
            self.__is_singleton__ = True      # avoid re-decoration

        def _get_instance(self, *a, **kw):
            if self._instance is None:
                try:
                    # CALL THE ORIGINAL FACTORY so Pydantic validation happens
                    self._instance = super(_SingletonMeta, self).__call__(*a, **kw)
                except Exception as e:
                    # annotate & re-raise with original traceback
                    raise type(e)(
                        f"Error while creating the singleton instance of "
                        f"{self.__name__}: {e}"
                    ).with_traceback(e.__traceback__) from e
            return self._instance

        # make *every* subclass a singleton automatically
        def __init_subclass__(subcls, **kw):
            super().__init_subclass__(**kw)
            singleton(subcls)                 # no-op if already wrapped

    attrs = {
        "__module__" : cls.__module__,
        "__qualname__": cls.__qualname__,
        "__doc__"   : cls.__doc__,
        "__is_singleton__": True,
    }
    Wrapped = _SingletonMeta(cls.__name__, (cls,), attrs)

    # descriptor so `Wrapped.obj` is an attribute
    class _ObjDescriptor:
        def __get__(self, _, owner):
            return owner._get_instance()

    Wrapped.obj = _ObjDescriptor()
    return Wrapped

