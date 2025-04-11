import typing
from ..utils import UNDEFINED


def get_or_set(d: typing.Dict, key, value) -> typing.Any:
    """Adds a value to the dictionary if not already
    set.

    Args:
        d (typing.Dict): The dictionary
        key: The key
        value: The value to add

    Returns:
        typing.Any: The value specified by dictionary key
    """
    if key in d:
        return d[key]
    d[key] = value
    return value


def get_or_setf(d: typing.Dict, key, f: typing.Callable[[], typing.Any]) -> typing.Any:
    """Adds a value to the dictionary if not already
    set.

    Args:
        d (typing.Dict): The dictionary
        key: The key
        f: The function to call to get the value

    Returns:
        typing.Any: The value specified by dictionary key
    """
    if key in d:
        return d[key]
    d[key] = f()
    return d[key]


def call_or_set(d: typing.Dict, key, value, f: typing.Callable[[typing.Any, typing.Any], typing.Any]) -> typing.Any:
    """Adds a value to the dictionary if not already
    set.

    Args:
        d (typing.Dict): The dictionary
        key: The key
        f: The function to call to get the value

    Returns:
        typing.Any: The value specified by dictionary key
    """
    if key not in d:
        d[key] = value
        return value
    d[key] = f(d[key], value)
    return d[key]


def acc(
    d: typing.Dict, key, value, init_val: str=''
) -> typing.Any:
    """Adds a value to the dictionary if not already
    set.

    Args:
        d (typing.Dict): The dictionary
        key: The key
        f: The function to call to get the value

    Returns:
        typing.Any: The value specified by dictionary key
    """

    if key not in d:
        d[key] = init_val
    
    if value is not UNDEFINED:
        d[key] = d[key] + value
    return d[key]
