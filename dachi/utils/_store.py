import typing
from ._utils import UNDEFINED


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


def sub_dict(d: typing.Dict, key: str) -> typing.Dict:
    """
    Retrieve or initialize a nested dictionary within a given dictionary.
    Args:
        d (typing.Dict): The dictionary to operate on.
        key (str): The key pointing to the nested dictionary.
    Returns:
        typing.Dict: The nested dictionary associated with the given key.
    Raises:
        ValueError: If the key exists but does not point to a dictionary.
    """
    if key in d:
        if not isinstance(d[key], typing.Dict):
            raise ValueError(
                f'The field pointed to be {key} is not a dict.'
            )
    else:
        d[key] = {}
    return d[key]


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


def get_or_spawn(state: typing.Dict, child: str) -> typing.Dict:
    """Get a child or spawn it if it does not exist

    Args:
        state (typing.Dict): The state
        child (str): The name of the child

    Returns:
        typing.Dict: The dictionary for the child
    """
    if child not in state:
        state[child] = {}
    return state[child]