# 1st party
import typing
import string
import re


class _PartialFormatter(string.Formatter):

    def __init__(self):
        super().__init__()

    def format(self, format_string, *args, required: bool=True, **kwargs):
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


str_formatter = _PartialFormatter()


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


# class Args(object):
#     """Encapsulates args and kwargs into an object
#     """

#     def __init__(self, *args, **kwargs):
#         """Create the Args object
#         """
#         self.args = args
#         self.kwargs = kwargs


import typing
from functools import reduce


class F(object):
    """F is a functor that allows the user to set the args and kwargs
    """

    def __init__(self, f: typing.Callable[[], typing.Any], *args, **kwargs):
        """Create a functor

        Args:
            f (typing.Callable[[], typing.Any]): The function called
        """
        super().__init__()
        self.f = f
        self.args = args
        self.kwargs = kwargs

    @property
    def value(self) -> typing.Any:
        return self.f(*self.args, **self.kwargs)


def _is_function(f) -> bool:
    """ 
    Args:
        f: The value to check

    Returns:
        bool: whether f is a function
    """
    f_type = type(f)
    return f_type == type(_is_function) or f_type == type(hasattr)


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


def get_or_set(state: typing.Dict, key: str, val: typing.Any) -> typing.Any:
    """Use to get or set a value in a state dict

    Args:
        state (typing.Dict): The state dict to update
        key (str): The key to update
        val (typing.Any): The value to add

    Returns:
        typing.Any: 
    """
    if key not in state:
        state[key] = val
    return state[key]


class Shared:
    """Allows for shared data between tasks
    """

    def __init__(
        self, data: typing.Any=None, 
        default: typing.Any=None,
        callbacks: typing.List[typing.Callable[[typing.Any], None]]=None
    ):
        """Create shard data

        Args:
            data (typing.Any, optional): The data. Defaults to None.
            default (typing.Any, optional): The default value of the shared. When resetting, will go back to this. Defaults to None.
            callbacks (typing.List[typing.Callable[[typing.Any], None]], optional): The callbacks to call on update. Defaults to None.
        """

        self._default = default
        self._data = data if data is not None else default
        self._callbacks = callbacks or []

    def register(self, callback) -> bool:
        """Register a callback to call on data updates

        Args:
            callback (function): The callback to register

        Returns:
            bool: True the callback was registered, False if already registered
        """

        if callback in self._callbacks:
            return False
        
        self._callbacks.append(callback)
        return True

    def unregister(self, callback) -> bool:
        """Unregister a callback to call on data updates

        Args:
            callback (function): The callback to unregister

        Returns:
            bool: True if the callback was removed, False if callback was not registered
        """

        if callback not in self._callbacks:
            return False
        
        self._callbacks.remove(callback)
        return True

    @property
    def data(self) -> typing.Any:
        """Get the data shared

        Returns:
            typing.Any: The shared data
        """
        return self._data

    @data.setter
    def data(self, data) -> typing.Any:
        """Update the data

        Args:
            data: The data to share

        Returns:
            typing.Any: The value for the data
        """
        self._data = data
        for callback in self._callbacks:
            callback(data)
        return data

    def reset(self):
        self.data = self._default


# TODO: FINISH!

# x, y, z, ..., STOP
# current index

class Buffer(object):

    def __init__(self) -> None:
        
        self._buffer = []
        self._opened = True
        
    def add(self, *data):
        
        if not self._opened:
            raise RuntimeError(
                'The buffer is currently closed so cannot add data to it.'
            )

        self._buffer.extend(data)

    def open(self):
        self._opened = True

    def close(self):

        self._opened = False

    def is_open(self):

        return self._opened
        
    def it(self) -> 'BufferIter':
        return BufferIter(self)

    # def reset(self, opened: bool=True):
    #     """Resets new buffer. Note that any iterators will
    #     still use the old buffer
    #     """
    #     self._opened = opened
    #     self._buffer = []

    def __getitem__(self, key) -> typing.Union[typing.Any, typing.List]:

        if isinstance(key, slice):
            return self._buffer[key]
        if isinstance(key, typing.Iterable):
            return [self._buffer[i] for i in key]
        return self._buffer[key]
        
    def __len__(self) -> int:
        """

        Returns:
            int: 
        """
        return len(self._buffer)


class BufferIter(object):

    def __init__(self, buffer: Buffer) -> None:
        """

        Args:
            buffer (typing.List): 
        """
        self._buffer = buffer
        self._i = 0

    def start(self):
        """Whether the iterator is at the start of the buffer
        """
        return self._i == 0

    def end(self):

        return self._i >= (len(self._buffer) - 1)
    
    def is_open(self):

        return self._buffer.is_open()

    def read(self) -> typing.Any:
        
        if self._i < (len(self._buffer) - 1):
            self._i += 1
            return self._buffer[self._i - 1]
        raise StopIteration('Reached the end of the buffer')

    def read_all(self) -> typing.List:
        
        if self._i < len(self._buffer):
            i = self._i
            self._i = len(self._buffer)
            return self._buffer[i:]
            
        return []

    def read_reduce(self, f, init_val=None) -> typing.Any:

        data = self.read_all()
        return reduce(
            f, data, init_val
        )

    def read_map(self, f) -> typing.List:

        data = self.read_all()
        return map(
            f, data
        )


class Context(dict):

    def get_or_set(self, key, value):

        if key not in self:
            self[key] = value
            return value
        
        return self[key]


class ContextStorage(object):

    def __init__(self):

        object.__setattr__(self, '_data', {})

    @property
    def states(self) -> typing.Dict[str, Context]:
        return {**self._data}
    
    def remove(self, key):

        del self._data[key]
    
    def add(self, key):

        if key not in self._data:
            d = Context()
            self._data[key] = d
        
        return self._data[key]

    def __getattr__(self, key):

        if key not in self._data:
            d = Context()
            self._data[key] = d
        
        return self._data[key]


class ContextSpawner(object):

    def __init__(self, manager: ContextStorage, base_name: str):
        """_summary_

        Args:
            manager (StateManager): _description_
            base_name (str): _description_
        """
        self.manager = manager
        self.base_name = base_name

    def __getitem__(self, i: int) -> Context:

        name = f'{self.base_name}_{i}'
        return self.manager.add(name)




class Blackboard(object):
    """A blackboard is for sharing information
    across agents
    """

    def __init__(self):
        """The data for the blackboard
        """
        self._data = {}

    def r(self, key) -> 'Retriever':
        """Get a retriever for the blackboard

        Args:
            key: The name of the key for the retriever

        Returns:
            Retriever: The retriever for the blackboard
        """
        return Retriever(self, key)


    def __getitem__(self, key) -> typing.Any:
        """Get an item from the blackboard

        Args:
            key: The key to retrieve for

        Returns:
            typing.Any: The value for the key
        """
        return self._data[key]

    def __setitem__(self, key, val) -> typing.Any:
        """Set an item in the blackboard

        Args:
            key: The name of the item to set
            val: The value to set to

        Returns:
            typing.Any: The name of the 
        """
        self._data[key] = val
        return val


class Retriever(object):

    def __init__(self, blackboard: Blackboard, key: str):

        self._blackboard = blackboard
        self._key = key

    def get(self) -> typing.Any:
        """Get a value from the blackboard

        Returns:
            typing.Any: The value to get
        """
        return self._blackboard[self._key]
    
    def set(self, val) -> typing.Any:
        """Set the value the retriever points to

        Args:
            val: The value to set to

        Returns:
            typing.Any: The value to set
        """
        self._blackboard[self._key] = val
        return val

