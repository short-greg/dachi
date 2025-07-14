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


# 1st party
import typing
from functools import reduce

# 3rd party
import pydantic
import typing
# local

T = typing.TypeVar("T")
K = bool | str | None | int | float


class Buffer(pydantic.BaseModel):
    """Create a buffer to add data to
    """
    buffer: typing.List
    opened: bool = True
        
    def add(self, *data):
        """Add data to the buffer

        Raises:
            RuntimeError: if the buffer is closed
        """
        
        if not self.opened:
            raise RuntimeError(
                'The buffer is currently closed so cannot add data to it.'
            )

        self.buffer.extend(data)

    def open(self):
        """Open the buffer
        """
        self.opened = True

    def close(self):
        """Close the buffer
        """
        self.opened = False

    def is_open(self) -> bool:
        """Get whether the buffer is open

        Returns:
            bool: Whether the buffer is open
        """
        return self.opened
    
    def clear(self):
        """Remove all elements in the buffer
        """
        self.buffer.clear()
        
    def it(self) -> 'BufferIter':
        return BufferIter(self)

    def __getitem__(self, idx) -> typing.Union[typing.Any, typing.List]:
        """Get a value from the buffer

        Args:
            idx: The index to to retrieve

        Returns:
            typing.Union[typing.Any, typing.List]: The retrieved data
        """
        if isinstance(idx, slice):
            return self.buffer[idx]
        if isinstance(idx, typing.Iterable):
            return [self.buffer[i] for i in idx]
        return self.buffer[idx]
        
    def __len__(self) -> int:
        """Get the length of the buffer

        Returns:
            int: the length
        """
        return len(self.buffer)
    
    def get(self):
        return [*self.buffer]


class BufferIter(pydantic.BaseModel):
    """An iterator to retrieve values from a Buffer
    """

    buffer: Buffer
    i: int = 0

    def start(self):
        """Whether the iterator is at the start of the buffer
        """
        return self.i == 0

    def end(self) -> bool:
        """Get whether the buffer is at an end

        Returns:
            whether it is the end of the buffer iter
        """
        return self.i >= (len(self._buffer) - 1)
    
    def is_open(self) -> bool:
        """Get if the buffer is open

        Returns:
            bool: Whether the buffer is open
        """
        return self._buffer.is_open()

    def read(self) -> typing.Any:
        """Read the next value in the buffer

        Raises:
            StopIteration: If the buffer is at an end

        Returns:
            typing.Any: The next value in the buffer
        """
        if self.i < (len(self._buffer) - 1):
            self.i += 1
            return self._buffer[self.i - 1]
        raise StopIteration('Reached the end of the buffer')

    def read_all(self) -> typing.List:
        """_summary_

        Returns:
            typing.List: _description_
        """
        
        if self.i < len(self._buffer):
            i = self.i
            self.i = len(self._buffer)
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
