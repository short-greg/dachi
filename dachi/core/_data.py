
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


# TODO: Change to a dataclass

CALLBACK = typing.Callable[[typing.Any], typing.NoReturn]


class Blackboard(pydantic.BaseModel):
    """A blackboard is for sharing information
    across tasks
    """

    _callbacks: typing.Dict[str, typing.List[CALLBACK]] = pydantic.PrivateAttr(
        default_factory=dict
    )

    def register(self, key, callback) -> bool:
        """Register a callback to call on data updates

        Args:
            callback (function): The callback to register

        Returns:
            bool: True the callback was registered, False if already registered
        """
        if key not in self._callbacks:
            self._callbacks[key] = [callback]
            return True

        if callback in self._callbacks[key]:
            return False
        
        self._callbacks[key].append(callback)
        return True

    def unregister(self, key, callback) -> bool:
        """Unregister a callback to call on data updates

        Args:
            callback (function): The callback to unregister

        Returns:
            bool: True if the callback was removed, False if callback was not registered
        """
        if key not in self._callbacks:
            return False

        if callback not in self._callbacks[key]:
            return False
        
        self._callbacks[key].remove(callback)
        return True

    def r(self, key) -> 'MemberRetriever':
        """Get a retriever for the blackboard

        Args:
            key: The name of the key for the retriever

        Returns:
            Retriever: The retriever for the blackboard
        """
        return MemberRetriever(self, key, False)
    
    def model_copy(self, udpate, deep: bool=True) -> 'Blackboard':

        copy = super().model_copy(udpate, deep)
        copy._data = {**self._data}
        copy._member_callbacks = {*self._callbacks}
        copy._dict_callbacks = {**self._dict_callbacks}
        return copy


# TODO: UPdate

class MemberRetriever(object):
    """Use to retrieve data and set data in the blackboard
    """

    _blackboard: Blackboard = pydantic.PrivateAttr()
    _key: str = pydantic.PrivateAttr()

    def __init__(self, blackboard: Blackboard, key: str):
        """Create a retriever to retrieve data from the blackboard

        Args:
            blackboard (Blackboard): The blackboard to retrieve from
            key (str): The key to retrieve from
        """
        self._blackboard = blackboard
        self._key = key

    def get(self) -> typing.Any:
        """Get a value from the blackboard

        Returns:
            typing.Any: The value to get
        """
        return getattr(self._blackboard, self._key)
    
    def set(self, val) -> typing.Any:
        """Set the value the retriever points to

        Args:
            val: The value to set to

        Returns:
            typing.Any: The value to set
        """
        setattr(self._blackboard, self._key, val)
        return val

    def register(self, callback) -> bool:
        """Register a callback to call on data updates

        Args:
            callback (function): The callback to register

        Returns:
            bool: True the callback was registered, False if already registered
        """
        self._blackboard.register(self._key, callback)

    def unregister(self, callback) -> bool:
        """Unregister a callback to call on data updates

        Args:
            callback (function): The callback to unregister

        Returns:
            bool: True if the callback was removed, False if callback was not registered
        """
        self._blackboard.unregister(self._key, callback)

    @property
    def data(self) -> typing.Any:
        """Get the data shared

        Returns:
            typing.Any: The shared data
        """
        return self.get()

    @data.setter
    def data(self, val) -> typing.Any:
        """Update the data

        Args:
            data: The data to share

        Returns:
            typing.Any: The value for the data
        """
        self.set(val)
        return val


class DictRetriever(SharedBase):
    """Use to retrieve data and set data in the blackboard
    """

    _blackboard: Blackboard = pydantic.PrivateAttr()
    _key: str = pydantic.PrivateAttr()

    def __init__(self, blackboard: Blackboard, key: str):
        """Create a retriever to retrieve data from the blackboard

        Args:
            blackboard (Blackboard): The blackboard to retrieve from
            key (str): The key to retrieve from
        """
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

    def register(self, callback) -> bool:
        """Register a callback to call on data updates

        Args:
            callback (function): The callback to register

        Returns:
            bool: True the callback was registered, False if already registered
        """
        self._blackboard.register_item(self._key, callback)

    def unregister(self, callback) -> bool:
        """Unregister a callback to call on data updates

        Args:
            callback (function): The callback to unregister

        Returns:
            bool: True if the callback was removed, False if callback was not registered
        """
        self._blackboard.unregister_item(self._key, callback)

    @property
    def data(self) -> typing.Any:
        """Get the data shared

        Returns:
            typing.Any: The shared data
        """
        return self._blackboard[self._key]

    @data.setter
    def data(self, data) -> typing.Any:
        """Update the data

        Args:
            data: The data to share

        Returns:
            typing.Any: The value for the data
        """
        self._blackboard[self._key] = data
        return data


# class ItemQueue(pydantic.BaseModel, typing.Generic[T]):
    
#     items: typing.Dict[K, typing.List[T]] = pydantic.Field(
#         default_factory=dict
#     )

#     def write(self, val: T, key: K=None):

#         if key not in self.items:
#             self.items[key] = []
#         self.items[key].append(val)

#     def get(self, key: K=None) -> T | None:

#         if key not in self.items:
#             return None
#         if len(self.items[key]) == 0:
#             return None
        
#         return self.items[key].pop(0)
    
#     def count(self, **key) -> int:

#         if key not in self.items:
#             return 0
#         return len(self.items[key])


