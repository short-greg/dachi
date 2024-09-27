import typing
from functools import reduce
from abc import ABC, abstractmethod



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


class SharedBase(ABC):
    """Allows for shared data between tasks
    """

    @abstractmethod
    def register(self, callback) -> bool:
        """Register a callback to call on data updates

        Args:
            callback (function): The callback to register

        Returns:
            bool: True the callback was registered, False if already registered
        """
        pass

    @abstractmethod
    def unregister(self, callback) -> bool:
        """Unregister a callback to call on data updates

        Args:
            callback (function): The callback to unregister

        Returns:
            bool: True if the callback was removed, False if callback was not registered
        """
        pass

    @property
    @abstractmethod
    def data(self) -> typing.Any:
        """Get the data shared

        Returns:
            typing.Any: The shared data
        """
        return self._data

    @data.setter
    @abstractmethod
    def data(self, data) -> typing.Any:
        """Update the data

        Args:
            data: The data to share

        Returns:
            typing.Any: The value for the data
        """
        pass

    @abstractmethod
    def get(self) -> typing.Any:
        pass
    
    @abstractmethod
    def set(self, value) -> typing.Any:
        pass


class Shared(SharedBase):
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

    def get(self) -> typing.Any:
        return self._data
    
    def set(self, value) -> typing.Any:
        self.data = value
        return value
    
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
        self._callbacks: typing.Dict[str, typing.List[typing.Callable[[typing.Any]]]] = {}

    def r(self, key) -> 'Retriever':
        """Get a retriever for the blackboard

        Args:
            key: The name of the key for the retriever

        Returns:
            Retriever: The retriever for the blackboard
        """
        return Retriever(self, key)

    def register(self, key, callback) -> bool:
        """Register a callback to call on data updates

        Args:
            callback (function): The callback to register

        Returns:
            bool: True the callback was registered, False if already registered
        """
        if key not in self._callbacks:
            self._callbacks[key] = [callable]
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


class Retriever(SharedBase):

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
