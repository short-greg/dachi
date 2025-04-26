import typing
from functools import reduce
from abc import ABC, abstractmethod
from ..msg._render import render, Renderable
from ..msg._messages import Msg
from ..utils import UNDEFINED
import pydantic
import typing


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


class SharedBase(pydantic.BaseModel, Renderable, ABC):
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

    def render(self) -> str:
        return render(self.get())


class Shared(SharedBase):
    """Allows for shared data between tasks
    """

    _data = pydantic.PrivateAttr()
    _callbacks = pydantic.PrivateAttr()
    _default = pydantic.PrivateAttr()

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
        super().__init__()
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
        """Get the value

        Returns:
            typing.Any: 
        """
        return self._data
    
    def set(self, value) -> typing.Any:
        """Set the shared value

        Args:
            value : Set the shared value

        Returns:
            typing.Any: The value the shared value was set to
        """
        self.data = value
        return value
    
    def reset(self):
        """Reset the shared value back to the default value
        """
        self.data = self._default


class Buffer(pydantic.BaseModel):
    """Create a buffer to add data to
    """

    _buffer = pydantic.PrivateAttr()
    _opened = pydantic.PrivateAttr(default=True)

    def __init__(self) -> None:
        """Create a buffer
        """
        super().__init__()
        self._buffer = []
        self._opened = True
        
    def add(self, *data):
        """Add data to the buffer

        Raises:
            RuntimeError: if the buffer is closed
        """
        
        if not self._opened:
            raise RuntimeError(
                'The buffer is currently closed so cannot add data to it.'
            )

        self._buffer.extend(data)

    def open(self):
        """Open the buffer
        """
        self._opened = True

    def close(self):
        """Close the buffer
        """
        self._opened = False

    def is_open(self) -> bool:
        """Get whether the buffer is open

        Returns:
            bool: Whether the buffer is open
        """
        return self._opened
        
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
            return self._buffer[idx]
        if isinstance(idx, typing.Iterable):
            return [self._buffer[i] for i in idx]
        return self._buffer[idx]
        
    def __len__(self) -> int:
        """Get the length of the buffer

        Returns:
            int: the length
        """
        return len(self._buffer)
    
    def get(self):
        return [*self._buffer]


class BufferIter(object):
    """An iterator to retrieve values from a Buffer
    """

    def __init__(self, buffer: Buffer) -> None:
        """Create a buffer iterator

        Args:
            buffer (Buffer): The buffer to iterate over
        """
        super().__init__()
        self._buffer = buffer
        self._i = 0

    def start(self):
        """Whether the iterator is at the start of the buffer
        """
        return self._i == 0

    def end(self) -> bool:
        """Get whether the buffer is at an end

        Returns:
            whether it is the end of the buffer iter
        """
        return self._i >= (len(self._buffer) - 1)
    
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
        if self._i < (len(self._buffer) - 1):
            self._i += 1
            return self._buffer[self._i - 1]
        raise StopIteration('Reached the end of the buffer')

    def read_all(self) -> typing.List:
        """_summary_

        Returns:
            typing.List: _description_
        """
        
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
    """Use to store state
    """

    def get_or_set(self, key, value):
        """Get or set a value in the context

        Args:
            key : The key for the value to set
            value : The value to set if not already set

        Returns:
            Any: the value specified by key
        """
        if key not in self:
            self[key] = value
            return value
        
        return self[key]

    def get_or_setf(self, key, f: typing.Callable[[], typing.Any]) -> typing.Any:
        """Adds a value to the dictionary if not already
        set.

        Args:
            d (typing.Dict): The dictionary
            key: The key
            f: The function to call to get the value

        Returns:
            typing.Any: The value specified by dictionary key
        """
        if key in self:
            return self[key]
        self[key] = f()
        return self[key]

    def call_or_set(self, key, value, f: typing.Callable[[typing.Any, typing.Any], typing.Any]) -> typing.Any:
        """Adds a value to the dictionary if not already
        set.

        Args:
            d (typing.Dict): The dictionary
            key: The key
            f: The function to call to get the value

        Returns:
            typing.Any: The value specified by dictionary key
        """
        if key not in self:
            self[key] = value
            return value
        self[key] = f(self[key], value)
        return self[key]


    def acc(
        self, key, value, init_val: str=''
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

        if key not in self:
            self[key] = init_val
        
        if value is not UNDEFINED:
            self[key] = self[key] + value
        return self[key]
    
    def __call__(self, key) -> 'ContextWriter':
        """Create a ContextWriter to set the value for
        a key in the context

        Args:
            key: The key to write to

        Returns:
            ContextWriter: The ContextWriter that writes
            the value
        """

        return ContextWriter(self, key)


class ContextWriter:
    """Use to write to the context
    """

    def __init__(self, context: Context, key: str):
        """_summary_

        Args:
            context (Context): _description_
            key (str): _description_
        """

        self.context = context
        self.key = key

    def __call__(self, value):

        self.context[self.key] = value


class ContextStorage(object):
    """Use to manage context storage such as spawning new
    contexts.
    """

    def __init__(self):
        """Create the context storage
        """
        object.__setattr__(self, '_data', {})
        self._data: typing.Dict

    @property
    def contexts(self) -> typing.Dict[str, Context]:
        """Retrieve all contexts

        Returns:
            typing.Dict[str, Context]: All contexts
        """
        return {**self._data}
    
    def clear(self):
        """Clear the entire context storage
        """
        self._data = {}

    def reset(self):
        """Clear the entire context storage
        """
        self.clear()
    
    def remove(self, key):
        """Remove a context

        Args:
            key: The name of the context to remove
        """
        del self._data[key]
    
    def add(self, key) -> 'Context':
        """Add context to the storage

        Args:
            key : The name of the context 

        Returns:
            Context: The spawned context
        """
        if key not in self._data:
            d = Context()
            self._data[key] = d
        
        return self._data[key]

    def __getattr__(self, key) -> 'Context':
        """Retrieve the context

        Args:
            key Retrieve the context

        Returns:
            Context: The created context
        """
        if key not in self._data:
            d = Context()
            self._data[key] = d
        
        return self._data[key]


class Blackboard(pydantic.BaseModel):
    """A blackboard is for sharing information
    across tasks
    """

    _data: typing.Dict[str, typing.Any] = pydantic.PrivateAttr(default_factory=dict)
    _member_callbacks: typing.Dict[str, typing.List[typing.Callable[[typing.Any], typing.NoReturn]]] = pydantic.PrivateAttr(
        default_factory=dict
    )
    _dict_callbacks: typing.Dict[str, typing.List[typing.Callable[[typing.Any], typing.NoReturn]]] = pydantic.PrivateAttr(
        default_factory=dict
    )

    # def __init__(
    #     self, 
    #     member_callbacks: typing.Dict=None, dict_callbacks: typing.Dict=None,
    #     **data
    # ):
    #     """The data for the blackboard
    #     """
    #     super().__init__(**data)
    #     self._member_callbacks = member_callbacks or {}
    #     self._dict_callbacks = dict_callbacks or {}

    def register_member(self, key, callback) -> bool:
        """Register a callback to call on data updates

        Args:
            callback (function): The callback to register

        Returns:
            bool: True the callback was registered, False if already registered
        """
        if key not in self._member_callbacks:
            self._member_callbacks[key] = [callback]
            return True

        if callback in self._member_callbacks[key]:
            return False
        
        self._member_callbacks[key].append(callback)
        return True

    def register_item(self, key, callback) -> bool:
        """Register a callback to call on data updates

        Args:
            callback (function): The callback to register

        Returns:
            bool: True the callback was registered, False if already registered
        """
        if key not in self._dict_callbacks:
            self._dict_callbacks[key] = [callback]

        if callback in self._dict_callbacks[key]:
            return False
        
        self._dict_callbacks[key].append(callback)
        return True

    def unregister_member(self, key, callback) -> bool:
        """Unregister a callback to call on data updates

        Args:
            callback (function): The callback to unregister

        Returns:
            bool: True if the callback was removed, False if callback was not registered
        """
        if key not in self._member_callbacks:
            return False

        if callback not in self._member_callbacks[key]:
            return False
        
        self._member_callbacks[key].remove(callback)
        return True

    def unregister_item(self, key, callback) -> bool:
        """Unregister a callback to call on data updates

        Args:
            callback (function): The callback to unregister

        Returns:
            bool: True if the callback was removed, False if callback was not registered
        """
        if key not in self._dict_callbacks:
            return False

        if callback not in self._dict_callbacks[key]:
            return False
        
        self._dict_callbacks[key].remove(callback)
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
        for callback in self._dict_callbacks.get(key, []):
            callback(val)
        
        return val

    def __setattr__(self, key, value):

        super().__setattr__(key, value)
        for callback in self._member_callbacks.get(key, []):
            callback(value)
        
        return value

    def d(self, key) -> 'ItemRetriever':
        """Get a retriever for the blackboard

        Args:
            key: The name of the key for the retriever

        Returns:
            Retriever: The retriever for the blackboard
        """
        return ItemRetriever(self, key, True)

    def m(self, key) -> 'MemberRetriever':
        """Get a retriever for the blackboard

        Args:
            key: The name of the key for the retriever

        Returns:
            Retriever: The retriever for the blackboard
        """
        return MemberRetriever(self, key, False)


class MemberRetriever(SharedBase):
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
        self._blackboard.register_member(self._key, callback)

    def unregister(self, callback) -> bool:
        """Unregister a callback to call on data updates

        Args:
            callback (function): The callback to unregister

        Returns:
            bool: True if the callback was removed, False if callback was not registered
        """
        self._blackboard.unregister_member(self._key, callback)

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


class ItemRetriever(SharedBase):
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



class ContextSpawner(object):
    """Use to Spawn contexts for your behaviors
    """

    def __init__(self, manager: ContextStorage, base_name: str):
        """Create a context Spawner specifying the Storage and the
        base name to use in spawning

        Args:
            manager (): The manager to use
            base_name (str): The base name of all contexts to spawn
        """
        object.__setattr__('manager', manager)
        object.__setattr__('base_name', base_name)

    def __getitem__(self, name) -> Context:
        """Get item specified by name

        Args:
            name: The context to get

        Returns:
            Context: The context
        """

        name = f'{self.base_name}_{name}'
        return self.manager.add(name)

    def __getattr__(self, name: int) -> Context:
        """Get item specified by name

        Args:
            name: The context to get

        Returns:
            Context: The context
        """
        name = f'{self.base_name}_{name}'
        return self.manager.add(name)


class Comm(object):
    """Use to have communication between two components 
    (two agents etc)
    """
    def __init__(self):
        """
        """
        self._processing = []

    def post(self, message: Msg, callback: typing.Callable[[], Msg]=None):
        """Post a message to be received

        Args:
            message (Msg): 
            callback (optional): . Defaults to None.
        """
        self._processing.append((message, callback))

    def get(self) -> Msg:
        """Get the current message to process

        Raises:
            RuntimeError: 

        Returns:
            Msg: The message to process
        """
        if self.empty():
            raise RuntimeError('No items to process')
        return self._processing[0][0]
    
    def empty(self) -> bool:
        """If nothing is being processed

        Returns:
            bool: Whether nothing is waiting for processing
        """
        return len(self._processing) == 0

    def respond(self, with_message: Msg):

        if self.empty():
            raise RuntimeError(
                'Cannot respond with a message '
            )
        
        callback = self._processing[0][1]
        if callback is not None:
            callback(with_message)
        self._processing.pop(0)
