# 1st party
from abc import abstractmethod, ABC, abstractproperty
import typing


class DataHook(ABC):
    """A hook that will be called if the data is updated
    """

    def __init__(self, name: str, *args, **kwargs):
        """Create a hook to be called when the data is updated

        Args:
            name (str): The name of the  hook
        """
        super().__init__(*args, **kwargs)
        self._name = name
    
    @property
    def name(self) -> str:
        """
        Returns:
            str: Retrieve the name of the hook
        """
        return self._name

    @abstractmethod
    def __call__(self, data: 'Data', prev_value):
        pass


class MetaHook(ABC):
    """
    """

    def __init__(self, hook: DataHook):
        """Create a hook that will be called when another is called

        Args:
            hook (DataHook): The hook to be called on
        """
        super().__init__()
        self._hook = hook
    
    @property
    def name(self) -> str:
        """
        Returns:
            str: Retrieve the name of the hook
        """
        return self._hook.name

    def __call__(self, data: 'Data', prev_value):
        """Call the hook

        Args:
            data (Data): The data updated
            prev_value : The previous value of the data
        """
        self._hook(data, prev_value)


class IData(ABC):
    """Define the base class for data
    """

    def __init__(self, name: str):
        """Create the data

        Args:
            name (str): The name of the data
        """
        super().__init__()
        self._name = name

    @abstractmethod
    def register_hook(self, hook):
        pass

    @property
    def name(self) -> str:
        """
        Returns:
            str: The name of the data
        """
        return self._name

    @abstractproperty
    def value(self) -> typing.Any:
        pass

    @abstractmethod
    def update(self, value) -> typing.Any:
        pass


class Data(IData):
    """Definitions of data for the behavior tree
    """

    def __init__(self, name: str, value: typing.Any, check_f=None):
        """Create data for the behavior tree

        Args:
            name (str): The name of the data
            value (typing.Any): The value for the data
            check_f (optional): Validation function if necessary. Defaults to None.
        """
        super().__init__(name)
        self._check_f = check_f
        self.validate(value)
        self._value = value
        self._hooks = CompositeHook(name)

    def register_hook(self, hook: DataHook):
        """Register a hook for when the data is updated

        Args:
            hook: The hook to call
        """
        self._hooks.append(hook)

    def has_hook(self, hook: DataHook):
        """Get whether 

        Args:
            hook (DataHook): _description_

        Returns:
            _type_: _description_
        """
        return hook in self._hooks

    def remove_hook(self, hook: DataHook):
        """Get whether 

        Args:
            hook (DataHook): _description_

        """
        self._hooks.remove(hook)
    
    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> typing.Any:
        return self._value
    
    def validate(self, value):
        """Check that the data is valid

        Args:
            value: The data stored by Data

        Raises:
            ValueError: If the validation fails
        """
        if self._check_f is not None:
            valid, msg = self._check_f(value)
            if not valid:
                raise ValueError(msg)

    def update(self, value) -> typing.Any:
        """Update the value of the data

        Args:
            value: The value of the data

        Returns:
            typing.Any: The updated value
        """
        self.validate(value)
        prev_value = self._value
        self._value = value
        self._hooks(self, prev_value)
        return value


class Synched(IData):
    """Data that refers to another data source
    """

    def __init__(self, name: str, base_data: Data):

        super().__init__(name)
        self._base_data = base_data
        self._hooks: typing.Dict[DataHook, MetaHook] = {}

    def register_hook(self, hook: DataHook):
        """Register a hook for when the data is updated. Will be
        attached to the base data

        Args:
            hook (DataHook): The hook to call when updating
        """
        meta = MetaHook(hook)
        self._hooks[hook] = meta
        self._base_data.register_hook(meta)

    @property
    def name(self) -> str:
        """The name of the data

        Returns:
            str: The name of the synched data
        """
        return self._name
    
    @property
    def synched_name(self) -> str:
        """The name of the value synched to

        Returns:
            str: The name of the base data
        """
        return self._base_data.name

    @property
    def value(self) -> typing.Any:
        """
        Returns:
            typing.Any: The value of the base data
        """
        return self._base_data.value

    def update(self, value) -> typing.Any:
        """
        Args:
            value: The value to update with

        Returns:
            typing.Any: The updated value
        """
        return self._base_data.update(value)

    def has_hook(self, hook: DataHook) -> bool:
        """Get whether the hook has been set

        Args:
            hook (DataHook): The hook to check

        Returns:
            bool: Whether the hook has been set
        """
        return hook in self._hooks

    def remove_hook(self, hook: DataHook):
        """Get whether 

        Args:
            hook (DataHook): _description_

        """
        meta = self._hooks[hook]
        self._base_data.remove_hook(meta)
        del self._hooks[hook]
        

class CompositeHook(DataHook, list):
    """A hook that will run multiple hooks
    """

    def __init__(self, name: str, hooks: typing.List[DataHook]=None):
        """Create a composite hook to run multiple hook

        Args:
            name (str): The name of the hook
            hooks (typing.List[DataHook], optional): The hooks to run. Defaults to None.
        """
        super().__init__(name, hooks or [])

    def __call__(self, data: 'Data', prev_value):
        """Run the hooks

        Args:
            data (Data): The data that was updated
            prev_value (typing.Any): The previous value for the data
        """
        for hook in self:
            hook(data, prev_value)


class DataStore(object):
    """
    """

    def __init__(self) -> None:
        super().__init__()
        self._data: typing.Dict[str, Data] = {}

    def add(self, key: str, value: typing.Any, check_f=None) -> None:
        """Add data to the data store

        Args:
            key (str): The name of the data
            value (typing.Any): The value for the data
            check_f (typing.Callable[[typing.Any], bool], optional): Add data to the datastore . Defaults to None.

        Raises:
            ValueError: 
        """
        if key in self._data:
            raise ValueError(f'Data with key {key} has already been added to the data store')
        self._data[key] = Data(key, value, check_f)
    
    def add_sync(self, key: str, data: Data):
        if key in self._data:
            raise ValueError()
        self._data[key] = Synched(key, data)
    
    def update(self, key: str, value: typing.Any):
        """Update the DataStore specified by key

        Args:
            key (str): The key for the value to update
            value (typing.Any): The value to update 
        """
        if key not in self._data:
            self.add(key, value)
        else:
            self._data[key].update(value)

    def get(self, key: str, default=None) -> typing.Any:
        """Get a value from the get()

        Args:
            key (str): The key for the 
            default (optional): The default value. Defaults to None.

        Returns:
            typing.Any: The value specified by key or default
        """
        result = self._data.get(key)
        if result is None:
            return default
        return result.value

    def get_or_set(self, key: str, default) -> typing.Any:
        """Get the value specified by key or if it does not exist set it

        Args:
            key (str): The key to retrieve for
            default: The value to set if the key has not been specified. Defaults to None.

        Returns:
            typing.Any: The value specified by key
        """
        try:
            result = self._data[key]
        except KeyError:
            self[key] = default
            return default
        return result.value

    def __contains__(self, key: str) -> bool:
        """
        Args:
            key (str): The key to check if exists

        Returns:
            bool: Whether the key is in the data
        """
        return key in self._data
    
    def __getitem__(self, key: str) -> typing.Any:
        """Get the value specified by key

        Args:
            key (str): The key to retrieve for

        Returns:
            typing.Any: The value
        """
        return self._data[key].value
    
    def __setitem__(self, key: str, value) -> typing.Any:

        self.update(key, value)
        return value

    def register_hook(self, key: str, hook: DataHook):
        """Register a hook for when the data is updated. Will be
        attached to the base data

        Args:
            hook (DataHook): The hook to call when updating
        """
        self._data[key].register_hook(hook)
    
    def has_hook(self, key: str, hook: DataHook):
        """
        Args:
            key : The key to check if exists
            hook (DataHook): The hook to 

        Returns:
            bool: Whether the hook data specified by key has the hook
        """
        return self._data[key].has_hook(hook)

    def remove_hook(self, key: str, hook: DataHook):
        """Get whether 

        Args:
            hook (DataHook): _description_

        """
        self._data[key].remove_hook(hook)

    def reset(self):
        """Clear out the data in the data store
        """
        self._data.clear()


def callback(cb, contents, store: DataStore):
    """Callback if the callback is not None.
    Will set the storage if the callback is a string
    Otherwise call it if callback is a function

    Args:
        cb (function): The callback to call
        contents: the contents of the to call
        store (DataStore): The store to set if callback is a 
            string
    """
    if cb is None:
        return
    if isinstance(cb, str):
        store[cb] = contents
    else:
        cb(contents)
