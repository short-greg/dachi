from abc import abstractmethod, ABC, abstractproperty
import typing

class DataHook(ABC):

    def __init__(self, name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def __call__(self, data: 'Data', prev_value):
        pass


class MetaHook(ABC):

    def __init__(self, hook: DataHook):
        super().__init__()
        self._hook = hook
    
    @property
    def name(self) -> str:
        return self._hook.name

    def __call__(self, data: 'Data', prev_value):
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
        if self._check_f is not None:
            valid, msg = self._check_f(value)
            if not valid:
                raise ValueError(msg)

    def update(self, value) -> typing.Any:
        self.validate(value)
        prev_value = self._value
        self._value = value
        self._hooks(self, prev_value)
        return value


class Synched(IData):

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
        return self._name
    
    @property
    def synched_name(self) -> str:
        return self._base_data.name

    @property
    def value(self) -> typing.Any:
        return self._base_data.value

    def update(self, value) -> typing.Any:
        return self._base_data.update(value)

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
        
        if key not in self._data:
            self.add(key, value)
        else:
            self._data[key].update(value)

    def get(self, key: str) -> typing.Any:

        result = self._data.get(key)
        if result is None:
            return None
        return result.value
    
    def __contains__(self, key: str) -> bool:
        return key in self._data
    
    def __getitem__(self, key: str) -> typing.Any:
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