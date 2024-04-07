# 1st party
import typing
from typing import Dict
from dataclasses import dataclass, field
from abc import abstractmethod, ABC

# local
from ...store import Storable

T = typing.TypeVar("T")
S = typing.TypeVar('S', bound='Struct')


@dataclass
class Arg:

    name: str
    description: str = field(default="")


class Retrieve(ABC, typing.Generic[T]):
    """
    Retrieves a value from a storage class
    """
    @abstractmethod
    def __call__(self, *args, **kwargs) -> T:
        raise NotImplementedError


class DRetrieve(Retrieve, typing.Generic[T]):
    """Retrieves data
    """
    def __init__(self, data):
        """
        Args:
            data: The data to 'retrieve'
        """
        self._data = data

    def __call__(self) -> T:

        return self._data


class FRetrieve(Retrieve, typing.Generic[T]):
    """Retriever that calls a function

    """
    def __init__(self, f, *args, **kwargs):
        """Define the function to call and its
        args

        Args:
            f: The function
        """
        self._f = f
        self._args = args
        self._kwargs = kwargs
    
    def __call__(self, *args, **kwargs) -> T:
        """Will call the function

        Returns:
            T: The output of the function
        """
        kwargs = {
            **self._kwargs,
            **kwargs
        }
        args = [*args, *self._args]
        return self._f(*args, **kwargs)


class SRetrieve(Retrieve, typing.Generic[T]):
    """Retriever for accessing data in a struct
    """

    def __init__(self, data: 'Struct', name: str):
        """Define the struct retriever

        Args:
            data (Struct): The struct to retrieve from
            name (str): The name of the variable to
             retrieve
        """
        self._data = data
        self._name = name

    def __call__(self) -> T:
        """
        Returns:
            T: The data in the struct
        """
        return self._data.get(self._name)


class Struct(Storable):
    """An object used for storing data
    """

    @abstractmethod
    def as_text(self) -> str:
        """
        Returns:
            str: Retrieve the object as text
        """
        pass

    @abstractmethod
    def as_dict(self) -> typing.Dict:
        """
        Returns:
            typing.Dict: Retrieve the object as a dict
        """
        pass

    @staticmethod
    def structure(
        text: str, heading: str=None, 
        empty: float='Undefined'
    ) -> str:
        """

        Args:
            text (str): The text
            heading (str, optional): The heading for text. Defaults to None.
            empty (float, optional): Text to use if empty. Defaults to 'Undefined'.

        Returns:
            str: Return the struct as structured text
        """
        text = text if text is not None else empty

        if heading is None:
            return f'{text}'
        
        return (
            f'{heading}\n'
            f'{text}'
        )

    def get(self, name: str) -> typing.Any:
        """
        Args:
            name (str): The name of the attribute to retrieve

        Returns:
            typing.Any: The retrieved attribute
        """
        try:
            return getattr(self, name)
        except AttributeError:
            # TODO: Decide how to handle this
            print('No such attribute as ', name)
            return None

    def r(self, name: str) -> SRetrieve:
        """
        Args:
            name (str): The name of the value to retrieve

        Returns:
            SRetrieve: A retriever for a value in the struct
        """
        return SRetrieve(self, name)

    @property
    def d(self) -> DRetrieve:
        """
        Returns:
            DRetrieve: A retriever for this struct
        """
        return DRetrieve(self)
    
    def f(self, name: str, *args, **kwargs) -> FRetrieve:
        """Create an FRetriever

        Args:
            name (str): The name of the func to execute

        Returns:
            FRetrieve: A function retriever for the struct
        """
        return FRetrieve(getattr(self, name), *args, **kwargs)

    def set(self, name: str, value) -> typing.Any:
        """Set a value in the struct

        Args:
            name (str): The name of the value to set
            value : The value to set

        Returns:
            typing.Any: The value set
        """
        if name not in self.__dict__:
            raise KeyError(f'Key by name of {name} does not exist.')
        self.__dict__[name] = value
        return value

    def reset(self):
        """Reset the struct
        """
        pass


class DList(typing.List, Struct):
    """Create a list of data
    """
    def as_dict(self) -> Dict:
        """
        Returns:
            Dict: Return the list as a struct
        """
        return dict(enumerate(self))

    def as_text(self, heading: str=None) -> str:
        """
        Args:
            heading (str, optional): The heading to use for the list. Defaults to None.

        Returns:
            str: Return the list as_text
        """
        text = '\n'.join(
            [d.as_text() if isinstance(d, Struct) else d for d in self])
        if heading is not None:
            return (
                f'{heading}'
                f'{text}'
            )
        return text
    
    def spawn(self) -> 'DList':
        """
        Returns:
            DList: Spawn the list into a new list
        """
        return DList(
            [x.spawn() if isinstance(x, Storable) else x for x in self]
        )

    def load_state_dict(self, state_dict: typing.Dict):
        """
        Args:
            state_dict (typing.Dict): The state dict to load
        """
        for i, v in enumerate(self):
            if isinstance(v, Storable):
                self[i] = v.load_state_dict(state_dict[i])
            else:
                self[i] = state_dict[i]
        
    def state_dict(self) -> typing.Dict:
        """
        Returns:
            typing.Dict: The state dict
        """
        cur = {}

        for i, v in enumerate(self):
            if isinstance(v, Storable):
                cur[i] = v.state_dict()
            else:
                cur[i] = v
        return cur

    def get(self, name: str) -> typing.Any:
        """
        Args:
            name (str): The name of the value to get

        Returns:
            typing.Any: The value to get
        """
        return self[name]

    def set(self, name: str, value) -> typing.Any:
        """Set the value

        Args:
            name (str): The name of the value to set
            value: The value to set

        Returns:
            typing.Any: 
        """
        self[name] = value

        return value
    
    def as_dict_list(self) -> typing.List[typing.Dict]:
        """
        Returns:
            typing.List[typing.Dict]: A list of dicts
        """
        return [
            item.as_dict() if isinstance(item, Struct) else {'value': item}
            for item in self
        ]


class DDict(typing.Dict, Struct):
    
    def as_dict(self) -> Dict:
        """
        Returns:
            Dict: Return as a dictionary
        """
        return self

    def as_text(self, heading: str=None) -> str:
        """
        Args:
            heading (str, optional): The heading for the dict. Defaults to None.

        Returns:
            str: Return the dict as a string
        """
        text = '\n'.join([f"{k}: {v}" for k, v in self.items()])
        if heading is not None:
            return (
                f'{heading}'
                f'{text}'
            )
        return text
    
    def spawn(self) -> 'DDict':
        """
        Returns:
            DDict: Return the 
        """
        return DDict(
            {k: x.spawn() if isinstance(x, Storable) else x for k, x in self.items()}
        )

    def load_state_dict(self, state_dict: typing.Dict):
        """

        Args:
            state_dict (typing.Dict): 
        """
        for k, v in enumerate(self.items()):
            if isinstance(v, Storable):
                self[k] = v.load_state_dict(state_dict[k])
            else:
                self[k] = state_dict[k]
        
    def state_dict(self) -> typing.Dict:
        """
        Returns:
            typing.Dict: Get as a state dict
        """
        cur = {}

        for k, v in self.items():
            if isinstance(v, Storable):
                cur[k] = v.state_dict()
            else:
                cur[k] = v
        return cur

    def get(self, name: str) -> typing.Any:
        """
        Args:
            name (str): The name of the value to get

        Returns:
            typing.Any: The value to get
        """
        return self[name]

    def set(self, name: str, value) -> typing.Any:
        self[name] = value

        return value


class Wrapper(Struct, typing.Generic[S]):
    """Wrap a struct
    """

    def __init__(self, wrapped: S=None):
        """
        Args:
            wrapped (S, optional): The struct to wrap. Defaults to None.
        """
        super().__init__()
        self.wrapped = wrapped

    def as_text(self) -> str:
        """
        Returns:
            str: The wrapped value as text
        """
        if self.wrapped is None:
            return ''
        return self.wrapped.as_text()

    def as_dict(self) -> typing.Dict:
        """

        Returns:
            typing.Dict: The wrapped value as a dict
        """
        if self.wrapped is None:
            return {}
        return self.wrapped.as_text()

    def reset(self):
        """Reset the wrapped value
        """
        super().__init__()
        self.wrapped = None


# TODO: Reconsider the best way to do this
class Transfer(object):

    def __init__(self, q: Retrieve, d: DDict, name: str):
        """_summary_

        Args:
            q (Retrieve): 
            d (DDict): 
            name (str): 
        """
        super().__init__()
        self.q = q
        self.d = d
        self.name = name

    def __call__(self):
        
        val = self.q()
        self.d.set(self.name, val)
        print(self.name, self.d.get(self.name), val)

