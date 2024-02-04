import typing
from typing import Dict
from abc import abstractmethod, ABC
from ..base import Storable

import typing
from dataclasses import dataclass, field
from typing import Dict
from abc import abstractmethod, ABC
from ..base import Storable


T = typing.TypeVar("T")


S = typing.TypeVar('S', bound='Struct')

@dataclass
class Arg:

    name: str
    description: str = field(default="")


class Q(ABC, typing.Generic[T]):

    @abstractmethod
    def __call__(self, *args, **kwargs) -> T:
        raise NotImplementedError


class D(Q, typing.Generic[T]):

    def __init__(self, data):

        self._data = data

    def __call__(self) -> T:

        return self._data


class F(Q, typing.Generic[T]):

    def __init__(self, f, *args, **kwargs):

        self._f = f
        self._args = args
        self._kwargs = kwargs
    
    def __call__(self, *args, **kwargs) -> T:

        kwargs = {
            **self._kwargs,
            **kwargs
        }
        args = [*args, *self._args]
        return self._f(*args, **kwargs)


class R(Q, typing.Generic[T]):

    def __init__(self, data: 'Struct', name: str):
        
        self._data = data
        self._name = name

    def __call__(self) -> T:

        return self._data.get(self._name)


class Struct(Storable):

    @abstractmethod
    def as_text(self) -> str:
        pass

    @abstractmethod
    def as_dict(self) -> typing.Dict:
        pass

    @staticmethod
    def structure(text: str, heading: str=None, empty: float='Undefined'):

        text = text if text is not None else empty

        if heading is None:
            return f'{text}'
        
        return (
            f'{heading}\n'
            f'{text}'
        )

    def get(self, name: str) -> typing.Any:
        try:
            return getattr(self, name)
        except AttributeError:
            # TODO: Decide how to handle this
            print('No such attribute as ', name)
            return None

    def r(self, name: str) -> R:
        
        return R(self, name)

    @property
    def d(self) -> D:
        
        return D(self)
    
    def f(self, name: str, *args, **kwargs) -> F:
        return F(getattr(self, name), *args, **kwargs)

    def set(self, name: str, value):
        if name not in self.__dict__:
            raise KeyError(f'Key by name of {name} does not exist.')
        self.__dict__[name] = value
        return value

    def reset(self):
        pass


class DList(typing.List, Struct):
    
    def as_dict(self) -> Dict:
        return dict(enumerate(self))

    def as_text(self, heading: str=None) -> str:

        text = '\n'.join(
            [d.as_text() if isinstance(d, Struct) else d for d in self])
        if heading is not None:
            return (
                f'{heading}'
                f'{text}'
            )
        return text
    
    def spawn(self) -> 'DList':

        return DList(
            [x.spawn() if isinstance(x, Storable) else x for x in self]
        )

    def load_state_dict(self, state_dict: typing.Dict):
        """

        Args:
            state_dict (typing.Dict): 
        """
        for i, v in enumerate(self):
            if isinstance(v, Storable):
                self[i] = v.load_state_dict(state_dict[i])
            else:
                self[i] = state_dict[i]
        
    def state_dict(self) -> typing.Dict:
        """

        Returns:
            typing.Dict: 
        """
        cur = {}

        for i, v in enumerate(self):
            if isinstance(v, Storable):
                cur[i] = v.state_dict()
            else:
                cur[i] = v
        return cur

    def get(self, name: str) -> typing.Any:
        return self[name]

    def set(self, name: str, value) -> typing.Any:
        self[name] = value

        return value
    
    def as_dict_list(self) -> typing.List[typing.Dict]:

        return [
            item.as_dict() if isinstance(item, Struct) else {'value': item}
            for item in self
        ]


class DDict(typing.Dict, Struct):
    
    def as_dict(self) -> Dict:
        return self

    def as_text(self, heading: str=None) -> str:
        
        text = '\n'.join([f"{k}: {v}" for k, v in self.items()])
        if heading is not None:
            return (
                f'{heading}'
                f'{text}'
            )
        return text
    
    def spawn(self) -> 'DDict':

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
            typing.Dict: 
        """
        cur = {}

        for k, v in self.items():
            if isinstance(v, Storable):
                cur[k] = v.state_dict()
            else:
                cur[k] = v
        return cur
    
    # def get(self, name: str) -> typing.Any:
    #     return self.get(name)

    def set(self, name: str, value) -> typing.Any:
        self[name] = value

        return value


class Wrapper(Struct, typing.Generic[S]):

    def __init__(self, wrapped: S=None):
        super().__init__()
        self.wrapped = wrapped

    def as_text(self) -> str:
        if self.wrapped is None:
            return ''
        return self.wrapped.as_text()

    def as_dict(self) -> typing.Dict:
        if self.wrapped is None:
            return {}
        return self.wrapped.as_text()

    def reset(self):
        super().__init__()
        self.wrapped = None


# TODO: Reconsider the best way to do this
class Transfer(object):

    def __init__(self, q: Q, d: DDict, name: str):

        super().__init__()
        self.q = q
        self.d = d
        self.name = name

    def __call__(self):
        
        val = self.q()
        self.d.set(self.name, val)
        print(self.name, self.d.get(self.name), val)

