from abc import ABC
from uuid import uuid4
import typing


class Storable(ABC):
    """Object to serialize objects to make them easy to recover
    """

    def __init__(self):
        """Create the storable object
        """
        self._id = str(uuid4())

    @property
    def id(self) -> str:
        return self._id

    def load_state_dict(self, state_dict: typing.Dict):
        """

        Args:
            state_dict (typing.Dict): 
        """
        for k, v in self.__dict__.items():
            if isinstance(v, Storable):
                self.__dict__[k] = v.load_state_dict(state_dict[k])
            else:
                self.__dict__[k] = state_dict[k]
        
    def state_dict(self) -> typing.Dict:
        """

        Returns:
            typing.Dict: 
        """
        cur = {}

        for k, v in self.__dict__.items():
            if isinstance(v, Storable):
                cur[k] = v.state_dict()
            else:
                cur[k] = v
        return cur


T = typing.TypeVar('T')

class Ref(typing.Generic[T]):

    def __init__(self, val: T):
        self.val = val
