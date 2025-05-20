from abc import ABC, abstractmethod
import typing
from uuid import uuid4

import pydantic

from ._render import render



class BaseSpec(pydantic.BaseModel, ABC):
    pass


class BuildContext(object):
    pass


class BaseComponent(ABC):

    __spec__ = None
    
    def to_schema(cls) -> typing.Dict:
        pass

    def from_spec(
        cls, 
        data: typing.Dict, 
        context: BuildContext
    ) -> typing.Self:
        pass

    def to_spec(self, save_private: bool) -> typing.Dict:
        pass

    def state_dict(self) -> typing.Dict:
        pass

    def load_state_dict(self, state_dict: typing.Dict) -> typing.Self:
        pass
