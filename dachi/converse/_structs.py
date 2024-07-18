# 1st party
import typing
from abc import abstractmethod, ABC
from .._core import StructList
from dachi._core._core2 import UNDEFINED

from .._core import Struct, Str, StructModule

T = typing.TypeVar('T', bound=Struct)


class Message(StructModule):

    role: Str
    content: typing.Dict[str, str]

    def __getitem__(self, key: str):
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.content:
            return self.content[key]
        raise KeyError(f'{key}')

    def __setitem__(self, key: str, value):
        if hasattr(self, key):
            setattr(self, key, value)
        if key in self.content:
            self.content[key] = value
        raise KeyError(f'{key}')
    
    @classmethod
    def create(cls, role: Str, **kwargs) -> 'Message':

        return Message(
            role=role,
            content=kwargs
        )


class Doc(StructModule):

    name: Str
    text: Str


class MessageList(StructList[Message]):

    @property
    def messages(self) -> typing.List[Message]:
        return self.messages

    def filter(self, roles: typing.Iterable[str]) -> 'MessageList[Message]':

        roles = set(roles)
        
        return MessageList(
            s for s in self._structs if s.role in roles
        )

