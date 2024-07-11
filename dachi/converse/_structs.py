# 1st party
import typing
from abc import abstractmethod, ABC

from dachi.process._core2 import UNDEFINED

from ..process import StructModule

from .._core import Struct, Str

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


class StructList(Struct, typing.Generic[T]):

    structs: typing.List[T]

    def __getitem__(self, key) -> typing.Any:
        
        return self.structs[key]
    
    def __setitem__(self, key, value) -> typing.Any:
        
        if key is None:
            self.structs.append(value)
        else:
            self.structs[key] = value
        return value


class MessageList(StructList[Message]):

    @property
    def messages(self) -> typing.List[Message]:
        return self.messages

    def filter(self, roles: typing.Iterable[str]) -> 'MessageList[Message]':

        roles = set(roles)
        
        return MessageList(
            s for s in self._structs if s.role in roles
        )


# class Lookup(StructModule, ABC):

#     score: float
#     content: typing.Dict[str, typing.Any]
#     meta: typing.Dict[str, typing.Any]

#     @abstractmethod
#     def forward(self, key: str, value: typing.Any = ..., get_struct: bool = False) -> typing.Any:
#         return super().forward(key, value, get_struct)


# class LookupList(StructList[Lookup]):

#     @property
#     def lookups(self) -> typing.List[Lookup]:
#         return self.structs

#     def filter(self, threshold: float):

#         return LookupList(
#             lookup for lookup in self.lookups 
#             if lookup.score >= threshold
#         )




# class ImageMessage(StructModule):

#     role: Str
#     text: Str
#     images: typing.List[Str]
