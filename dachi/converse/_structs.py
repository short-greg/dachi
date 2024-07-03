# 1st party
import typing
from abc import abstractmethod
from typing import get_type_hints
from typing_extensions import Self
import inspect
import json
from io import StringIO
import csv
from ..process import StructModule

from .._core import Struct, Str

T = typing.TypeVar('T', bound=Struct)


class Message(StructModule):

    role: Str
    text: Str


class ImageMessage(StructModule):

    role: Str
    text: Str
    images: typing.List[Str]


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


class Chat(StructModule):

    messages: typing.List[Message]

    def filter(self, roles: typing.Iterable[str]) -> 'Chat[Message]':

        roles = set(roles)
        
        return Chat(
            s for s in self._structs if s.role in roles
        )
    
    def __getitem__(self, key) -> typing.Any:
        return super().__getitem__(key)
    
    def __setitem__(self, key, value) -> typing.Any:
        return super().__setitem__(key, value)
