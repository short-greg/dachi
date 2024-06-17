# 1st party
import typing
from abc import abstractmethod
from typing import get_type_hints
from typing_extensions import Self
import inspect
import json
from io import StringIO
import csv

from .._core import Struct, Str

T = typing.TypeVar('T', bound=Struct)

class Message(Struct):

    role: Str
    text: Str


class Doc(Struct):

    name: Str
    text: Str


class StructList(Struct, typing.Generic[T]):

    structs: typing.List[T]


class Chat(Struct):

    messages: typing.List[Message]

    def filter(self, roles: typing.Iterable[str]) -> 'Chat[Message]':

        roles = set(roles)
        
        return Chat(
            s for s in self._structs if s.role in roles
        )
