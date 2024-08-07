# 1st party
import typing
from ._core import StructList
from ._process import Struct, StructModule

T = typing.TypeVar('T', bound=Struct)


class Message(StructModule):

    data: typing.Dict[str, str]

    def __getitem__(self, key: str):
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.data:
            return self.data[key]
        raise KeyError(f'{key}')

    def __setitem__(self, key: str, value):
        if hasattr(self, key):
            setattr(self, key, value)
        if key in self.data:
            self.data[key] = value
        raise KeyError(f'{key}')


class TextMessage(Message):

    def __init__(self, role: str, text: str) -> 'Message':

        super().__init__(
            data={
                'role': role,
                'text': text
            }
        )

# TODO: Improve document similar to message

class Doc(StructModule):

    name: str
    text: str
