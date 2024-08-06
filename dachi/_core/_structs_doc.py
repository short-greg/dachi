# 1st party
import typing
from ._core import StructList
from ._process import Struct, StructModule

T = typing.TypeVar('T', bound=Struct)


class Message(StructModule):

    source: typing.Dict[str, str]
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
    def text(cls, role: str, text: str) -> 'Message':

        return Message(
            source={
                'role': role,
            },
            content={
                'text': text
            }
        )


class Doc(StructModule):

    name: str
    text: str


# TODO: Do I need this?

# class MessageList(StructList[Message]):

#     @property
#     def messages(self) -> typing.List[Message]:
#         return self.structs

#     def filter(self, roles: typing.Iterable[str]) -> 'MessageList[Message]':

#         roles = set(roles)
        
#         return MessageList(
#             s for s in self._structs if s.role in roles
#         )

