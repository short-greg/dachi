# 1st party
import typing
from ._process import Struct, StructModule

import pydantic

T = typing.TypeVar('T', bound=Struct)


class Media:

    descr: str
    data: str


Content = typing.Union[Media, str, typing.List[typing.Union[Media, str]]]


class Message(StructModule):

    source: str
    data: typing.Dict[str, Content]

    def __getitem__(self, key: str):
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.data:
            return self.data[key]
        raise KeyError(f'{key}')

    def __setitem__(self, key: str, value: Content):
        if hasattr(self, key):
            setattr(self, key, value)
        if key in self.data:
            self.data[key] = value
        raise KeyError(f'{key}')


class TextMessage(Message):

    def __init__(self, source: str, text: str) -> 'Message':

        super().__init__(
            data={
                'source': source,
                'text': text
            }
        )

# TODO: Improve document similar to message

# Can extend the dialog to provide more functionality


class Dialog(Struct):

    messages: typing.List[Message] = pydantic.Field(default_factory=list)

    def __iter__(self) -> typing.Iterator[Message]:

        for message in self.messages:
            yield message

    def __add__(self, other: 'Dialog'):

        return Dialog(
            self.messages + other.messages
        )

    def __getitem__(self, idx) -> 'Dialog':

        return Dialog(
            messages=self.messages[idx]
        )

    def __setitem__(self, idx, message) -> 'Dialog':

        self.messages[idx] = message

    def insert(self, index: int, message: Message):

        self.messages.insert(index, message)

    def pop(self, index: int):

        self.messages.pop(index)

    def remove(self, message: Message):

        self.messages.remove(message)

    def append(self, message: Message):

        self.messages.append(message)

    def extend(self, dialog: typing.Union['Dialog', typing.List[Message]]):

        if isinstance(dialog, Dialog):
            dialog = dialog.messages
        
        self.messages.extend(dialog)

    def source(self, source: str, text: typing.Optional[str]=None, _index: typing.Optional[int]=None, _replace: bool=False, **kwargs):
        if len(kwargs) == 0 and text is not None:
            message = TextMessage(source, text)
        elif text is not None:
            message = Message(text=text, **kwargs)
        elif text is None:
            message = Message(**kwargs)
        else:
            raise ValueError('No message has been passed. The text and kwargs are empty')

        if _index is None:
            self.messages.append(message)
        elif not _replace:
            self.messages.insert(_index, message)
        else:
            self.messages[_index] = message

    def user(self, text: str=None, **kwargs):
        self.source('user', text, **kwargs)

    def assistant(self, text: str=None, **kwargs):
        self.source('assistant', text, **kwargs)

    def system(self, text: str=None, **kwargs):
        self.source('system', text, **kwargs)

    def instruct(self, text: str=None, **kwargs):
        to_replace = len(self.messages) > 0 and self.messages[0].source != 'system'
        self.source('system', text, 0, to_replace, **kwargs)


    # def squash(self, source: str='system') -> 'Message':

    #     return '\n'.join(
    #         message.render()
    #         for message in self.messages
    #     )

    # message: 1) 

# class Doc(StructModule):

#     name: str
#     text: str
