# 1st party
import typing
from abc import ABC, abstractmethod

from ._core import (
    Struct, Renderable
)
from ._utils import generic_class

# 3rd party
import pydantic


S = typing.TypeVar('S', bound='Struct')
T = typing.TypeVar('T', bound=Struct)


class StructList(Struct, typing.Generic[S]):
    """
    """

    structs: typing.List[S]

    def __getitem__(self, key) -> typing.Any:
        """

        Args:
            key (_type_): 

        Returns:
            typing.Any: 
        """
        return self.structs[key]
    
    def __setitem__(self, key: typing.Optional[int], value: S) -> typing.Any:
        """Set a value in the 

        Args:
            key (str): The key for the value to set
            value : The value to set

        Returns:
            S: the value that was set
        """
        if key is None:
            self.structs.append(value)
        else:
            self.structs[key] = value
        return value
    
    @classmethod
    def load_records(cls, records: typing.List[typing.Dict]) -> 'StructList[S]':
        """Load the struct list from records

        Args:
            records (typing.List[typing.Dict]): The list of records to load

        Returns:
            StructList[S]: The list of structs
        """
        structs = []
        struct_cls: typing.Type[Struct] = generic_class(S)
        for record in records:
            structs.append(struct_cls.load(record))
        return StructList[S](
            structs=structs
        )


class Description(Struct, Renderable, ABC):
    """Provide context in the prompt template
    """
    name: str = pydantic.Field(description='The name of the description.')

    # @abstractmethod
    # def update(self, **kwargs) -> Self:
    #     pass

    @abstractmethod
    def render(self) -> str:
        pass



class Ref(Struct):
    """Reference to another description.
    Useful when one only wants to include the 
    name of a description in part of the prompt
    """
    desc: Description

    @property
    def name(self) -> str:
        """Get the name of the ref

        Returns:
            str: The name of the ref
        """
        return self.desc.name

    def render(self) -> str:
        """Generate the text rendering of the ref

        Returns:
            str: The name for the ref
        """
        return self.desc.name



class Media:

    descr: str
    data: str


Content = typing.Union[Media, str, typing.List[typing.Union[Media, str]]]


class Message(Struct):

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
            source=source,
            data={
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
