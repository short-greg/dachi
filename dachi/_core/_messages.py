# 1st party
import typing
import asyncio
from abc import ABC, abstractmethod
from typing import Self
import typing

from ._core import (
    Cue, Reader, 
    render, NullRead
)
from ._core import Renderable
from .._core import is_renderable, render

# 3rd party
import pydantic
from pydantic import Field

from pydantic import BaseModel, Field
import typing
import pandas as pd
import numpy as np


# Add in delta to the message

class Message(pydantic.Model, Renderable):
    """A prompt that consists of a single message to send the AI
    """

    source: str # The source of the messsage (user, assistant etc)
    data: typing.Dict[str, typing.Any] # The contents of the message

    def __init__(self, source: str, **data):

        super().__init__(source=source, data=data)

    def __getitem__(self, key: str) -> typing.Any:
        """Get an item from the message

        Args:
            key (str): The key to retrieve for

        Raises:
            KeyError: 

        """
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.data:
            return self.data[key]
        raise KeyError(f'{key}')

    def __setitem__(self, key: str, value: typing.Any):
        """Set an item in the message

        Args:
            key (str): The key to set for
            value (typing.Any): The value to set

        Raises:
            KeyError: An error
        """
        if hasattr(self, key):
            setattr(self, key, value)
        if key in self.data:
            self.data[key] = value
        raise KeyError(f'{key}')

    # def process_response(self, response: 'AIResponse') -> 'AIResponse':
    #     """Process the response of the Model

    #     Args:
    #         response (AIResponse): The response to process

    #     Returns:
    #         AIResponse: The updated response
    #     """
    #     return response

    # def prompt(self, model: 'AIModel', **kwarg_overrides) -> 'AIResponse':
    #     """prompt the ai model with the message

    #     Args:
    #         model (AIModel): The model to prmopt

    #     Returns:
    #         AIResponse: the response
    #     """
    #     return model(self, **kwarg_overrides)

    # def instruct(
    #     self, instruct: 'Instruct', 
    #     ai_model: 'AIModel', 
    # ) -> AIResponse:
    #     """Instruct the AI

    #     Args:
    #         instruct (Instruct): The cue to use
    #         ai_model (AIModel): The AIModel to use
    #         ind (int, optional): The index to set to. Defaults to 0.
    #         replace (bool, optional): Whether to replace at the index if already set. Defaults to True.

    #     Returns:
    #         AIResponse: The output from the AI
    #     """
    #     # TODO: Think if I want to remove this 
    #     # circular dependency
    #     dialog = Dialog(
    #         [self]
    #     )
    #     return dialog.instruct(
    #         instruct, ai_model, 0, False
    #     )

    # def reader(self) -> 'Reader':
    #     """Get the reader for the prompt

    #     Returns:
    #         Reader: Returns the base reader
    #     """
    #     # TODO: update this to allow it to change
    #     return NullRead()

    def clone(self) -> typing.Self:
        """Do a shallow copy of the message

        Returns:
            Message: The cloned message
        """
        return self.__class__(
            source=self.source,
            data=self.data
        )

    # def aslist(self) -> typing.List['Message']:
    #     """Get the message as a list

    #     Returns:
    #         typing.List[Message]: The message as a list
    #     """
    #     return [self]
    
    def render(self) -> str:
        """Render the message

        Returns:
            str: Return the message and the source
        """
        return f'{self.source}: {render(self.data)}'


class MessageConverter(ABC):

    @abstractmethod
    def to_message(self, message: typing.Dict) -> Message:
        pass

    @abstractmethod
    def from_message(self, message: Message) -> typing.Dict:
        pass


class EmbeddingMessage(Message):
    """A message that contains text. Typically used for LLMs
    """

    def __init__(self, source: str, embedding: np.array, source_data: typing.Any):
        """Create a text message with a source

        Args:
            source (str): the source of the message
            text (typing.Union[str, Cue]): the content of the message

        """
        super().__init__(
            source=source,
            data={
                'embedding': embedding,
                'source_data': source_data
            }
        )

    def render(self) -> str:
        """Render the text message

        Returns:
            str: Return the message and the text for the message
        """
        pass
        # text = self.data['text']
        # return f'{self.source}: {
        #     text.render() if isinstance(text, Cue) else text
        # }'

    def clone(self) -> typing.Self:
        """Do a shallow copy of the message

        Returns:
            Message: The cloned message
        """
        return self.__class__(
            source=self.source,
            embedding=self.embedding,
            source_data=self.source_data
        )

    @property
    def text(self) -> str:
        """Get the text for the message

        Returns:
            str: The text for the message
        """
        return self.data['source_data']

# Use this later

# def embedding_df(messages: typing.List[EmbeddingMessage]) -> pd.DataFrame:
#     """Create a dataframe from embedding messages

#     Args:
#         messages (typing.List[EmbeddingMessage]): The messages to create from

#     Returns:
#         pd.DataFrame: The dataframe
#     """
    
#     rows = [
#         {
#             "embedding": row.data['embedding'], 
#             "source": row.source, 
#             "data": row["source_data"]
#             **row.data["meta"]
#         } for row in messages
#     ]
#     return pd.DataFrame(rows)


class TextMessage(Message):
    """A message that contains text. Typically used for LLMs
    """

    def __init__(self, source: str, text: typing.Union[str, 'Cue']):
        """Create a text message with a source

        Args:
            source (str): the source of the message
            text (typing.Union[str, Cue]): the content of the message

        """
        super().__init__(
            source=source,
            data={
                'text': text
            }
        )

    # def reader(self) -> 'Reader':
    #     """The reader to use

    #     Returns:
    #         Reader: The reader used by message
    #     """
    #     text = self['text']
    #     if isinstance(text, Cue) and text.out is not None:
    #         return text.out
    #     return NullRead(name='')
    
    def render(self) -> str:
        """Render the text message

        Returns:
            str: Return the message and the text for the message
        """
        text = self.data['text']
        return f'{self.source}: {
            text.render() if isinstance(text, Cue) else text
        }'

    def clone(self) -> typing.Self:
        """Do a shallow copy of the message

        Returns:
            Message: The cloned message
        """
        return self.__class__(
            source=self.source,
            text=self.data['text']
        )

    @property
    def text(self) -> str:
        """Get the text for the message

        Returns:
            str: The text for the message
        """
        return self.data['text']


class CueMessage(Message):
    """A message that contains text. Typically used for LLMs
    """

    def __init__(self, source: str, cue: Cue):
        """Create a text message with a source

        Args:
            source (str): the source of the message
            text (typing.Union[str, Cue]): the content of the message

        """
        super().__init__(
            source=source,
            data={
                'cue': cue
            }
        )
        
    def reader(self) -> 'Reader':
        """The reader to use

        Returns:
            Reader: The reader used by message
        """
        cue = self['cue']
        return cue.out

    def render(self) -> str:
        """Render the text message

        Returns:
            str: Return the message and the text for the message
        """
        text = self.cue.text
        return f'{self.source}: {
            text.render() if isinstance(text, Cue) else text
        }'

    def clone(self) -> typing.Self:
        """Do a shallow copy of the message

        Returns:
            Message: The cloned message
        """
        return self.__class__(
            source=self.source,
            text=self.data['text']
        )

    @property
    def text(self) -> str:
        """Get the text for the message

        Returns:
            str: The text for the message
        """
        return self.cue.text


class ObjMessage(Message):
    """A message that contains text. Typically used for LLMs
    """

    def __init__(self, source: str, obj: typing.Any, text_source: str):
        """Create a text message with a source

        Args:
            source (str): the source of the message
            text (typing.Union[str, Cue]): the content of the message

        """
        super().__init__(
            source=source,
            data={
                'object': obj,
                'text_source': text_source
            }
        )

    def render(self) -> str:
        """Render the text message

        Returns:
            str: Return the message and the text for the message
        """
        obj = self.object
        return f'{self.source}: {
            obj.render() if is_renderable(obj) else self.text_source
        }'

    def clone(self) -> typing.Self:
        """Do a shallow copy of the message

        Returns:
            Message: The cloned message
        """
        return self.__class__(
            source=self.source,
            text=self.data['text']
        )

    @property
    def text(self) -> str:
        """Get the text for the message

        Returns:
            str: The text for the message
        """
        return self.text_source


class ToolParam(BaseModel, Renderable):
    name: str
    type_: str
    descr: str = ''
    required: typing.List[str] = Field(default_factory=list)
    enum: typing.Optional[typing.List[typing.Any]] = None
    minimum: typing.Optional[float] = None
    maximum: typing.Optional[float] = None
    minLength: typing.Optional[int] = None
    maxLength: typing.Optional[int] = None
    default: typing.Optional[typing.Any] = None
    format: typing.Optional[str] = None

    def to_dict(self):
        pass

    def render(self) -> str:
        pass


class ToolObjParam(ToolParam):
    params: typing.List[ToolParam] = Field(default_factory=list)


class ToolArrayParam(ToolParam):
    items: ToolParam  # Specifies the type of items in the array

    def __init__(self, name: str, type_: str, descr: str='', **kwargs):

        super().__init__(
            name=name, type_=type_, descr=descr, **kwargs
        )

    def __init__(
        self, name: str, params: typing.List[ToolParam], type_: str, descr: str='', **kwargs
    ):

        super().__init__(
            name=name, type_=type_, descr=descr, params=params, **kwargs
        )


class FunctionMessage(Message):
    """A message that contains text. Typically used for LLMs
    """

    def __init__(self, source: str, name: str, content: typing.Any):
        """Create a text message with a source

        Args:
            source (str): the source of the message
            text (typing.Union[str, Cue]): the content of the message

        """
        super().__init__(
            source=source,
            data={
                'name': name,
                'content': content
            }
        )

    def render(self) -> str:
        """Render the text message

        Returns:
            str: Return the message and the text for the message
        """
        return f'{self.source}: [{self.name}] => {str(self.content)}'

    def clone(self) -> typing.Self:
        """Do a shallow copy of the message

        Returns:
            Message: The cloned message
        """
        return self.__class__(
            source=self.source,
            text=self.data['text']
        )

    @property
    def text(self) -> str:
        """Get the text for the message

        Returns:
            str: The text for the message
        """
        return f"{self.name} => {self.content}"


class ToolOptionMessage(object):

    def __init__(self, name: str, type_: str, param: ToolParam, descr: str=None, required: typing.List[str]=None, strict: bool=True):
        """Create a text message with a source

        Args:
            source (str): the source of the message
            text (typing.Union[str, Cue]): the content of the message

        """
        super().__init__(
            source='tool',
            data={
                'name': name,
                'type_': type_,
                'param': param,
                'descr': descr,
                'required': required,
                'strict': strict
            }
        )
    
    def render(self) -> str:
        """Render the text message

        Returns:
            str: Return the message and the text for the message
        """

        d = self.data
        d['param'] = d['param'].to_dict()

        obj = self.object
        return f'{self.source}: {
            obj.render() if is_renderable(obj) else self.text_source
        }'

    def clone(self) -> typing.Self:
        """Do a shallow copy of the message

        Returns:
            Message: The cloned message
        """
        return self.__class__(
            source=self.source,
            text=self.data['text']
        )

    @property
    def text(self) -> str:
        """Get the text for the message

        Returns:
            str: The text for the message
        """
        return self.data.text_source


class FunctionCall(pydantic.BaseModel):

    name: str
    arguments: typing.Dict


class ToolMessage(Message):
    """A message that contains text. Typically used for LLMs
    """

    def __init__(self, function_call: FunctionCall, text: str=''):
        """Create a text message with a source

        Args:
            source (str): the source of the message
            text (typing.Union[str, Cue]): the content of the message

        """
        super().__init__(
            source='assistant',
            data={
                'function_call': function_call,
                'text': text
            }
        )

    def render(self) -> str:
        """Render the text message

        Returns:
            str: Return the message and the text for the message
        """
        pass
        # d = self.data
        # d['param'] = d['param'].to_dict()

        # obj = self.object
        # return f'{self.source}: {
        #     obj.render() if is_renderable(obj) else self.text_source
        # }'

    def clone(self) -> typing.Self:
        """Do a shallow copy of the message

        Returns:
            Message: The cloned message
        """
        pass
        # return self.__class__(
        #     source=self.source,
        #     text=self.data['text']
        # )

    @property
    def text(self) -> str:
        """Get the text for the message

        Returns:
            str: The text for the message
        """
        pass


class Dialog(pydantic.BaseModel, Renderable):
    """A Dialog stores the interactions between the system/user and the assistant
    (i.e. the prompts and the responses)
    """
    messages: typing.List[Message] = pydantic.Field(default_factory=list)

    def __init__(self, messages=None):
        """Create a dialog

        Args:
            messages: The messages
        """
        super().__init__(messages=messages or [])

    def __iter__(self) -> typing.Iterator[Message]:
        """Iterate over each message in the dialog

        Yields:
            Iterator[typing.Iterator[Message]]: Each message in the dialog
        """
        for message in self.messages:
            yield message

    def __add__(self, other: 'Dialog') -> 'Dialog':
        """Concatenate two dialogs together

        Args:
            other (Dialog): The other dialog to concatenate

        Returns:
            Dialog: The concatenated dialog
        """
        return Dialog(
            self.messages + other.messages
        )

    def __getitem__(self, idx) -> Message:
        """Retrieve a value from the dialog

        Args:
            idx : The index to add at

        Returns:
            Message: The message in the dialog
        """
        return self.messages[idx]

    def __setitem__(self, idx, message) -> Self:
        """Set idx with a message

        Args:
            idx: The index to set
            message: The message to set

        Returns:
            Dialog: The updated dialog
        """
        self.messages[idx] = message
        return self

    def insert(self, index: int, message: Message):
        """Insert a value into the dialog

        Args:
            index (int): The index to insert at
            message (Message): The message to insert
        """
        self.messages.insert(index, message)

    def pop(self, index: int):
        """Remove a value from the dialog

        Args:
            index (int): The index to pop
        """
        self.messages.pop(index)

    def remove(self, message: Message):
        """Remove a message from the dialog

        Args:
            message (Message): The message to remove
        """
        self.messages.remove(message)

    def append(self, message: Message):
        """Append a message to the end of the dialog

        Args:
            message (Message): The message to add
        """
        self.messages.append(message)

    # def process_response(self, response: 'AIResponse') -> 'AIResponse':
    #     """Process the response of the Model

    #     Args:
    #         response (AIResponse): The response to process

    #     Returns:
    #         AIResponse: The updated response
    #     """
    #     p = self.reader()
    #     response = response.clone()
    #     response.val = p.read(response.message)
    #     return response

    # def instruct(
    #     self, instruct: 'Instruct', ai_model: 'AIModel', 
    #     ind: int=0, replace: bool=True
    # ) -> AIResponse:
    #     """Instruct the AI

    #     Args:
    #         instruct (Instruct): The cue to use
    #         ai_model (AIModel): The AIModel to use
    #         ind (int, optional): The index to set to. Defaults to 0.
    #         replace (bool, optional): Whether to replace at the index if already set. Defaults to True.

    #     Returns:
    #         AIResponse: The output from the AI
    #     """
    #     cue = instruct.i()
        
    #     self.system(cue, ind, replace)
    #     response = ai_model.forward(self.messages)
    #     response = self.process_response(response)
    #     self.assistant(response.content)
    #     return response

    # def prompt(self, model: 'AIModel', append: bool=True) -> 'AIResponse':
    #     """Prompt the AI

    #     Args:
    #         model (AIModel): The model to usee
    #         append (bool, optional): Whether to append the output. Defaults to True.

    #     Returns:
    #         AIResponse: The response from the AI
    #     """
    #     response = model(self)

    #     if append:
    #         self.append(response.message)
    #     return response

    # def stream_prompt(
    #     self, model: 'AIModel', append: bool=True, **kwarg_override
    # ) -> typing.Iterator[typing.Tuple['AIResponse', 'AIResponse']]:
    #     """Prompt the AI

    #     Args:
    #         model (AIModel): The model to usee
    #         append (bool, optional): Whether to append the output. Defaults to True.

    #     Returns:
    #         AIResponse: The response from the AI
    #     """
    #     for d, dx in model.stream(self, **kwarg_override):
    #         yield d, dx

    #     if append:
    #         self.append(d.message)
    #     return d

    def add(self, message: Message, ind: typing.Optional[int]=None, replace: bool=False):
        """Add a message to the dialog

        Args:
            message (Message): The message to add
            ind (typing.Optional[int], optional): The index to add. Defaults to None.
            replace (bool, optional): Whether to replace at the index. Defaults to False.

        Raises:
            ValueError: If the index is not correct
        """
        if ind is not None and ind < 0:
            ind = max(len(self.messages) + ind, 0)

        if ind is None or ind == len(self.messages):
            if not replace or ind == len(self.messages):
                self.messages.append(message)
            else:
                self.messages[-1] = message
        elif ind > len(self.messages):
            raise ValueError(
                f'The index {ind} is out of bounds '
                f'for size {len(self.messages)}')
        elif replace:
            self.messages[ind] = message
        else:
            self.messages.insert(ind, message)
        
    def clone(self) -> 'Dialog':
        """Clones the dialog

        Returns:
            Dialog: A dialog cloned with shallow copying of the messages
        """
        return Dialog(
            messages=[message.clone() for message in self.messages]
        )

    def extend(self, dialog: typing.Union['Dialog', typing.List[Message]]):
        """Extend the dialog with another dialog or a list of messages

        Args:
            dialog (typing.Union[&#39;Dialog&#39;, typing.List[Message]]): _description_
        """
        if isinstance(dialog, Dialog):
            dialog = dialog.messages
        
        self.messages.extend(dialog)

    def message(self, source: str, text: typing.Optional[str]=None, _ind: typing.Optional[int]=None, _replace: bool=False, **kwargs):
        """Add a message to the 

        Args:
            source (str): the source of the message
            text (typing.Optional[str], optional): The text message. Defaults to None.
            _ind (typing.Optional[int], optional): The index to set at. Defaults to None.
            _replace (bool, optional): Whether to replace the the text at the index. Defaults to False.

        Raises:
            ValueError: If no message was passed in
        """
        if len(kwargs) == 0 and text is not None:
            message = TextMessage(source, text)
        elif text is not None:
            message = Message(text=text, **kwargs)
        elif text is None:
            message = Message(**kwargs)
        else:
            raise ValueError('No message has been passed. The text and kwargs are empty')

        self.add(message, _ind, _replace)

    def user(self, text: str=None, _ind: int=None, _replace: bool=False, **kwargs):
        """Add a user message

        Args:
            text (str, optional): The text for the message. Defaults to None.
            _ind (int, optional): The index to add to. Defaults to None.
            _replace (bool, optional): Whether to replace at the index. Defaults to False.
        """
        self.message('user', text, _ind, _replace, **kwargs)

    def assistant(self, text: str=None, _ind=None, _replace: bool=False, **kwargs):
        """Add an assistant message

        Args:
            text (str, optional): The text for the message. Defaults to None.
            _ind (int, optional): The index to add to. Defaults to None.
            _replace (bool, optional): Whether to replace at the index. Defaults to False.
        """
        self.message('assistant', text, _ind, _replace, **kwargs)

    def system(self, text: str=None, _ind=None, _replace: bool=False, **kwargs):
        """Add a system message

        Args:
            text (str, optional): The text for the message. Defaults to None.
            _ind (int, optional): The index to add to. Defaults to None.
            _replace (bool, optional): Whether to replace at the index. Defaults to False.
        """
        self.message('system', text, _ind, _replace, **kwargs)

    def reader(self) -> 'Reader':
        """Get the "Reader" for the dialog. By default will use the last one
        that is available.

        Returns:
            Reader: The reader to retrieve
        """
        for r in reversed(self.messages):
            if isinstance(r, Cue):
                if r.reader is not None:
                    return r.reader
        return NullRead(name='')
    
    def exclude(self, *source: str) -> 'Dialog':

        exclude = set(source)
        return Dialog(
            messages=[message for message in self.messages
            if message.source not in exclude]
        )
    
    def include(self, *source: str) -> 'Dialog':
        include = set(source)
        return Dialog(
            messages=[message for message in self.messages
            if message.source in include]
        )

    def render(self) -> str:
        """Render the dialog as a series of turns 
        <source>: <text>

        Returns:
            str: The dialog
        """
        return '\n'.join(
            message.render() for message in self.messages
        )

    def aslist(self) -> typing.List['Message']:
        """Retrieve the message list

        Returns:
            typing.List[Message]: the messages in the dialog
        """
        return self.messages
    
    def __len__(self) -> int:
        """Get the size of the dialog

        Returns:
            int: the number of turns in the dialog
        """
        return len(self.messages)
