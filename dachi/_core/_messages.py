# 1st party
import typing
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


from pydantic import BaseModel, Field
from typing import Dict, Any


class Delta(pydantic.BaseModel):

    data: typing.Dict[str, typing.Any]

    def __init__(self, **data):

        super().__init__(data=data)

    def __getattr__(self, key: str) -> typing.Any:
        """Get an item from the message

        Args:
            key (str): The key to retrieve for

        Raises:
            KeyError: If key is not valid
        """
        if key in self.data:
            return self.data[key]
        raise KeyError(f'{key} is not a member of {type(self)}')

    def __setattr__(
        self, key: str, value: typing.Any
    ) -> typing.Any:
        """Set an item in the message

        Args:
            key (str): The key to set for
            value (typing.Any): The value to set

        Raises:
            KeyError: An error
        """
        self.data[key] = value
        return value


class FileBase(pydantic.BaseModel):

    type_: str
    description: str


class ChatMessage(pydantic.BaseModel, ABC):
    
    alias: typing.Optional[str] = None
    role: str
    text: typing.Optional[str] = None
    files: typing.Optional[FileBase]=None,


class ByteFile(FileBase):

    bytes: str
    

class URLFile(FileBase):

    url: str
    

class Schema(pydantic.BaseModel):
    
    @abstractmethod
    def to_str(self) -> str:
        raise NotImplementedError


class JSONSchema(Schema):
    """A general-purpose class to store and represent JSON Schemas."""
    schema_cls: typing.Type[pydantic.BaseModel] = Field(..., description="The Pydantic model class for this schema")

    def to_dict(self) -> typing.Dict:
        """Convert the schema to a JSON string."""
        return self.schema_cls.model_json_schema()


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

    def render(self) -> str:
        return render(self)


class ToolObjParam(ToolParam):
    params: typing.List[ToolParam] = Field(default_factory=list)


class ToolArrayParam(ToolParam):
    items: typing.List[ToolParam]

    def __init__(self, name: str, items: typing.List[ToolParam], descr: str='', **kwargs):
        """Create an array of tools

        Args:
            name (str): The name of the array
            items (typing.List[ToolParam]): The items in the array
            descr (str, optional): The description. Defaults to ''.
        """
        super().__init__(
            name=name, type_="array", items=items, descr=descr, **kwargs
        )


class Tool(pydantic.BaseModel):
    
    type_: str


class FunctionTool(Tool):
    
    name: str
    params: typing.Optional[typing.List[ToolParam]]
    descr: typing.Optional[str] = None

    def __init__(
        self, name: str, params: typing.List[ToolParam], descr: str=None
    ):
        super().__init__(
            type_='function', name=name, params=params, descr=descr
        )


class SystemMessage(ChatMessage):

    cue: typing.Optional[Cue] = None
    schema: typing.Optional[Schema] = None
    tools: typing.Optional[typing.List[Tool]] = None

    def __init__(
        self,  instruction: typing.Union[Cue, str],
        files: typing.Optional[FileBase]=None,
        schema: typing.Optional[Schema]=None,
        tools: typing.Optional[typing.List[Tool]]=None,
        alias: str=None
    ):
        if isinstance(instruction, Cue):
            cue = instruction
            text = None
        else:
            cue = None
            text = instruction
        super().__init__(
            alias=alias, role='system', 
            text=text, cue=cue, tools=tools, files=files,
            schema=schema
        )


class AssistantMessage(ChatMessage):

    response: typing.Optional[typing.Any] = None
    delta: typing.Optional[Delta] = None
    parsed: typing.Optional[typing.Any] = None

    def __init__(
        self,  text: str,
        files: typing.Optional[FileBase]=None,
        response: typing.Optional[typing.Any]=None,
        delta: Delta=None, alias: str=None, parsed: typing.Optional[typing.Any]=None
    ):
        super().__init__(
            alias=alias, role='assistant', text=text,
            response=response, files=files, delta=delta,
            parsed=parsed

        )


class ToolMessage(ChatMessage):

    name: str
    return_value: str

    def __init__(
        self, 
        return_value: str,
        alias: str=None,
        files: typing.Optional[FileBase]=None,
    ):
        super().__init__(
            alias, role='tool', text=return_value,
            files=files
        )


class UserMessage(ChatMessage):

    def __init__(
        self, text: str=None, alias: typing.Optional[str]=None,
        files: typing.Optional[str]=None
    ):
        super().__init__(
            role='user', text=text, alias=alias, files=files
        )


class Dialog(pydantic.BaseModel, Renderable):
    """A Dialog stores the interactions between the system/user and the assistant
    (i.e. the prompts and the responses)
    """
    messages: typing.List[ChatMessage] = pydantic.Field(default_factory=list)

    def __init__(self, messages=None):
        """Create a dialog

        Args:
            messages: The messages
        """
        super().__init__(messages=messages or [])

    def __iter__(self) -> typing.Iterator[ChatMessage]:
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

    def __getitem__(self, idx) -> ChatMessage:
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

    def insert(self, index: int, message: ChatMessage):
        """Insert a value into the dialog

        Args:
            index (int): The index to insert at
            message (ChatMessage): The message to insert
        """
        self.messages.insert(index, message)

    def pop(self, index: int):
        """Remove a value from the dialog

        Args:
            index (int): The index to pop
        """
        self.messages.pop(index)

    def remove(self, message: ChatMessage):
        """Remove a message from the dialog

        Args:
            message (ChatMessage): The message to remove
        """
        self.messages.remove(message)

    def append(self, message: ChatMessage):
        """Append a message to the end of the dialog

        Args:
            message (Message): The message to add
        """
        self.messages.append(message)

    def add(self, message: ChatMessage, ind: typing.Optional[int]=None, replace: bool=False):
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

    def extend(self, dialog: typing.Union['Dialog', typing.List[ChatMessage]]):
        """Extend the dialog with another dialog or a list of messages

        Args:
            dialog (typing.Union[&#39;Dialog&#39;, typing.List[Message]]): _description_
        """
        if isinstance(dialog, Dialog):
            dialog = dialog.messages
        
        self.messages.extend(dialog)

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

    def render(self) -> str:
        """Render the dialog as a series of turns 
        <role>: <text>

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
        
    def clone(self) -> 'Dialog':
        """Clones the dialog

        Returns:
            Dialog: A dialog cloned with shallow copying of the messages
        """
        return Dialog(
            messages=[message for message in self.messages]
        )

    @property
    def cue(self) -> typing.Optional[Cue]:
        """Get the final cue in the dialog

        Returns:
            Cue: The last cue in the dialog
        """
        cue = None
        for message in self.messages:
            if isinstance(message, SystemMessage):
                if message.cue is not None:
                    cue = message.cue
        return cue

        

    # def message(self, role: str, text: typing.Optional[str]=None, _ind: typing.Optional[int]=None, _replace: bool=False, **kwargs):
    #     """Add a message to the 

    #     Args:
    #         role (str): the role of the message
    #         text (typing.Optional[str], optional): The text message. Defaults to None.
    #         _ind (typing.Optional[int], optional): The index to set at. Defaults to None.
    #         _replace (bool, optional): Whether to replace the the text at the index. Defaults to False.

    #     Raises:
    #         ValueError: If no message was passed in
    #     """
    #     if len(kwargs) == 0 and text is not None:
    #         message = TextMessage(role, text)
    #     elif text is not None:
    #         message = Message(text=text, **kwargs)
    #     elif text is None:
    #         message = Message(**kwargs)
    #     else:
    #         raise ValueError('No message has been passed. The text and kwargs are empty')

    #     self.add(message, _ind, _replace)

    # def user(self, text: str=None, _ind: int=None, _replace: bool=False, **kwargs):
    #     """Add a user message

    #     Args:
    #         text (str, optional): The text for the message. Defaults to None.
    #         _ind (int, optional): The index to add to. Defaults to None.
    #         _replace (bool, optional): Whether to replace at the index. Defaults to False.
    #     """
    #     self.message('user', text, _ind, _replace, **kwargs)

    # def assistant(self, text: str=None, _ind=None, _replace: bool=False, **kwargs):
    #     """Add an assistant message

    #     Args:
    #         text (str, optional): The text for the message. Defaults to None.
    #         _ind (int, optional): The index to add to. Defaults to None.
    #         _replace (bool, optional): Whether to replace at the index. Defaults to False.
    #     """
    #     self.message('assistant', text, _ind, _replace, **kwargs)

    # def system(
    #     self, text: str=None, _ind=None, 
    #     _replace: bool=False, **kwargs
    # ):
    #     """Add a system message

    #     Args:
    #         text (str, optional): The text for the message. Defaults to None.
    #         _ind (int, optional): The index to add to. Defaults to None.
    #         _replace (bool, optional): Whether to replace at the index. Defaults to False.
    #     """
    #     self.message('system', text, _ind, _replace, **kwargs)

# class ToolOption(pydantic.BaseModel):

#     # type_ = function
#     def __init__(
#         self, tools: typing.List[Tool], delta: typing.Optional[Delta]=None
        
#         # name: str, type_: str, param: ToolParam, descr: str=None, required: typing.List[str]=None, strict: bool=True, delta: Delta=None
#     ):
#         """Create a text message with a role

#         Args:
#             role (str): the role of the message
#             text (typing.Union[str, Cue]): the content of the message

#         """
#         super().__init__(
#             tools=tools,
#         )
    
#     def render(self) -> str:
#         """Render the text message

#         Returns:
#             str: Return the message and the text for the message
#         """
#         text = 'Tool: [{}]'.format(
#             ','.join(f'{tool.name}: {tool.type_}' for tool in self.tools
#         ))
#         return text

#     @property
#     def text(self) -> str:
#         """Get the text for the message

#         Returns:
#             str: The text for the message
#         """
#         text = '[{}]'.format(
#             ','.join(f'{tool.name}' for tool in self.tools
#         ))
#         return text


# class MessageConverter(ABC):
#     """Use the message converter to convert between
#     the message and what is used in the API
#     """

#     @abstractmethod
#     def to_message(self, response: typing.Dict) -> Message:
#         """Convert from an AI response to a message

#         Args:
#             response (typing.Dict): The response to convert

#         Returns:
#             Message: the message from the API
#         """
#         pass

#     @abstractmethod
#     def from_message(self, message: Message) -> typing.Dict:
#         """_summary_

#         Args:
#             message (Message): 

#         Returns:
#             typing.Dict: _description_
#         """
#         pass


# class EmbeddingMessage(Message):
#     """A message that contains text. Typically used for LLMs
#     """

#     def __init__(
#         self, role: str, embedding: np.array, 
#         source: typing.Any
#     ):
#         """Create an embeddign message 

#         Args:
#             role (str): the role of the message
#             embedding (typing.Union[str, Cue]): the content of the message
#             source_data (typing.Any)

#         """
#         super().__init__(
#             role=role,
#             embedding=embedding,
#             source=source
#         )

#     def render(self) -> str:
#         """Render the text message

#         Returns:
#             str: Return the message and the text for the message
#         """
        
#         data = self.source
#         return f'{self.role}: {render(data)}'

#     @property
#     def text(self) -> str:
#         """Get the text for the message

#         Returns:
#             str: The text for the message
#         """
#         return self.data['source']


# class TextMessage(Message):
#     """A message that contains text. Typically used for LLMs
#     """

#     def __init__(self, role: str, content: str, delta: Delta=None):
#         """Create a text message with a source

#         Args:
#             source (str): the source of the message
#             content (typing.Union[str, Cue]): the content of the message

#         """
#         super().__init__(
#             role=role,
#             content=content,
#             delta=delta
#         )
    
#     def render(self) -> str:
#         """Render the text message

#         Returns:
#             str: Return the message and the text for the message
#         """
#         text = self.data['content']
#         return f'{self.role}: {text}'

#     @property
#     def text(self) -> str:
#         """Get the text for the message

#         Returns:
#             str: The text for the message
#         """
#         return self.content


# class CueMessage(Message):
#     """A message that contains text. Typically used for LLMs
#     """

#     def __init__(self, role: str, cue: Cue, delta: Delta=None):
#         """Create a text message with a source

#         Args:
#             source (str): the source of the message
#             cue (Cue): the cue specified by the message

#         """
#         super().__init__(
#             role=role,
#             cue=cue,
#             delta=delta
#         )
        
#     @property
#     def reader(self) -> 'Reader':
#         """The reader to use

#         Returns:
#             Reader: The reader used by message
#         """
#         return self.cue.out

#     def render(self) -> str:
#         """Render the text message

#         Returns:
#             str: Return the message and the text for the message
#         """
#         text = self.cue.text
#         return f'{self.role}: {
#             text.render()
#         }'

#     @property
#     def text(self) -> str:
#         """Get the text for the message

#         Returns:
#             str: The text for the message
#         """
#         return self.cue.text


# class ObjMessage(Message):
#     """A message that contains text. Typically used for LLMs
#     """

#     def __init__(self, role: str, obj: typing.Any, source: str, delta: Delta=None):
#         """Create a text message with a source

#         Args:
#             role (str): the role of the message
#             obj : The data in the message after processing
#             source (str): the source of the message

#         """
#         super().__init__(
#             role=role,
#             obj=obj,
#             source=source,
#             delta=delta
#         )

#     def render(self) -> str:
#         """Render the text message

#         Returns:
#             str: Return the message and the text for the message
#         """
#         obj = self.obj
#         return f'{self.role}: {
#             render(obj) if is_renderable(obj) else self.source
#         }'

#     @property
#     def text(self) -> str:
#         """Get the text for the message

#         Returns:
#             str: The text for the message
#         """
#         return self.source


# class ToolParam(BaseModel, Renderable):
#     name: str
#     type_: str
#     descr: str = ''
#     required: typing.List[str] = Field(default_factory=list)
#     enum: typing.Optional[typing.List[typing.Any]] = None
#     minimum: typing.Optional[float] = None
#     maximum: typing.Optional[float] = None
#     minLength: typing.Optional[int] = None
#     maxLength: typing.Optional[int] = None
#     default: typing.Optional[typing.Any] = None
#     format: typing.Optional[str] = None

#     def render(self) -> str:
#         return render(self)


# class ToolObjParam(ToolParam):
#     params: typing.List[ToolParam] = Field(default_factory=list)


# class ToolArrayParam(ToolParam):
#     items: typing.List[ToolParam]

#     def __init__(self, name: str, items: typing.List[ToolParam], descr: str='', **kwargs):
#         """Create an array of tools

#         Args:
#             name (str): The name of the array
#             items (typing.List[ToolParam]): The items in the array
#             descr (str, optional): The description. Defaults to ''.
#         """
#         super().__init__(
#             name=name, type_="array", items=items, descr=descr, **kwargs
#         )


# class FunctionMessage(Message):
#     """A message that contains text. Typically used for LLMs
#     """

#     def __init__(self, role: str, name: str, response: typing.Any, delta: Delta=None):
#         """Create a text message with a role

#         Args:
#             role (str): the role of the message
#             name (str): the name of the function
#             response : The response from the function
#             delta: any change in the function output

#         """
#         super().__init__(
#             role=role,
#             name=name,
#             response=response,
#             delta=delta
#         )

#     def render(self) -> str:
#         """Render the text message

#         Returns:
#             str: Return the message and the text for the message
#         """
#         return f'{self.role}: [{self.name}] => {str(self.response)}'

#     @property
#     def text(self) -> str:
#         """Get the text for the message

#         Returns:
#             str: The text for the message
#         """
#         return f"{self.name} => {self.response}"


# class Tool(pydantic.BaseModel):
    
#     type_: str


# class FunctionTool(Tool):
    
#     name: str
#     params: typing.Optional[typing.List[ToolParam]]
#     descr: typing.Optional[str] = None

#     def __init__(
#         self, name: str, params: typing.List[ToolParam], descr: str=None
#     ):
#         super().__init__(
#             type_='function', name=name, params=params, descr=descr
#         )


# class ToolOptionMessage(Message):

#     # type_ = function
#     def __init__(
#         self, tools: typing.List[Tool], delta: typing.Optional[Delta]=None
        
#         # name: str, type_: str, param: ToolParam, descr: str=None, required: typing.List[str]=None, strict: bool=True, delta: Delta=None
#     ):
#         """Create a text message with a role

#         Args:
#             role (str): the role of the message
#             text (typing.Union[str, Cue]): the content of the message

#         """
#         super().__init__(
#             role='system',
#             tools=tools,
#             delta=delta
#         )
    
#     def render(self) -> str:
#         """Render the text message

#         Returns:
#             str: Return the message and the text for the message
#         """
#         text = 'Tool: [{}]'.format(
#             ','.join(f'{tool.name}: {tool.type_}' for tool in self.tools
#         ))
#         return text

#     @property
#     def text(self) -> str:
#         """Get the text for the message

#         Returns:
#             str: The text for the message
#         """
#         text = '[{}]'.format(
#             ','.join(f'{tool.name}' for tool in self.tools
#         ))
#         return text


# class FunctionCall(pydantic.BaseModel):

#     name: str
#     arguments: typing.Dict


# class ToolUseMessage(Message):
#     """A message that contains a function call
#     """

#     def __init__(self, function_call: FunctionCall, content: str='', delta: Delta=None):
#         """Create a text message with a role

#         Args:
#             role (str): the role of the message
#             text (typing.Union[str, Cue]): the content of the message

#         """
#         super().__init__(
#             role='assistant',
#             function_call=function_call,
#             content=content,
#             delta=delta
#         )

#     def render(self) -> str:
#         """Render the text message

#         Returns:
#             str: Return the message and the text for the message
#         """
#         return (
#             f"{self.role}: {self.content}"
#         )

#     @property
#     def text(self) -> str:
#         """Get the text for the message

#         Returns:
#             str: The text for the message
#         """
#         return self.content




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



# Add in delta to the message

# 
# message(
#     role='system',
#     content=Text(...),
#     cls=...,
#     dx=None
# )

# by default it returns the 
# dialog.... This seems easier
# to use.. even if dialog is 

# add a message
# dialog.append(model.system())
# dialog[0] = 
# dialog.append(model.system(tools=...))
# dialog.append(model.function(response, dialog=...))
# model.assistant()
# model.user()
# message, dialog = model.forward(role, message, dialog)
# message, 
# for message, dialog in model.stream(message, dialog)
# message.
# if you don't pass a
# message = model.prompt(
#   text=...
# )
# message, dialog = model.prompt('Help me figure this out', dialog=dialog)
# message.content.text
# message.content.images
# message.content.source ...

# message, dialog = model.forward(...)
# 



# class Message(pydantic.BaseModel, Renderable):
#     """A prompt that consists of a single message to send the AI
#     """
#     role: str # The source of the messsage (user, assistant etc)
#     data: typing.Dict[str, typing.Any] # The contents of the message
#     delta: typing.Optional[Delta] = None # the delta of the data if used for the model

#     def __init__(
#         self, role: str, delta: 'Message'=None, **data
#     ):
#         """Create a message

#         Args:
#             role (str): The rolee of the message
#             delta (typing.Dict, optional): The delta of the data - use when streaming. Defaults to None.
#         """
#         super().__init__(role=role, data=data, delta=delta)

#     def __getattr__(self, key: str) -> typing.Any:
#         """Get an item from the message

#         Args:
#             key (str): The key to retrieve for

#         Raises:
#             KeyError: If key is not valid
#         """
#         if key in self.data:
#             return self.data[key]
#         raise KeyError(f'{key} is not a member of {type(self)}')

#     def __setattr__(
#         self, key: str, value: typing.Any
#     ) -> typing.Any:
#         """Set an item in the message

#         Args:
#             key (str): The key to set for
#             value (typing.Any): The value to set

#         Raises:
#             KeyError: An error
#         """
#         self.data[key] = value
#         return value

#     def render(self) -> str:
#         """Render the message

#         Returns:
#             str: Return the message and the role
#         """
#         return f'{self.role}: {render(self.data)}'