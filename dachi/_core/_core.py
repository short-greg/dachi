# 1st party
import typing
from abc import ABC, abstractmethod
from typing import get_type_hints
from typing import Self
import typing

from uuid import uuid4
from enum import Enum
import asyncio
from dataclasses import dataclass

import inspect
import json

# 3rd party
import pydantic

# local
from ._utils import (
    is_primitive, escape_curly_braces, 
    unescape_curly_braces,
    generic_class
)


class _Types(Enum):

    UNDEFINED = 'UNDEFINED'
    WAITING = 'WAITING'


UNDEFINED = _Types.UNDEFINED
WAITING = _Types.WAITING


S = typing.TypeVar('S', bound='Struct')
# X = typing.Union[str, 'Description', 'Instruction']


class Renderable(ABC):

    @abstractmethod
    def render(self) -> str:
        pass


def model_template(model_cls: typing.Type[pydantic.BaseModel]) -> str:
    """Get the template for a pydantic.Model

    Args:
        model_cls (typing.Type[pydantic.BaseModel]): The model to retrieve for

    Returns:
        str: The model template string
    """
    template = {}
    for name, field_type in get_type_hints(model_cls).items():
        
        if inspect.isclass(field_type) and issubclass(field_type, pydantic.BaseModel):
            template[name] = model_template(field_type)
        else:
            template[name] = {
                "is_required": model_cls.model_fields[name].is_required(),
                "type": field_type
            }
    return template


class Struct(pydantic.BaseModel, Renderable):
    """Struct is used to contain data that is used
    """
    model_config = pydantic.ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True
    )

    @classmethod
    def template(cls) -> typing.Dict:
        """Get the template for the Struct

        Returns:
            typing.Dict: The template 
        """
        template = {}
        
        base_template = model_template(cls)
        for field_name, field in cls.model_fields.items():
            field_type = field.annotation
            if isinstance(field_type, type) and issubclass(field_type, Struct):
                template[field_name] = field_type.template()
            else:
                template[field_name] = {
                    "type": field.annotation,
                    "description": field.description,
                    "default": field.default if field.default is not None else None,
                    "is_required": base_template[field_name]['is_required']
                }
        return template
    
    @classmethod
    def from_text(cls, data: str, escaped: bool=False) -> Self:
        """Load the struct from a string

        Args:
            data (str): The data for the struct

        Returns:
            Self: The loaded struct
        """
        if escaped:
            data = unescape_curly_braces(data)
        return cls(**json.loads(data))
    
    def to_text(self, escape: bool=False) -> str:
        """Dump the struct to a string

        Returns:
            str: The string
        """
        if escape:  
            return escape_curly_braces(self.to_dict())
        return self.model_dump_json()
    
    @classmethod
    def from_dict(cls, data: typing.Dict) -> Self:
        """Load the struct from a dictionary

        Args:
            data (typing.Dict): The dictionary containing the values

        Returns:
            Self: The result
        """
        return cls(**data)
    
    def to_dict(self) -> typing.Dict:
        """Convert the model to a dictionary

        Returns:
            typing.Dict: The model dumped
        """
        return self.model_dump()
    
    def render(self) -> str:
        """Render the struct for display

        Returns:
            str: The text version of the struct
        """
        return self.to_text(True)
    
    def forward(self, key, value) -> Self:

        if not hasattr(self, key):
            raise AttributeError(f'There is no attribute named {key}')
        setattr(self, key, value)
        return self

    # def __getitem__(self, key) -> typing.Any:
    #     """Get an attribute in 

    #     Args:
    #         key: The key to get

    #     Returns:
    #         typing.Any: Get attribute specified by key
    #     """
    #     return getattr(self, key)
    
    # def __setitem__(
    #     self, key, value
    # ) -> typing.Any:
    #     """Update a member of the Struct

    #     Args:
    #         key: The name of the value to update
    #         value: The value to update. 
    #             If it is a string and the member is a Str, it will be cast to a
    #             Str

    #     Returns:
    #         typing.Any: The value to set
    #     """
    #     if not hasattr(self, key):
    #         raise AttributeError(f'There is no attribute named {key}')
    #     setattr(self, key, value)
    #     return value


class Media:

    descr: str
    data: str


Content = typing.Union[Media, str, typing.List[typing.Union[Media, str]]]


@dataclass
class AIResponse(object):

    content: 'Message'
    source: typing.Dict
    val: typing.Any = None

    def clone(self) -> Self:

        return AIResponse(
            content=self.content,
            source=self.source,
            val=self.val
        )
    
    def __iter__(self) -> typing.Iterator:
        yield self.source
        yield self.content
        yield self.val


class Reader(Struct, ABC):
    """
    """

    name: str

    def example(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        return self.write_text(self.dump_data(data))

    @abstractmethod
    def dump_data(self, data: typing.Any) -> typing.Any:
        """Convert the data from the output of write_text
        to the original format

        Args:
            data (typing.Any): The data

        Returns:
            typing.Any: The data
        """
        pass

    @abstractmethod
    def write_text(self, data: typing.Any) -> str:
        """Write out the text for the data

        Args:
            data (typing.Any): The data to write the text for

        Returns:
            str: The text
        """
        pass

    def read(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        return self.load_data(self.read_text(message))
    
    @abstractmethod
    def read_text(self, message: str) -> typing.Any:
        """Read in the text and output to a "json" compatible format or a primitive

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The result of the reading process
        """
        pass

    @abstractmethod
    def load_data(self, data: typing.Dict) -> typing.Any:
        """Load the data output from reading the text

        Args:
            data (typing.Dict): The data to load (JSON format)

        Returns:
            typing.Any: The result of the reading
        """
        pass

    @abstractmethod
    def template(self) -> str:
        pass


class NullRead(Reader):
    """A Reader that does not change the data. 
    So in most cases will simply output a string
    """

    def dump_data(self, data: typing.Any) -> typing.Any:
        """Convert the data to JSON compatible format

        Args:
            data (typing.Any): The data to convert to "JSON" compatible format

        Returns:
            typing.Any: Returns the data passed in
        """
        pass

    def write_text(self, data: typing.Any) -> str:
        """Output the data to text

        Args:
            data (typing.Any): The JSON compatible data

        Returns:
            str: The data converted to text
        """
        pass

    def read_text(self, data: str) -> typing.Dict:
        """Read in the text as a JSON compatible structure

        Args:
            data (str): The data to read in

        Returns:
            typing.Dict: The JSON compatible object (does nothing because it is null)
        """
        return data
    
    def load_data(self, data) -> typing.Any:
        """Load the data

        Args:
            data: The data to load

        Returns:
            typing.Any: The data passed in (since null)
        """
        return data

    def template(self) -> str:
        return '<No Template>'


class AIPrompt(Struct, ABC):

    @abstractmethod
    def prompt(self, model: 'AIModel'):
        pass

    @abstractmethod
    def instruct(self, instruction: 'Instruction', model: 'AIModel'):
        pass

    @abstractmethod
    def reader(self) -> 'Reader':
        pass

    @abstractmethod
    def process_response(self, response: 'AIResponse') -> 'AIResponse':
        """

        Args:
            response (AIResponse): The response to process

        Returns:
            AIResponse: The updated response
        """
        pass

    @abstractmethod
    def clone(self) -> typing.Self:
        """Do a shallow clone of the message object

        Returns:
            typing.Self: The cloned object
        """
        pass

    @abstractmethod
    def aslist(self) -> typing.List['Message']:
        """
        """
        pass

    def __iter__(self) -> typing.Iterator['Message']:
        """

        Yields:
            Message: Each message
        """
        for message in self.aslist():
            yield message

    def __call__(self, prompt: 'AIPrompt') -> 'AIResponse':

        return self.forward(prompt)


class Message(Struct):

    source: str
    data: typing.Dict[str, typing.Any]

    def __getitem__(self, key: str):
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.data:
            return self.data[key]
        raise KeyError(f'{key}')

    def __setitem__(self, key: str, value: typing.Any):
        if hasattr(self, key):
            setattr(self, key, value)
        if key in self.data:
            self.data[key] = value
        raise KeyError(f'{key}')

    def prompt(self, model: 'AIModel', **kwarg_overrides) -> 'AIResponse':
        return model(self, **kwarg_overrides)

    def reader(self) -> 'Reader':
        return NullRead()

    def clone(self) -> typing.Self:
        """Do a shallow copy of the message

        Returns:
            Message: The cloned message
        """
        return self.__class__(
            source=self.source,
            data=self.data
        )

    def aslist(self) -> typing.List['Message']:
        return [self]
    
    def render(self) -> str:
        """Render the message

        Returns:
            str: Return the message and the source
        """
        return f'{self.source}: {render(self.data)}'


class TextMessage(Message):

    def __init__(self, source: str, text: typing.Union[str, 'Instruction']) -> 'Message':

        super().__init__(
            source=source,
            data={
                'text': text
            }
        )

    def reader(self) -> 'Reader':
        text = self['text']
        if isinstance(text, Instruction):
            return text.out
        return NullRead(name='')
    
    def render(self) -> str:
        """Render the text message

        Returns:
            str: Return the message and the text for the message
        """
        text = self.data['text']
        return f'{self.source}: {
            text.render() if isinstance(text, Instruction) else text
        }'


class Dialog(Struct):
    """A Dialog stores the interactions between the system/user and the assistant
    (i.e. the prompts and the responses)
    """

    messages: typing.List[Message] = pydantic.Field(default_factory=list)

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

    def add(self, message: Message, ind: typing.Optional[int]=None, replace: bool=False):
        """Add a message to the dialog

        Args:
            message (Message): The message to add
            ind (typing.Optional[int], optional): The index to add. Defaults to None.
            replace (bool, optional): Whether to replace at the index. Defaults to False.

        Raises:
            ValueError: If the index is not correct
        """
        if ind < 0:
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
            if isinstance(r, Instruction):
                return r.reader
        return NullRead(name='')
    
    def render(self) -> str:
        return '\n'.join(
            message.render() for message in self.messages
        )

    def instruct(
        self, instruct: 'Instruct', ai_model: 'AIModel', 
        ind: int=0, replace: bool=True
    ) -> AIResponse:
        """Instruct the AI

        Args:
            instruct (Instruct): The instruction to use
            ai_model (AIModel): The AIModel to use
            ind (int, optional): The index to set to. Defaults to 0.
            replace (bool, optional): Whether to replace at the index if already set. Defaults to True.

        Returns:
            AIResponse: The output from the AI
        """
        instruction = instruct.i()
        
        self.system(instruction, ind, replace)
        response = ai_model.forward(self.messages)
        response = self.process_response(response)
        self.assistant(response.content)
        return response

    def prompt(self, model: 'AIModel', append: bool=True) -> 'AIResponse':
        """Prompt the AI

        Args:
            model (AIModel): The model to usee
            append (bool, optional): Whether to append the output. Defaults to True.

        Returns:
            AIResponse: The response from the AI
        """
        response = model(self)

        if append:
            self.message('assistant', response.message())
        return response

    def aslist(self) -> typing.List['Message']:
        """Retrieve the message list

        Returns:
            typing.List[Message]: the messages in the dialog
        """
        return self.messages

    # def process_response(self, response: AIResponse) -> AIResponse:

    #     response = response.clone()

    #     out = None
    #     for r in reversed(self.messages):
    #         if  isinstance(r, Instruction):
    #             out = r.out
    #             break

    #     if out is None:
    #         response.val = response.content
    #     else:
    #         # Not sure about this
    #         response.val = out.read(response.content['text'])
    #     return response



Data = typing.Union[Struct, typing.List[Struct]]


class StructLoadException(Exception):
    """Exception StructLoad
    """

    def __init__(self, message="Struct loading failed.", errors=None):
        """Create a StructLoadException with a message

        Args:
            message (str, optional): The message. Defaults to "Struct loading failed.".
            errors (optional): The errors. Defaults to None.
        """
        super().__init__(message)
        self.errors = errors


def is_nested_model(
    pydantic_model_cls: typing.Type[Struct]
) -> bool:
    """Helper function to check if it is a nested model

    Args:
        pydantic_model_cls (typing.Type[Struct]): The class to check if it is a nested model

    Returns:
        bool: If it is a nested model
    """
    for field in pydantic_model_cls.model_fields.values():
        
        if isinstance(field.annotation, type) and issubclass(field.annotation, Struct):
            return True
    return False


class StructList(Struct, typing.Generic[S]):
    """
    """

    structs: typing.List[S]

    def __init__(self, structs: typing.Iterable):
        """

        Args:
            structs (typing.Iterable): 
        """
        super().__init__(structs=structs)

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


def is_undefined(val) -> bool:
    """
    Args:
        val : The value to check

    Returns:
        bool: Whether the value is undefined or not
    """
    return val == UNDEFINED or val == WAITING


# TODO: Make "struct" storable?
#  Module as well
class Storable(ABC):
    """Object to serialize objects to make them easy to recover
    """
    def __init__(self):
        """Create the storable object
        """
        self._id = str(uuid4())

    @property
    def id(self) -> str:
        return self._id

    def load_state_dict(self, state_dict: typing.Dict):
        """

        Args:
            state_dict (typing.Dict): 
        """
        for k, v in self.__dict__.items():
            if isinstance(v, Storable):
                self.__dict__[k] = v.load_state_dict(state_dict[k])
            else:
                self.__dict__[k] = state_dict[k]
        
    def state_dict(self) -> typing.Dict:
        """

        Returns:
            typing.Dict: 
        """
        cur = {}

        for k, v in self.__dict__.items():
            if isinstance(v, Storable):
                cur[k] = v.state_dict()
            else:
                cur[k] = v
        return cur


def render(x: typing.Any) -> typing.Union[str, typing.List[str]]:
    """Convert an input to text. Will use the text for an instruction,
    the render() method for a description and convert any other value to
    text with str()

    Args:
        value (X): The input

    Returns:
        str: The resulting text
    """
    if isinstance(x, Renderable):
        return x.render()
    
    elif is_primitive(x) or isinstance(x, dict) or isinstance(x, list):
        return escape_curly_braces(x)
    
    raise ValueError(
        f'Cannot render value of type {type(x)}'
    )


def render_multi(xs: typing.Iterable[typing.Any]) -> typing.List[str]:
    """Convert an input to text. Will use the text for an instruction,
    the render() method for a description and convert any other value to
    text with str()

    Args:
        value (X): The input

    Returns:
        str: The resulting text
    """

    return [
        render(x) for x in xs
    ]


class Module(ABC):
    """Base class for Modules
    """
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> typing.Any:
        """Execute the module

        Returns:
            typing.Any: The output of the module
        """
        pass

    def __call__(self, *args, **kwargs) -> typing.Any:
        """Execute the module

        Returns:
            typing.Any: The output of the module
        """
        return self.forward(*args, **kwargs)

    def parameters(self, recurse: bool=True) -> typing.Iterator['Param']:
        """Loop over the parameters for the module

        Yields:
            Param: The parameters for the module
        """
        yielded = set()
        for k, v in self.__dict__.items():
            if isinstance(v, Param):
                if id(v) in yielded:
                    continue
                yielded.add(id(v))
                
                yield v
            if recurse and isinstance(v, Module):
                for v in v.parameters(True):
                    if id(v) in yielded:
                        continue
                    yielded.add(id(v))
                    yield v

    def children(self, recurse: bool=True) -> typing.Iterator['Module']:
        """Loop over all of the child modules

        Yields:
            Module: The child module
        """
        yielded = set()
        for k, v in self.__dict__.items():
            if isinstance(v, Module):
                if id(v) in yielded:
                    continue
                yielded.add(id(v))
                if recurse:
                    for v in v.children(True):
                        if id(v) in yielded:
                            continue
                        yielded.add(id(v))
                        yield v
    
    async def async_forward(self, *args, **kwargs) -> typing.Any:
        """Execute the forward method asynchronously

        Returns:
            typing.Any: 
        """
        return self.forward(*args, **kwargs)

    def stream_forward(self, *args, **kwargs) -> typing.Iterator[
        typing.Tuple[typing.Any, typing.Any]
    ]:
        """Stream the output

        Yields:
            Iterator[typing.Iterator[ typing.Tuple[typing.Any, typing.Any] ]]: The current value and the change in the value
        """
        # default behavior doesn't actually stream
        res = self.forward(*args, **kwargs) 
        yield res, None

    def streamer(self, *args, **kwargs) -> 'Streamer':
        """Retrieve a streamer
        Returns:
            Streamer: The Streamer to loop over
        """
        return Streamer(
            iter(self.stream_forward(*args, **kwargs))
        )

    # async def async_stream_iter(self, *args, **kwargs) -> typing.AsyncIterator[
    #     typing.Tuple[typing.Any, typing.Any]
    # ]:
    #     # default behavior doesn't actually stream
    #     res = self.forward(*args, **kwargs) 
    #     yield res, None

    # async def async_stream_forward(self, *args, **kwargs) -> 'Streamer':
    #     """
    #     Returns:
    #         Streamer: The Streamer to loop over
    #     """
    #     return Streamer(
    #         self.stream_iter(*args, **kwargs)
    #     )


class AIModel(Module, ABC):
    """APIAdapter allows one to adapt various WebAPI or otehr
    API for a consistent interface
    """

    @abstractmethod
    def forward(self, prompt: AIPrompt) -> AIResponse:
        """Run a standard query to the API

        Args:
            data : Data to pass to the API

        Returns:
            typing.Dict: The result of the API call
        """
        pass

    @abstractmethod
    def convert(self, message: Message) -> typing.Dict:
        """Convert a message to the format needed for the model

        Args:
            messages (Message): The messages to convert

        Returns:
            typing.List[typing.Dict]: The format to pass to the "model"
        """
        pass

    def convert_messages(self, messages: typing.List[Message]) -> typing.List[typing.Dict]:
        """Convienence method to convert a list of messages to the format needed for the model

        Args:
            messages (typing.List[Message]): The messages to convert

        Returns:
            typing.List[typing.Dict]: The format to pass to the "model"
        """
        return [self.convert(message) for message in messages]

    def stream_forward(self, prompt: AIPrompt) -> typing.Iterator[typing.Tuple[AIResponse, AIResponse]]:
        """API that allows for streaming the response

        Args:
            prompt (AIPrompt): Data to pass to the API

        Returns:
            typing.Iterator: Data representing the streamed response
            Uses 'delta' for the difference. Since the default
            behavior doesn't truly stream. This must be overridden 

        Yields:
            typing.Dict: The data
        """
        result = self.forward(prompt)
        yield result, None
    
    async def async_forward(
        self, prompt: AIPrompt, **kwarg_override
    ) -> AIResponse:
        """Run this query for asynchronous operations
        The default behavior is simply to call the query

        Args:
            data: Data to pass to the API

        Returns:
            typing.Any: 
        """
        return self.forward(prompt, **kwarg_override)
    
    async def bulk_async_forward(
        self, prompt: typing.List[AIPrompt], **kwarg_override
    ) -> typing.List[AIResponse]:
        """

        Args:
            messages (typing.List[typing.List[Message]]): 

        Returns:
            typing.List[typing.Dict]: 
        """
        tasks = []
        async with asyncio.TaskGroup() as tg:

            for prompt_i in prompt:
                tasks.append(
                    tg.create_task(self.async_forward(prompt_i, **kwarg_override))
                )
        return list(
            task.result() for task in tasks
        )
    
    async def async_stream_iter(self, prompt: AIPrompt, **kwarg_override) -> typing.AsyncIterator[AIResponse]:
        """Run this query for asynchronous streaming operations
        The default behavior is simply to call the query

        Args:
            prompt (AIPrompt): The data to pass to the API

        Yields:
            typing.Dict: The data returned from the API
        """
        result = self.forward(prompt, **kwarg_override)
        yield result

    async def _collect_results(generator, index, results, queue):
        async for item in generator:
            results[index] = item
            await queue.put(results[:])  # Put a copy of the current results
        results[index] = None  # Mark this generator as completed

    async def bulk_async_stream_iter(
        self, prompts: typing.List[AIPrompt], **kwarg_override
    ) -> typing.AsyncIterator[typing.List[AIResponse]]:
        """Process multiple 

        Args:
            prompts (AIPrompt): The prompts to process

        Returns:
            typing.List[typing.Dict]: 
        """
        results = [None] * len(prompts)
        queue = asyncio.Queue()

        async with asyncio.TaskGroup() as tg:
            for index, prompt_i in enumerate(prompts):
                tg.create_task(self._collect_results(
                    self.async_stream_iter(prompt_i, **kwarg_override), index, results, queue)
                )

        active_generators = len(prompts)
        while active_generators > 0:
            current_results = await queue.get()
            yield current_results
            active_generators = sum(result is not None for result in current_results)


@dataclass
class Partial(object):
    """Class for storing a partial output from a streaming process
    """
    cur: typing.Any
    prev: typing.Any = None
    dx: typing.Any = None
    complete: bool = False


class Streamer(object):
    """Streamer is an object used to stream over the response
    """

    def __init__(self, stream: typing.Iterator):
        """The Stream to loop over

        Args:
            stream: The stream to loop over in generating the stream
        """
        self._stream = stream
        self._cur = None
        self._output = UNDEFINED
        self._prev = None
        self._dx = None

    @property
    def complete(self) -> bool:
        return self._output is not UNDEFINED

    def __call__(self) -> typing.Union[Partial]:
        """Query the streamer and returned updated value if updated

        Returns:
            typing.Union[typing.Any, Partial]: Get the next value in the stream
        """
        if self._output is not UNDEFINED:
            return self._output
        try:
            self._prev = self._cur
            self._cur, self._dx = next(self._stream)
            return Partial(self._cur, self._prev, self._dx, False)    
        except StopIteration:
            self._output = Partial(self._cur, self._prev, self._dx, True) 
            return self._output
        
    def __iter__(self) -> typing.Iterator[Partial]:

        while True:

            cur = self()
            if cur.complete:
                break
            yield cur


class Instruct(ABC):

    @abstractmethod
    def i(self) -> 'Instruction':
        pass


class Instruction(Struct, Instruct, typing.Generic[S]):
    """Specific instruction for the model to use
    """
    text: str
    out: typing.Optional[Reader] = None

    def i(self) -> Self:
        return self

    @pydantic.field_validator('text', mode='before')
    def convert_renderable_to_string(cls, v):
        if isinstance(v, Renderable):
            return v.render()
        if is_primitive(v):
            return str(v)
        return v

    def render(self) -> str:
        """Render the instruction

        Returns:
            str: The text for the instruction 
        """
        return self.text

    def read(self, data: str) -> S:
        """Read the data

        Args:
            data (str): The data to read

        Raises:
            RuntimeError: If the instruction does not have a reader

        Returns:
            S: The result of the read process
        """
        if self.out is None:
            raise RuntimeError(
                "Out has not been specified so can't read it"
            )
        
        return self.out.read(data)


class Param(Struct):
    """Param is used to wrap an instruction
    """
    name: str
    instruction: Instruction
    training: bool=False

    @pydantic.field_validator('instruction', mode='before')
    def convert_renderable_to_string(cls, v):
        if isinstance(v, Instruction):
            return v
        if isinstance(v, Renderable):
            return Instruction(text=v.render())
        if is_primitive(v):
            return Instruction(text=str(v))
        return v

    def update(self, text: str):
        if self.training:
            self.instruction.text = text

    def render(self) -> str:
        """

        Returns:
            str: 
        """
        return self.instruction.render()

    def read(self, data: typing.Dict) -> S:
        """Read in the data

        Args:
            data (typing.Dict): The data to read in

        Returns:
            S: The result of the reading
        """
        return self.instruction.read(data)

    def reads(self, data: str) -> S:
        return self.instruction.read_out(data)


# class StructModule(Struct, Module):

#     def forward(self, key: str, value: typing.Any) -> typing.Any:
        
#         copy = self.model_copy()
#         copy[key] = value
#         return copy
