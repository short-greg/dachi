# 1st party
from abc import ABC, abstractmethod
from typing import Self, get_type_hints
import typing
import inspect
import json

from uuid import uuid4
from enum import Enum


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
    
    elif is_primitive(x):
        return str(x)
    
    elif isinstance(x, dict) or isinstance(x, list):
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
        return data

    def write_text(self, data: typing.Any) -> str:
        """Output the data to text

        Args:
            data (typing.Any): The JSON compatible data

        Returns:
            str: The data converted to text
        """
        return data

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
        return None


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
    name: str
    instruction: Instruction
    training: bool=False
    text: str = None

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
            self.text = text

    def render(self) -> str:
        """

        Returns:
            str: 
        """
        if self.text is None:
            return self.instruction.render()
        return self.text

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
        print(self.__dict__)
        for k, v in self.__dict__.items():
            if isinstance(v, Module):
                if id(v) in yielded:
                    continue
                yield v
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
        res = self.forward(*args, **kwargs)
        return res

    def stream_forward(self, *args, **kwargs) -> typing.Iterator[
        typing.Tuple[typing.Any, typing.Any]
    ]:
        """Stream the output

        Yields:
            Iterator[typing.Iterator[ typing.Tuple[typing.Any, typing.Any] ]]: The current value and the change in the value
        """
        # default behavior doesn't actually stream
        res = self.forward(*args, **kwargs) 
        yield res, res

    # def streamer(self, *args, **kwargs) -> 'Streamer':
    #     """Retrieve a streamer

    #     Returns:
    #         Streamer: The Streamer to loop over
    #     """
    #     return Streamer(
    #         iter(self.stream_forward(*args, **kwargs))
    #     )


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

