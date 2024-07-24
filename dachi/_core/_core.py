# 1st party
import typing
from typing import get_type_hints
from typing_extensions import Self
from abc import ABC, abstractmethod
from uuid import uuid4
from enum import Enum
import inspect
import string
import json
import re


# 3rd party
import pydantic


class _Types(Enum):

    UNDEFINED = 'UNDEFINED'
    WAITING = 'WAITING'

UNDEFINED = _Types.UNDEFINED
WAITING = _Types.WAITING



class _Types(Enum):

    UNDEFINED = 'UNDEFINED'
    WAITING = 'WAITING'

UNDEFINED = _Types.UNDEFINED
WAITING = _Types.WAITING



# S = typing.TypeVar('S', bound=Struct)
S = typing.TypeVar('S', bound='Struct')
X = typing.Union[str, 'Description', 'Instruction']


class Renderable(ABC):

    @abstractmethod
    def render(self) -> str:
        pass


class _PartialFormatter(string.Formatter):
    def __init__(self):
        super().__init__()

    def format(self, format_string, *args, **kwargs):
        if args and kwargs:
            raise ValueError("Cannot mix positional and keyword arguments")

        self.args = args
        self.kwargs = kwargs
        return super().format(format_string)

    def get_value(self, key, args, kwargs):
        if isinstance(key, str):
            return self.kwargs.get(key, '{' + key + '}')
        if isinstance(key, int):
            return self.args[key] if key < len(self.args) else '{' + str(key) + '}'
        return super().get_value(key, args, kwargs)

    def __call__(self, format_string, *args, **kwargs):
        return self.format(format_string, *args, **kwargs)

# def get_variables(format_string) -> typing.List[str]:
#     # Ensure only named variables are used
#     if re.search(r'\{\d*\}', format_string):
#         raise ValueError("Only named variables are allowed")

#     # Extract named variables
#     variables = re.findall(r'\{([a-zA-Z_]\w*)\}', format_string)
    
#     return variables


# def get_str_variables(format_string):
#     # This regex matches anything inside curly braces { }
#     return re.findall(r'\{(.*?)\}', format_string)

def get_str_variables(format_string):
    has_positional = re.search(r'\{\d*\}', format_string)
    has_named = re.search(r'\{[a-zA-Z_]\w*\}', format_string)
    
    if has_positional and has_named:
        raise ValueError("Cannot mix positional and named variables")

    # Extract variables
    if has_positional:
        variables = [int(var) if var.isdigit() else None for var in re.findall(r'\{(\d*)\}', format_string)]
        if None in variables:
            variables = list(range(len(variables)))
    else:
        variables = re.findall(r'\{([a-zA-Z_]\w*)\}', format_string)
    
    return variables

    # # Ensure only named variables are used
    # if re.search(r'\{\d*\}', format_string):
    #     raise ValueError("Only named variables are allowed")

    # # Extract named variables
    # variables = re.findall(r'\{([a-zA-Z_]\w*)\}', format_string)
    
    # return variables


str_formatter = _PartialFormatter()


def model_template(model_cls: typing.Type[pydantic.BaseModel]) -> str:
    
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


class Struct(pydantic.BaseModel):

    model_config = pydantic.ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True
    )

    @classmethod
    def template(cls) -> str:
        return model_template(cls)
    
    def to_text(self) -> str:
        return str(self.model_dump())
    
    def __getitem__(self, key) -> typing.Any:
        """Get an attribute in 

        Args:
            key: The key to get

        Returns:
            typing.Any: Get attribute specified by key
        """
        return getattr(self, key)
    
    def __setitem__(
        self, key, value
    ) -> typing.Any:
        """Update a member of the Struct

        Args:
            key: The name of the value to update
            value: The value to update. 
                If it is a string and the member is a Str, it will be cast to a
                Str

        Returns:
            typing.Any: The value to set
        """
        if not hasattr(self, key):
            raise AttributeError('There is no')
        setattr(self, key, value)
        return value
    
    @classmethod
    def loads(cls, data: str) -> Self:
        return cls(**json.loads(data))
    
    def dumps(self) -> str:
        return self.model_dump_json()
    
    @classmethod
    def load(cls, data: typing.Dict) -> Self:
        return cls(**data)
    
    def dump(self) -> typing.Dict:
        return self.model_dump()

    @classmethod
    def from_text(cls, text: str) -> Self:
        return cls(
            **json.loads(text)
        )


class StructList(Struct, typing.Generic[S]):

    structs: typing.List[S]

    def __getitem__(self, key) -> typing.Any:
        """

        Args:
            key (_type_): 

        Returns:
            typing.Any: 
        """
        
        return self.structs[key]
    
    def __setitem__(self, key, value) -> typing.Any:
        
        if key is None:
            self.structs.append(value)
        else:
            self.structs[key] = value
        return value


def is_undefined(val) -> bool:
    """
    Args:
        val : The value to check

    Returns:
        bool: Whether the value is undefined or not
    """
    return val is UNDEFINED or val is WAITING


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


class Description(Struct, Renderable, ABC):
    """Provide context in the prompt template
    """
    name: str

    @abstractmethod
    def update(self, **kwargs) -> Self:
        pass

_primitives = (bool, str, int, float, type(None))


def is_primitive(obj):
    return type(obj) in _primitives


def render(x: typing.Union[X, typing.Iterable[X]]) -> typing.Union[str, typing.List[str]]:
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
    
    raise ValueError(
        f'Cannot render value of type {type(x)}'
    )
    # if isinstance(x, Instruction):
    #     return x.text
    # if isinstance(x, Description):
    #     return x.render()
    # if isinstance(x, Ref):
    #     return x.render()
    
def render_multi(xs: typing.Iterable[X]) -> typing.List[str]:
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


class Ref(Struct, Renderable):
    """Reference to another description.
    Useful when one only wants to include the 
    name of a description in part of the prompt
    """
    desc: Description

    @property
    def name(self) -> str:
        return self.desc.name

    def render(self) -> str:
        return self.desc.name

    def update(self, **kwargs) -> Self:
        # doesn't do anything since
        # it is a reference
        return self


def generic_class(t: typing.TypeVar, idx: int=0):

    return t.__orig_class__.__args__[idx]


class Out(Struct, typing.Generic[S]):

    out_cls: typing.Type[Struct]

    def read(self, data: typing.Dict) -> S:
        return self.out_cls.load(data)

    def reads(self, data: str) -> S:
        return self.out_cls.loads(data)

    def out_template(self) -> str:
        return self.out_cls.template()


class Style(Struct, typing.Generic[S], ABC):

    data: S

    @abstractmethod
    def dumps(self) -> str:
        pass

    @classmethod
    @abstractmethod
    def loads(cls, data: str) -> Self:
        pass

    def to_text(self) -> str:
        return self.data.to_text()


class Instruction(Struct, Renderable, typing.Generic[S]):
    """Specific instruction for the model to use
    """

    text: str
    out: typing.Optional[Out] = None

    @pydantic.field_validator('text', mode='before')
    def convert_renderable_to_string(cls, v):
        if isinstance(v, Renderable):
            return v.render()
        if is_primitive(v):
            return str(v)
        return v

    def render(self) -> str:
        return self.text

    def read(self, data: typing.Dict) -> S:
        if self.out is None:
            raise RuntimeError(
                "Out has not been specified so can't read it"
            )
        return self.out.read(data)

    def reads(self, data: str) -> S:
        if self.out is None:
            raise RuntimeError(
                "Out has not been specified so can't read it"
            )
        return self.out.reads(data)
    

class Param(Struct, Renderable):

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

        return self.instruction.render()

    def read(self, data: typing.Dict) -> S:
        return self.instruction.read(data)

    def reads(self, data: str) -> S:
        return self.instruction.reads(data)
