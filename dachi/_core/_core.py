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


def get_str_variables(format_string: str) -> typing.List[str]:
    """Get the variables in a string to format

    Args:
        format_string (str): The string to get variables for 

    Raises:
        ValueError: If the string has both positional and named
        variables

    Returns:
        typing.List[str]: The list of variables
    """
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


str_formatter = _PartialFormatter()


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


def escape_curly_braces(value: typing.Any, render: bool=False) -> str:
    """Escape curly braces for dictionary-like structures."""
    if isinstance(value, str):
        return value
    if isinstance(value, typing.Dict):
        items = ', '.join(f"'{k}': {escape_curly_braces(v)}" for k, v in value.items())
        return f"{{{{{items}}}}}"
    if isinstance(value, typing.List):
        return '[{}]'.format(', '.join(escape_curly_braces(v) for v in value))
    if render:
        return render(value)
    return str(value)


def unescape_curly_braces(value: typing.Any) -> str:
    """Invert the escaping of curly braces."""
    if isinstance(value, str):
        return value.replace('{{', '{').replace('}}', '}')
    return value


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
        """Load the struct from a string

        Args:
            data (str): The data for the struct

        Returns:
            Self: The loaded struct
        """
        return cls(**json.loads(data))
    
    def dumps(self) -> str:
        """Dump the struct to a string

        Returns:
            str: The string
        """
        return self.model_dump_json()
    
    @classmethod
    def load(cls, data: typing.Dict) -> Self:
        """Load the struct from a dictionary

        Args:
            data (typing.Dict): The dictionary containing the values

        Returns:
            Self: The result
        """
        return cls(**data)
    
    def dump(self) -> typing.Dict:
        """Convert the model to a dictionary

        Returns:
            typing.Dict: The model dumped
        """
        return self.model_dump()

    @classmethod
    def from_text(cls, text: str) -> Self:
        """Create the struct from text that has been
        escaped with curly braces

        Args:
            text (str): 

        Returns:
            Self: 
        """
        return cls(
            **json.loads(unescape_curly_braces(text))
        )

    def to_text(self) -> str:
        """Convert the struct to text with escaped curly
        braces

        Returns:
            str: The text for the model
        """
        model_dict = self.model_dump()
        escaped_str = escape_curly_braces(model_dict)
        return escaped_str
    
    def render(self) -> str:
        """Render the struct for display

        Returns:
            str: The text version of the struct
        """
        return self.to_text()


class StructLoadException(Exception):

    def __init__(self, message="Struct loading failed.", errors=None):
        super().__init__(message)
        self.errors = errors


def is_nested_model(pydantic_model_cls: typing.Type[Struct]) -> bool:
    for field in pydantic_model_cls.model_fields.values():
        
        if isinstance(field.annotation, type) and issubclass(field.annotation, Struct):
            return True
    return False

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
    
    @classmethod
    def load_records(cls, records) -> 'StructList[S]':

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
    name: str = pydantic.Field(description='The name of the description.')

    @abstractmethod
    def update(self, **kwargs) -> Self:
        pass

    @abstractmethod
    def render(self) -> str:
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


class Ref(Struct):
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


class Result(Struct, ABC):

    @abstractmethod
    def write(self, data: Struct) -> str:
        pass

    @abstractmethod
    def read(self, data: str) -> S:
        pass

    @abstractmethod
    def stream_from_text(self, data: str) -> S:
        pass

    @abstractmethod
    def out_template(self) -> str:
        pass


class Out(Result, typing.Generic[S]):

    out_cls: typing.Type[S]

    def to_text(self, data: S) -> str:
        return data.to_text()

    def write(self, data: S) -> str:
        return data.dumps()

    def read(self, data: str) -> S:
        return self.out_cls.loads(data)

    def stream_read(self, data: str) -> S:
        return self.out_cls.loads(data)

    def out_template(self) -> str:
        return self.out_cls.template()


class ListOut(Result, typing.Generic[S]):

    out_cls: typing.Type[S]

    def write(self, data: StructList[S]) -> str:
        return json.dumps(data)

    def read(self, data: str) -> StructList[S]:
        return StructList.load_cls(
            self.out_cls, data
        )

    def to_text(self, data: StructList[S]) -> str:
        return data.to_text()

    def stream_read(self, data: str) -> S:
        return StructList.load_cls(
            self.out_cls, data
        )
    
    def out_template(self) -> str:
        return self.out_cls.template()


class MultiOut(Result):
    
    outs: typing.List[Out]
    names: typing.List[str]
    conn: str = '::OUT::{name}::\n'
    signal: str = '\u241E'
    
    @pydantic.field_validator('names', 'outs', mode='before')
    def validate_names_types_data(cls, values):
        names = values.get('names', [])
        outs = values.get('outs', [])
        
        if len(names) != len(outs):
            raise ValueError("The number of names must match the number of types")

        return values

    def write(self, data: typing.List[Struct]) -> str:
        result = ''
        for struct, name in zip(data, self.names):
            result = result + '\n' + self.signal + self.conn.format(name=name)
            result = f'{result}\n{struct.render()}'

        return result

    def read(self, data: str) -> typing.List[Struct]:

        structs = []

        d = data
        for t, name in zip(self.outs, self.names):
            from_loc = d.find('\u241E')
            to_loc = d[from_loc + 1:].find('\u241E')
            cur = self.conn.format(name=name)
            data_loc = from_loc + len(cur)
            data_str = d[data_loc:to_loc]
            structs.append(t.read(data_str))
            d = d[to_loc:]

        return structs

    def to_text(self, data: typing.List[S]) -> str:

        text = ""
        for data_i, out, name in zip(data, self.outs, self.names):
            cur = out.render(data_i)
            cur_conn = self.conn.format(name)
            text += f"""
            {self.signal}{cur_conn}
            {cur}
            """
        return text

    def stream_read(self, data: str) -> typing.Tuple[S, bool, str]:
        structs = []

        d = data
        for i, (t, name) in enumerate(zip(self.outs, self.names)):
            from_loc = d.find('\u241E')
            to_loc = d[from_loc + 1:].find('\u241E')
            cur = self.conn.format(name=name)
            data_loc = from_loc + len(cur)
            data_str = d[data_loc:to_loc]
            try: 
                structs.append(t.read(data_str))
            except StructLoadException as e:
                return structs, i
            d = d[to_loc:]

        return structs, None
    
    def out_template(self) -> str:

        text = ""
        for out, name in zip(self.outs, self.names):
            cur = out.out_template()
            cur_conn = self.conn.format(name)
            text += f"""
            {self.signal}{cur_conn}
            {cur}
            """
        return text


class JSONOut(Out):

    def stream_read(self, text: str) -> typing.Tuple[
        typing.Optional[typing.Dict], bool
    ]:
        try: 
            result = json.loads(text)
            return result, True
        except json.JSONDecodeError:
            return None, False

    def read(self, text: str) -> typing.Dict:
        result = json.loads(text)
        return result

    def write(self, data: Struct) -> str:
        return data.dumps()
    
    def template(self, out_cls: Struct) -> str:
        return out_cls.template()


class Instruction(Struct, typing.Generic[S]):
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




class Param(Struct):

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


# class IO(Struct):

#     @abstractmethod
#     def stream_read(self, text: str) -> typing.Dict:
#         pass

#     @abstractmethod
#     def read(self, text: str) -> typing.Dict:
#         pass

#     @abstractmethod
#     def write(self, data: typing.Dict) -> str:
#         pass

#     @abstractmethod
#     def template(self, out_cls: Struct) -> str:
#         return out_cls.template()



# # Old functionality for MultiOut

# @classmethod
# def reads(cls, data: str) -> 'MultiOut':


# @classmethod
# def stream_reads(cls, stream: typing.Iterable[str], types: typing.List[S]) -> typing.Tuple['MultiOut', int, bool]:
#     names = []
#     structs = StructList(structs=[])
#     lines = []
#     read_count = 0
#     ended_in_failure = False
    
#     i = 0
#     for line in stream:
#         lines.append(line.strip())
#         if len(lines) == 2:
#             try:
#                 conn_line, struct_line = lines
#                 try:
#                     name = conn_line.split('::')[1]
#                     names.append(name)
#                 except IndexError:
#                     raise ValueError("Invalid format for connection line")

#                 # Create a Struct object from the struct line
#                 types[i].reads()
#                 struct = Struct.from_text(struct_line)
#                 structs.add_struct(struct)

#                 read_count += 1
#                 lines = []
#             except Exception as e:
#                 ended_in_failure = True
#                 break

#     instance = cls()
#     instance.names = names
#     instance.data = structs
#     return instance, read_count, ended_in_failure

# def writes(self) -> str:
#     result = ''
#     if self.names is None:
#         names = []
#     else:
#         names = self.names
#     if len(names) < self.data.structs:
#         residual = range(len(names), len(self.data.structs))
#         names = [*names, *residual]
#     for struct, name in zip(self.data.structs, names):
#         result = result + self.conn.format(name=name)
#         result = f'{result}\n{struct.render()}'

#     return result

