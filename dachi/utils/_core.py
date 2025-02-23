# 1st party
from abc import ABC
import typing
from uuid import uuid4

from abc import ABC, abstractmethod
import json
from dataclasses import dataclass
import inspect
import typing
from typing import Self, get_type_hints

# 3rd party
import pydantic

# local
from . import (
    is_primitive, 
    escape_curly_braces
)

S = typing.TypeVar('S', bound=pydantic.BaseModel)


import pydantic
from . import unescape_curly_braces, escape_curly_braces


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



class Renderable(ABC):
    """Mixin for classes that implement the render()
    method. Render is used to determine how to represent an
    object as a string to send to thte LLM
    """

    @abstractmethod
    def render(self) -> str:
        """Convert an object to a string representation for 
        an llm

        Returns:
            str: the string representation of the object
        """
        pass


class Templatable(ABC):
    """A mixin to indicate that the class 
    has a template function defined. Templates are
    used by the LLM to determine how to output.
    """

    @abstractmethod
    def template(self) -> str:
        """Get the template 

        Returns:
            str: 
        """
        pass


@dataclass
class TemplateField(Renderable):
    """Use for rendering a field in a BaseModel
    """
    type_: str
    description: str
    default: typing.Any = None
    is_required: bool = True

    def to_dict(self) -> typing.Dict:
        """Convert the template to a dict

        Returns:
            typing.Dict: the template
        """
        return {
            'type': self.type_,
            'description': self.description,
            'default': self.default,
            'is_required': self.is_required
        }
    
    def render(self) -> str:
        """Convert the template to a string

        Returns:
            str: The string of the template.
        """
        return str(self.to_dict())


def model_template(model_cls: typing.Type[pydantic.BaseModel]) -> str:
    """Get the template for a pydantic.BaseModel

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


def struct_template(model: pydantic.BaseModel) -> typing.Dict:
    """Get the template for the Struct

    Returns:
        typing.Dict: The template 
    """
    template = {}
    
    base_template = model_template(model)
    for field_name, field in model.model_fields.items():
        field_type = field.annotation
        if isinstance(field_type, type) and issubclass(field_type, pydantic.BaseModel):

            template[field_name] = struct_template(field_type)
        else:

            if 'is_required' in base_template[field_name]:
                is_required = base_template[field_name]['is_required']
            else:
                is_required = True
            template[field_name] = TemplateField(
                type_=field.annotation,
                description=field.description,
                default=field.default if field.default is not None else None,
                is_required=is_required
            )

    return template


def model_to_text(model: pydantic.BaseModel, escape: bool=False) -> str:
    """Dump the struct to a string

    Returns:
        str: The string
    """
    if escape:  
        return escape_curly_braces(model.model_dump())
    return model.model_dump_json()


def model_from_text(model_cls: typing.Type[pydantic.BaseModel], data: str, escaped: bool=False) -> Self:
    """Load the struct from a string

    Args:
        data (str): The data for the struct

    Returns:
        Self: The loaded struct
    """
    if escaped:
        data = unescape_curly_braces(data)
    return model_cls(**json.loads(data))


def doc(obj) -> str:
    """utility to get the docstring for the object

    Args:
        obj: the object to get the docstring for

    Returns:
        str: The docstring for the object
    """
    d = obj.__doc__
    return d if d is not None else ''


class Storable(ABC):
    """Object to serialize objects to make them easy to recover
    """
    def __init__(self):
        """Create the storable object
        """
        self._id = str(uuid4())

    @property
    def id(self) -> str:
        """The object id of the storable

        Returns:
            str: The ID
        """
        return self._id

    def load_state_dict(self, state_dict: typing.Dict):
        """Load the state dict for the object

        Args:
            state_dict (typing.Dict): The state dict
        """
        for k, v in self.__dict__.items():
            if isinstance(v, Storable):
                self.__dict__[k] = v.load_state_dict(state_dict[k])
            else:
                self.__dict__[k] = state_dict[k]
        
    def state_dict(self) -> typing.Dict:
        """Retrieve the state dict for the object

        Returns:
            typing.Dict: The state dict
        """
        cur = {}

        for k, v in self.__dict__.items():
            if isinstance(v, Storable):
                cur[k] = v.state_dict()
            else:
                cur[k] = v
        return cur


def render(
    x: typing.Any, escape_braces: bool=True, 
    template_render: typing.Optional[typing.Callable[[TemplateField], str]]=None
) -> typing.Union[str, typing.List[str]]:
    """Convert an input to text. Will use the text for a cue,
    the render() method for a description and convert any other value to
    text with str()

    Args:
        value (X): The input

    Returns:
        str: The resulting text
    """
    if isinstance(x, TemplateField):
        if template_render is not None:
            x = template_render(x)
        else: 
            x = x.render()

    if isinstance(x, Renderable):
        return x.render()

    elif isinstance(x, pydantic.BaseModel):
        return model_to_text(x, escape_curly_braces)
    elif is_primitive(x):
        return str(x)
    elif isinstance(x, typing.Dict):
        items = {}
        for k, v in x.items():
            v = render(v, escape_braces)
            if isinstance(v, str):
                v = f'"{v}"'
            items[k] = v    
        items = ', '.join(
            f'"{k}": {v}' 
            for k, v in items.items()
        )

        if escape_braces:
            return f"{{{{{items}}}}}"
        else:
            return f'{{{items}}}'
    elif isinstance(x, typing.List):

        items = []
        for v in x:
            v = render(v, escape_braces)
            if isinstance(v, str):
                v = f'"{v}"'
            items.append(v)

        return '[{}]'.format(', '.join(render(v) for v in items))
    elif isinstance(x, Renderable):
        return x.render()
    return str(x)


def is_renderable(obj: typing.Any) -> bool:
    """Return whether an object is renderable

    Args:
        obj (typing.Any): The object to check

    Returns:
        bool: whether the object is renderable
    """

    return (
        isinstance(obj, Renderable)
        or is_primitive(obj)
        or isinstance(obj, list)
        or isinstance(obj, dict)
        or isinstance(obj, pydantic.BaseModel)
    )


def render_multi(xs: typing.Iterable[typing.Any]) -> typing.List[str]:
    """Convert an input to text. Will use the text for an cue,
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


class _Final:
    """A unique object to mark the end of a streaming response."""
    def __repr__(self):
        return "<Final Token>"

END_TOK = _Final()

