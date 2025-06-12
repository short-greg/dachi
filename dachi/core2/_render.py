
# 1st party
import typing
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Literal
from uuid import uuid4
import inspect
import typing as t
from typing import Any, Dict, get_type_hints, Literal
import sys
# 3rd party
from pydantic import (
    BaseModel, create_model, ConfigDict, Field
)
# from pydantic.generics import GenericModel
from pydantic.fields       import FieldInfo
import pydantic

from ._base4 import Renderable
from enum import Enum

# local
from ..utils import escape_curly_braces, unescape_curly_braces, is_primitive



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


def render(
    x: typing.Any, escape_braces: bool=True, 
    template_render: typing.Optional[typing.Callable[['TemplateField'], str]]=None
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
        return model_to_text(x, escape_braces)
    elif is_primitive(x):
        return str(x)
    elif isinstance(x, typing.Dict):
        items = {}
        for k, v in x.items():
            if isinstance(v, str):
                v = f'"{v}"'
            else:
                v = render(v, escape_braces)
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
            if isinstance(v, str):
                v = f'"{v}"'
            else:
                v = render(v, escape_braces)
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
