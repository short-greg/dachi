# 1st party
from abc import ABC, abstractmethod
import typing
import json
import inspect
from uuid import uuid4
from dataclasses import dataclass

# 3rd party
import pydantic

# local
from . import Renderable

S = typing.TypeVar('S', bound=pydantic.BaseModel)


class StructLoadException(Exception):
    """Exception StructLoad
    """

    def __init__(
        self, message="Struct loading failed.", errors=None
    ):
        """Create a StructLoadException with a message

        Args:
            message (str, optional): The message. Defaults to "Struct loading failed.".
            errors (optional): The errors. Defaults to None.
        """
        super().__init__(message)
        self.errors = errors


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
