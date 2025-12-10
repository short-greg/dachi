# 1st party
from __future__ import annotations
from abc import abstractmethod, ABC
import typing as t
import pydantic
from enum import Enum
from abc import ABC, abstractmethod

from abc import abstractmethod
from enum import Enum
from pydantic import BaseModel


# 3rd Party
# , ConfigDict, create_model

# Local


# from ._restricted_schema import RestrictedSchemaMixin  # mix‑in defined in previous patch

"""Drop‑in core definitions for process‑style objects and shareable leaves.

Usage::

    from baseitem_core import BaseItem, Param, State, Shared

    class MyProc(BaseItem):
        weight: Param[float]
        steps:  State[int]
        cfg:    Shared[str]
        name:   str

    p = MyProc(weight=Param(val=1.0), steps=State(val=0), cfg=Shared(val="foo"), name="proc")
    print(p.spec().model_dump())
    print(p.state_dict())
"""

T = t.TypeVar("T")
CORE_TYPE = t.TypeVar("J", bound=t.Union[BaseModel, dict, str, int, float, bool])

# class _Types(Enum):

#     UNDEFINED = 'UNDEFINED'
#     WAITING = 'WAITING'

# UNDEFINED = _Types.UNDEFINED
# """Constant for UNDEFINED. usage: value is UNDEFINED"""
# WAITING = _Types.WAITING
# """Constant for WAITING when streaming. usage: value is WAITING"""

class StorableState(ABC):
    """Mixin for classes that implement state_dict() and load_state_dict() methods."""

    @abstractmethod
    def state_dict(self, *, recurse: bool = True) -> dict:
        """Return a dictionary representing the state of the object."""
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: dict, *, recurse: bool = True):
        """Load the state of the object from a dictionary."""
        pass


def to_kind(cls): 
    """Convert a class to its kind."""

    if isinstance(cls, type) and issubclass(cls, pydantic.BaseModel):
        module_ = cls.__module__
        cls_name = cls.__qualname__.split('[')[0]
        qual_name = f'{module_}.{cls_name}'
        return qual_name
    
    return cls.__qualname__


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


class ExampleMixin(ABC):
    """A mixin to indicate that the class 
    has an example function
    """
    @abstractmethod
    def example(self) -> str:
        """Get the template 

        Returns:
            str: 
        """
        pass


class TokenType(str, Enum):
    END = 'END_TOK'
    NULL = 'NULL_TOK'

END_TOK = TokenType.END
NULL_TOK = TokenType.NULL


@pydantic.dataclasses.dataclass
class Msg:
    """A message in a dialog
    """
    role: str


@pydantic.dataclasses.dataclass
class TextMsg(Msg):
    """A message in a dialog
    """
    role: str
    text: str


ITEM = t.TypeVar('ITEM')
Inp: t.TypeAlias = pydantic.BaseModel | t.Dict[str, t.Any] | str | Msg
