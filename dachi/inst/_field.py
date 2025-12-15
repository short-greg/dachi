from __future__ import annotations

import typing as t
from abc import abstractmethod
from pydantic import BaseModel, Field
import pydantic

from ._base import RespField


class BoundInt(RespField):
    """Integer field with min/max bounds."""

    min_val: int
    max_val: int

    def get_field(self) -> tuple:
        return (int, Field(description=self.description, ge=self.min_val, le=self.max_val))


class BoundFloat(RespField):
    """Float field with min/max bounds."""

    min_val: float
    max_val: float

    def get_field(self) -> tuple:
        return (float, Field(description=self.description, ge=self.min_val, le=self.max_val))


class TextField(RespField):
    """String text field."""

    def get_field(self) -> tuple:
        return (str, Field(description=self.description))


class BoolField(RespField):
    """Boolean field."""

    def get_field(self) -> tuple:
        return (bool, Field(description=self.description))


class TypedDictField(RespField):
    """Typed dictionary field."""

    typed_dict: t.Type[dict]

    def get_field(self) -> tuple:
        return (self.typed_dict, Field(description=self.description))


class DictField(RespField):
    """Dictionary field for dynamic key-value pairs."""

    value_type: t.Type = str
    max_length: int | None = None
    min_length: int | None = None

    def get_field(self) -> tuple:
        return (t.Dict[str, self.value_type], Field(description=self.description, min_length=self.min_length, max_length=self.max_length))


class ListField(RespField):
    """List field."""

    item_type: t.Type = str
    max_len: int | None = None
    min_len: int | None = None

    def get_field(self) -> tuple:
        return (t.List[self.item_type], Field(description=self.description, default_factory=list, min_items=self.min_len, max_items=self.max_len))


class TupleField(RespField):
    """Tuple field."""

    item_types: t.List[t.Type] = [str]

    def get_field(self) -> tuple:
        return (t.Tuple[tuple(self.item_types)], Field(description=self.description))
