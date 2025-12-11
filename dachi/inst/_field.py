from __future__ import annotations

import typing as t
from abc import abstractmethod
from pydantic import BaseModel, Field
import pydantic

from ._base import EvalField


class BoundInt(EvalField):
    """Integer field with min/max bounds."""

    min_val: int
    max_val: int

    def get_field(self) -> tuple:
        return (int, Field(description=self.description, ge=self.min_val, le=self.max_val))


class BoundFloat(EvalField):
    """Float field with min/max bounds."""

    min_val: float
    max_val: float

    def get_field(self) -> tuple:
        return (float, Field(description=self.description, ge=self.min_val, le=self.max_val))


class TextField(EvalField):
    """String text field."""

    def get_field(self) -> tuple:
        return (str, Field(description=self.description))


class BoolField(EvalField):
    """Boolean field."""

    def get_field(self) -> tuple:
        return (bool, Field(description=self.description))


class DictField(EvalField):
    """Dictionary field for dynamic key-value pairs."""

    value_type: t.Type = str

    def get_field(self) -> tuple:
        return (t.Dict[str, self.value_type], Field(description=self.description))


class ListField(EvalField):
    """List field."""

    item_type: t.Type = str

    def get_field(self) -> tuple:
        return (t.List[self.item_type], Field(description=self.description, default_factory=list))
