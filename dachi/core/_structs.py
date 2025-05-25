from abc import ABC, abstractmethod
import typing
from typing import Dict, Literal
from uuid import uuid4
# process_core.py
from __future__ import annotations
import inspect, json
from typing import Any, Dict, Generic, List, TypeVar, get_type_hints, Literal
from pydantic import BaseModel, create_model, ConfigDict
from pydantic.generics import GenericModel
from pydantic_core import core_schema
import pydantic

from ._base import BaseItem, BaseProcess, BaseStruct, BaseSpec

import typing as t

V = typing.TypeVar("V", bound=BaseItem | BaseProcess)

class ItemList(BaseItem, t.Generic[V]):

    _element_type: t.ClassVar[t.Type[BaseItem]] = None

    def __init__(self, items: t.List[V], element_type: t.Type[BaseItem] = None):
        if not items and element_type is None:
            raise ValueError("Cannot infer element type from empty list")
        if element_type is None:
            element_type = type(items[0])
        self._element_type = element_type
        if not all(isinstance(i, BaseItem) for i in items):
            raise TypeError("All elements must be BaseItem instances.")
        self._items = list(items)

    @classmethod
    def to_schema(cls) -> typing.Type[pydantic.BaseModel]:
        if cls._element_type is None:
            raise TypeError("Element type not set for ItemList")
        return list[cls._element_type.to_spec_class()]
    
    def __getitem__(self, idx: int) -> V:
        return self._items[idx]

    def __setitem__(self, idx: int, value: V):
        if not isinstance(value, BaseItem):
            raise TypeError("Item must be a BaseItem")
        self._items[idx] = value

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def append(self, item: V):
        if not isinstance(item, BaseItem):
            raise TypeError("Item must be a BaseItem")
        self._items.append(item)
    
    def to_spec(self) -> BaseSpec:
        return [item.to_spec() for item in self._items]

    def from_spec(cls, spec: BaseSpec):
        pass
        # return [item.to_spec_instance() for item in self._items]

    def state_dict(self):
        return [item.state_dict() for item in self._items]

    def load_state_dict(self, state_list):
        for item, state in zip(self._items, state_list):
            item.load_state_dict(state)



T = t.TypeVar("T", bound=BaseItem | BaseProcess)

class ItemTuple(BaseStruct, t.Generic[T]):
    def __init__(self, items: t.Tuple[T, ...]):
        if not all(isinstance(i, BaseItem) for i in items):
            raise TypeError("All elements of ItemTuple must be BaseItem instances.")
        self._items: tuple[T, ...] = items

    def __getitem__(self, idx: int) -> T:
        return self._items[idx]

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def to_spec(self) -> BaseSpec:
        return [item.to_spec_instance() for item in self._items]

    def state_dict(self):
        return [item.state_dict() for item in self._items]

    def load_state_dict(self, state_list):
        for item, state in zip(self._items, state_list):
            item.load_state_dict(state)


K = t.TypeVar("K", bound=str)
V = t.TypeVar("V", bound=BaseItem | BaseProcess)

class ItemDict(BaseStruct, t.Generic[K, V]):
    def __init__(self, items: t.Dict[K, V]):
        if not all(isinstance(v, BaseItem) for v in items.values()):
            raise TypeError("All values in ItemDict must be BaseItem instances.")
        self._items: dict[K, V] = dict(items)

    def __getitem__(self, key: K) -> V:
        return self._items[key]

    def __setitem__(self, key: K, value: V):
        if not isinstance(value, BaseItem):
            raise TypeError("Item must be a BaseItem")
        self._items[key] = value

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def keys(self):
        return self._items.keys()

    def values(self):
        return self._items.values()

    def items(self):
        return self._items.items()

    def to_spec(self) -> BaseSpec:
        return {k: v.to_spec() for k, v in self._items.items()}
    
    @classmethod
    def to_schema(cls):
        if not cls._items:
            raise ValueError("Cannot infer spec class from empty ItemTuple")
        return tuple[item.to_spec_class() for item in self._items]

    def state_dict(self):
        return {k: v.state_dict() for k, v in self._items.items()}

    def load_state_dict(self, state_dict: dict[K, t.Any]):
        for k, v in state_dict.items():
            if k in self._items:
                self._items[k].load_state_dict(v)

