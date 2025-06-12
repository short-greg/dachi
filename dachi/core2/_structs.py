from __future__ import annotations
import typing as t
from pydantic import BaseModel
from ._base4 import BaseModule, Param, State, Shared, BuildContext, BaseSpec, registry  # adjust import path
from typing import TypeVar, Generic, Iterable, ClassVar, Iterator
from pydantic import BaseModel, Field, ConfigDict, create_model, field_validator


import typing as t
from dataclasses import InitVar
from uuid import uuid4

from ._base4 import BaseModule, BuildContext, registry, BaseSpec

from typing import Optional, Union, List, Iterator, Iterable
from dataclasses import InitVar

V_co = t.TypeVar("V_co", bound=BaseModule, covariant=True)
V = t.TypeVar("V", bound=BaseModule)
T = TypeVar("T", bound=BaseModule)


class ModuleList(BaseModule, t.Generic[V]):
    """
    A list-like container whose elements are themselves `BaseModule`
    instances.  Works seamlessly with the new serialization / dedup rules.
    """
    __spec_hooks__: ClassVar[t.List[str]] = ["items"]
    items: InitVar[list[V]]

    def __post_init__(self, items: Optional[Iterable[T]] = None):
        self._module_list = []

        if items is not None:
            for m in items:
                self.append(m)

    @classmethod
    def __build_schema_hook__(cls, name: str, type_: t.Any, default: t.Any):
        if name != "items":
            raise ValueError(f"No hook specified for {name}")
        return list[BaseSpec]

    def __len__(self) -> int:  # Positive test: len reflects number added
        return len(self._module_list)

    def __iter__(self) -> Iterator[T]:  # Positive test: order preserved
        return iter(self._module_list)

    def __getitem__(self, idx: int) -> T:  # Edge test: negative index ok
        return self._module_list[idx]

    def __setitem__(self, idx: int, value: V):
        if not isinstance(value, BaseModule):
            raise TypeError("ModuleList accepts only BaseModule instances")
        if idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds.")
        # unregister old, register new
        old_key = str(idx)
        del self._modules[old_key]
        self._module_list[idx] = value
        self.register_module(old_key, value)

    # public API – intentionally *append‑only*
    def append(self, module: V):

        if not isinstance(module, BaseModule):
            raise TypeError("ModuleList accepts only BaseModule instances")
        key = str(len(self._module_list))
        self._module_list.append(module)
        self.register_module(key, module)

    def spec_hook(
        self, *, 
        name: str,
        val: t.Any,
        to_dict: bool = False,
    ):
        """
        Serialise *this* runtime object → its spec counterpart.

        Nested `BaseModule` instances are recursively converted.
        `ModuleList` containers are converted element-wise.
        """
        if name == "items":
            # Special case for _items, which is a list of modules
            if isinstance(val, list):
                val = [
                    item.spec(to_dict=to_dict) 
                    for item in self._module_list
                ]
            else:
                raise TypeError(f"Expected _items to be a list, got {type(val)}")
        else:
            raise ValueError(f"Unknown spec hook name: {name}")
        return val

    @classmethod
    def from_spec_hook(
        cls,
        name: str,
        val: t.Any,
        ctx: "dict | None" = None,
    ) -> t.Any:
        """
        Hook for the registry to call when a spec is encountered.
        This is used to create a ModuleList from a spec.
        """
        res = None
        if name == "items":
            if isinstance(val, list):
                res = []
                for item in val:
                    cur_item = ctx.get(item.id)
                    if cur_item is None:
                        cur_item = registry[item.kind].obj.from_spec(item, ctx) 
                        ctx[item.id] = cur_item
                    res.append(cur_item)
                
            else:
                raise TypeError(f"Expected _items to be a list, got {type(val)}")
        else: 
            raise ValueError(f"Unknown spec hook name: {name}")
        return res


class ModuleDict(BaseModule, t.Generic[V]):
    """
    A dict-like container whose values are themselves `BaseModule`
    instances. Keys must be strings.
    """

    __spec_hooks__: ClassVar[t.List[str]] = ["items"]
    items: InitVar[dict[str, V]]

    def __post_init__(self, items: Optional[dict[str, V]] = None):
        self._module_dict = {}

        if items is not None:
            for k, m in items.items():
                self[k] = m

    @classmethod
    def __build_schema_hook__(cls, name: str, type_: t.Any, default: t.Any):
        if name != "items":
            raise ValueError(f"No hook specified for {name}")
        return dict[str, BaseSpec]

    def __getitem__(self, key: str) -> V:
        return self._module_dict[key]

    def __setitem__(self, key: str, val: V):
        if not isinstance(key, str):
            raise TypeError("Keys must be strings")
        if not isinstance(val, BaseModule):
            raise TypeError("Values must be BaseModule instances")
        self._module_dict[key] = val
        self.register_module(key, val)

    def __iter__(self):
        return iter(self._module_dict)

    def __len__(self):
        return len(self._module_dict)

    def keys(self):
        return self._module_dict.keys()

    def values(self):
        return self._module_dict.values()

    def items(self):
        return self._module_dict.items()

    def spec_hook(
        self,
        *,
        name: str,
        val: t.Any,
        to_dict: bool = False,
    ):
        if name == "items":
            return {
                k: v.spec(to_dict=to_dict)
                for k, v in self._module_dict.items()
            }
        raise ValueError(f"Unknown spec hook name: {name}")

    @classmethod
    def from_spec_hook(
        cls,
        name: str,
        val: t.Any,
        ctx: "dict | None" = None,
    ) -> dict[str, V]:
        if name != "items":
            raise ValueError(f"Unknown spec hook name: {name}")
        if not isinstance(val, dict):
            raise TypeError(f"Expected a dict for 'items', got {type(val)}")

        ctx = ctx or {}
        out = {}
        for k, v in val.items():
            if isinstance(v, BaseSpec):
                id_ = v.id
                if id_ in ctx:
                    out[k] = ctx[id_]
                else:
                    mod = registry[v.kind].obj.from_spec(v, ctx)
                    ctx[id_] = mod
                    out[k] = mod
            else:
                raise TypeError(f"Expected BaseSpec values in dict, got {type(v)}")
        return out
