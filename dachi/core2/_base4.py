from __future__ import annotations

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

from dataclasses import InitVar
import inspect
import typing as t
from uuid import uuid4

try:  # 3.12+
    from typing import dataclass_transform
except ImportError:  # 3.8–3.11
    from typing_extensions import dataclass_transform

from pydantic import BaseModel, Field, ConfigDict, create_model

T = t.TypeVar("T")

# -----------------------------------------------------------
# Shareable leaf hierarchy
# -----------------------------------------------------------

class ShareableItem(BaseModel, t.Generic[T]):
    """Lightweight, serialisable leaf object with a single ``val`` field."""

    data: T

    class Config:
        arbitrary_types_allowed = False  # allow torch tensors, numpy arrays, Path, …
        frozen = False                   # hashable & immutable, good for caching


class Param(ShareableItem[T]):
    """Trainable parameter; ``training`` may be toggled to freeze it."""

    training: bool = Field(True, description="Participates in optimisation?")


class State(ShareableItem[T]):
    """Mutable runtime state (e.g. counters, RNG seeds, rolling averages)."""
    pass


class Shared(ShareableItem[T]):
    """Pointer‑like wrapper whose value should *not* enter ``state_dict``."""

    ref_name: str | None = Field(
        None,
        description="Optional global reference – e.g. hash or registry key.",
    )


# -----------------------------------------------------------
# BaseSpec – schema node emitted by BaseItem.spec()
# -----------------------------------------------------------

class BaseSpec(BaseModel):
    kind: str
    id: str = Field(default_factory=lambda: str(uuid4()))
    style: t.Literal["structured"] = "structured"

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)


# -----------------------------------------------------------
# BaseItem – runtime process node
# -----------------------------------------------------------

@dataclass_transform(kw_only_default=True, field_specifiers=(Field,))
class BaseItem:
    """Dataclass‑like runtime object without exec‑generated ``__init__``."""

    # populated by __init_subclass__
    __spec__: t.ClassVar[type[BaseSpec]]
    __item_fields__: t.ClassVar[list[tuple[str, t.Any, t.Any, bool]]]
    __is_initvar__: t.ClassVar[dict[str, bool]]

    # per‑instance containers
    # _children: list["BaseItem"]

    # ---------------------------------------------------
    # class construction hook
    # ---------------------------------------------------

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls is BaseItem:
            return
        ann = t.get_type_hints(cls, include_extras=True)
        fields: list[tuple[str, t.Any, t.Any, bool]] = []

        for name, typ in ann.items():
            # skip private attrs and typing.ClassVar
            if t.get_origin(typ) is t.ClassVar:
                continue
            default = getattr(cls, name, inspect._empty)

            # Detect InitVar from dataclasses, not typing
            is_init = isinstance(typ, InitVar)
            if is_init:
                typ = t.get_args(typ)[0] if t.get_origin(typ) is InitVar else t.Any
            fields.append((name, typ, default, is_init))

        cls.__item_fields__ = fields
        cls.__is_initvar__ = {n: is_init for n, *_, is_init in fields}

        # build & cache pydantic spec model
        spec_fields: dict[str, tuple[t.Any, t.Any]] = {}

        print('=====')
        for n, typ, dflt, _ in fields:
            if isinstance(typ, type) and issubclass(typ, ShareableItem):
                origin = typ.__base__ if typ is not ShareableItem else ShareableItem
            else:
                origin = typ
                if isinstance(origin, type) and issubclass(origin, BaseItem):
                    origin = origin.schema()  # recurse for nested BaseItems
            # origin = t.get_origin(typ)
            # if origin is None:
            #     origin = getattr(typ, '__origin__', typ)
            # else:
            #     origin = typ

            # if the field itself is a nested BaseItem, recurse to its schema
            # if isinstance(origin, type) and issubclass(origin, BaseItem):
            #     typ = origin.schema()
            # else:
            #     # strip Param[float] → Param, State[int] → State, etc.
            #     typ = origin
            #     print('Typ: ', typ)

            spec_fields[n] = (typ, ... if dflt is inspect._empty else dflt)
        cls.__spec__ = create_model(
            f"{cls.__name__}Spec",
            __base__=BaseSpec,
            model_config=ConfigDict(arbitrary_types_allowed=True),
            **spec_fields,
        )

    # ---------------------------------------------------
    # generic __init__
    # ---------------------------------------------------

    def __init__(self, **kwargs: t.Any):
        self._children = []
        self._init_vars = {}

        for name, _typ, default, is_init in self.__class__.__item_fields__:
            if name in kwargs:
                val = kwargs.pop(name)
            elif default is not inspect._empty:
                val = default
            else:
                raise TypeError(f"Missing required keyword argument: {name!r}")

            if is_init:
                self._init_vars[name] = val
            else:
                setattr(self, name, val)

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {', '.join(kwargs)}")

        # child registration
        for v in vars(self).values():
            if isinstance(v, BaseItem):
                self._children.append(v)

        # optional post‑init
        if hasattr(self, "__post_init__"):
            self.__post_init__(**self._init_vars)
        elif len(self._init_vars) > 0:
            raise RuntimeError(
                'InitVars have been defined but there is no __post_init__ defined.'
            )

    # ---------------------------------------------------
    # schema & spec helpers
    # ---------------------------------------------------

    @classmethod
    def schema(cls) -> type[BaseSpec]:
        return cls.__spec__

    def spec(self) -> BaseSpec:
        data: dict[str, t.Any] = {}
        for n, is_init in self.__class__.__is_initvar__.items():
            if is_init:
                data[n] = self._init_vars[n]
            else:
                v = getattr(self, n)
                if isinstance(v, BaseItem):
                    if getattr(v, "_spec_in_progress", False):
                        raise RuntimeError("Cycle detected in BaseItem hierarchy")
                    v._spec_in_progress = True
                    try:
                        v = v.spec()
                    finally:
                        v._spec_in_progress = False
                elif isinstance(v, BaseModel):
                    v = v.model_dump() 
                data[n] = v
        return self.__class__.__spec__(kind=self.__class__.__qualname__, **data)

    # ---------------------------------------------------
    # parameter & state traversal
    # ---------------------------------------------------

    def _walk(self, recurse: bool):
        yield self
        if recurse:
            for ch in self._children:
                yield from ch._walk(True)

    def parameters(self, *, recurse: bool = True, train_only: bool | None = None):
        seen = set()
        for item in self._walk(recurse):
            for v in vars(item).values():
                if isinstance(v, Param) and id(v) not in seen:
                    if train_only is None or v.training is train_only:
                        seen.add(id(v))
                        yield v

    def state_dict(self, *, recurse: bool = True, train: bool = True, runtime: bool = True) -> dict[str, t.Any]:
        out: dict[str, t.Any] = {}

        def _collect(item: BaseItem, prefix: str):
            for n, v in vars(item).items():
                p = f"{prefix}{n}"
                if isinstance(v, Param):
                    if train:
                        out[p] = v.data
                elif isinstance(v, State):
                    if runtime:
                        out[p] = v.data
                elif isinstance(v, BaseItem) and recurse:
                    _collect(v, p + ".")

        _collect(self, "")
        return out

    def load_state_dict(self, sd: dict[str, t.Any], *, recurse: bool = True, strict: bool = True):
        found = set()

        def _assign(item: BaseItem, key: str, value: t.Any):
            tgt = getattr(item, key, None)
            if isinstance(tgt, (Param, State)):
                print(value)
                tgt.data = value
                return True
            return False

        if recurse:
            # two‑pass: children first bucketed by head of dotted path
            buckets: dict[BaseItem, dict[str, t.Any]] = {}
            for k, v in sd.items():
                if "." in k:
                    head, rest = k.split(".", 1)
                    child = getattr(self, head, None)
                    if isinstance(child, BaseItem):
                        buckets.setdefault(child, {})[rest] = v
                        found.add(k)
                        continue
                if _assign(self, k, v):
                    found.add(k)
            for ch, sub in buckets.items():
                ch.load_state_dict(sub, recurse=True, strict=strict)
                found.update(f"{head}.{k}" for k in sub.keys())
        else:
            for k, v in sd.items():
                if _assign(self, k, v):
                    found.add(k)

        if strict and (missing := set(sd.keys()) - found):
            raise KeyError(f"Keys not found in target object: {sorted(missing)}")
