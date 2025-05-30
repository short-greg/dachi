# ---------------------------------------------------------------------------
#  Dachi core runtime – BuildContext, BaseSpec, BaseItem, Shared, Param, State
# ---------------------------------------------------------------------------
#  Python ≥ 3.8  (uses typing_extensions for back‑compat if <3.12)
# ---------------------------------------------------------------------------

from __future__ import annotations

import inspect
import typing as t
from uuid import uuid4

try:  # Python 3.12+
    from typing import dataclass_transform
except ImportError:  # Python 3.8–3.11
    from typing_extensions import dataclass_transform  # type: ignore

from __future__ import annotations
import typing as t
from typing import Generic, TypeVar, cast
# from .base_item import BaseItem, BaseSpec, BuildContext, register_kind, _KINDS


from pydantic import BaseModel, Field, create_model, ConfigDict

# ---------------------------------------------------------------------------
#  Build‑time context (prevents duplicate Shared instantiation)
# ---------------------------------------------------------------------------

class BuildContext:
    """Caches already‑constructed items by spec.id to guarantee aliasing."""

    def __init__(self) -> None:
        self._reg: dict[str, tuple["BaseItem", dict]] = {}

    def fetch(self, spec: "BaseSpec") -> "BaseItem | None":
        tup = self._reg.get(spec.id)
        if tup and tup[1] != spec.model_dump():
            raise ValueError(
                f"Duplicate id {spec.id!r} maps to different specs"  # noqa: E501
            )
        return tup[0] if tup else None

    def store(self, spec: "BaseSpec", obj: "BaseItem") -> None:
        self._reg.setdefault(spec.id, (obj, spec.model_dump()))

# ---------------------------------------------------------------------------
#  Leaf helpers – Param / State
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  BaseSpec – every runtime item turns into one of these
# ---------------------------------------------------------------------------

class BaseSpec(BaseModel):
    kind : str
    id   : str = Field(default_factory=lambda: str(uuid4()))
    style: t.Literal["structured"] = "structured"
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

# ---------------------------------------------------------------------------
#  Global dispatch table (kind -> class)
# ---------------------------------------------------------------------------

_KINDS: dict[str, type] = {}

def register_kind(cls: type) -> type:  # decorator
    _KINDS[cls.__qualname__] = cls
    return cls

# ---------------------------------------------------------------------------
#  BaseItem core
# ---------------------------------------------------------------------------

from __future__ import annotations
import inspect
import typing as t
from uuid import uuid4

try:                                      # 3.12+
    from typing import dataclass_transform
except ImportError:                       # 3.8-3.11
    from typing_extensions import dataclass_transform

from pydantic import BaseModel, create_model, Field, ConfigDict


# ------------------------------------------------------------
#  Simple Param / State stubs  (unchanged)
# ------------------------------------------------------------
class _LeafBase:
    def __init__(self, *, val): self.val = val
    def __repr__(self): return f"{self.__class__.__name__}(val={self.val!r})"

class Param(_LeafBase):  trainable = True
class State(_LeafBase):  trainable = False


# ------------------------------------------------------------
#  BaseSpec  (unchanged)
# ------------------------------------------------------------
class BaseSpec(BaseModel):
    kind : str
    id   : str = Field(default_factory=lambda: str(uuid4()))
    style: t.Literal["structured"] = "structured"
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


# ------------------------------------------------------------
#  BaseItem  —  no exec, no code-gen
# ------------------------------------------------------------
@dataclass_transform(kw_only_default=True, field_specifiers=(Field,))
class BaseItem:
    # populated in __init_subclass__
    __spec__:        t.ClassVar[t.Type[BaseSpec]]
    __item_fields__: t.ClassVar[list[tuple[str, t.Any, t.Any, bool]]]
    __is_initvar__:  t.ClassVar[dict[str, bool]]

    # per-instance
    _children:   list["BaseItem"]
    _init_vars:  dict[str, t.Any]

    # --------------------------------------------------------
    #  CLASS HOOK
    # --------------------------------------------------------
    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls is BaseItem:
            return

        ann = t.get_type_hints(cls, include_extras=True)
        fields: list[tuple[str, t.Any, t.Any, bool]] = []

        for name, typ in ann.items():
            if name.startswith('_') or t.get_origin(typ) is t.ClassVar:
                continue                                   # ignore ClassVars / privates
            default = getattr(cls, name, inspect._empty)
            is_init = t.get_origin(typ) is t.InitVar
            if is_init:
                typ = t.get_args(typ)[0]
            fields.append((name, typ, default, is_init))

        cls.__item_fields__ = fields
        cls.__is_initvar__  = {n: init for n, *_ , init in fields}

        # build schema once
        spec_fields: dict[str, tuple[t.Any, t.Any]] = {}
        for n, typ, dflt, _ in fields:
            origin = t.get_origin(typ) or typ
            if isinstance(origin, type) and issubclass(origin, BaseItem):
                typ = origin.schema()
            spec_fields[n] = (typ, ... if dflt is inspect._empty else dflt)

        cls.__spec__ = create_model(
            f"{cls.__name__}Spec",
            __base__     = BaseSpec,
            model_config = ConfigDict(arbitrary_types_allowed=True),
            **spec_fields
        )

    # --------------------------------------------------------
    #  GENERIC __init__  (no exec)
    # --------------------------------------------------------
    def __init__(self, **kwargs: t.Any):
        self._children:  list[BaseItem] = []
        self._init_vars: dict[str, t.Any] = {}

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

        # register child items
        for v in vars(self).values():
            if isinstance(v, BaseItem):
                self._children.append(v)

        # optional post-init
        if hasattr(self, "__post_init__"):
            self.__post_init__(**self._init_vars)

    # --------------------------------------------------------
    #  PUBLIC API
    # --------------------------------------------------------
    @classmethod
    def schema(cls) -> t.Type[BaseSpec]:
        return cls.__spec__

    def spec(self) -> BaseSpec:
        data = {}
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
                data[n] = v
        return self.__class__.__spec__(kind=self.__class__.__qualname__, **data)

    # ---------------- parameter / state helpers ----------------
    def _walk(self, recurse: bool):
        yield self
        if recurse:
            for ch in self._children:
                yield from ch._walk(True)

    def parameters(self, *, recurse: bool=True) -> t.Iterator[Param]:
        seen = set()
        for item in self._walk(recurse):
            for v in vars(item).values():
                if isinstance(v, Param) and id(v) not in seen:
                    seen.add(id(v))
                    yield v

    def state_dict(
        self, *, recurse: bool=True, train: bool=True, runtime: bool=True
    ) -> dict[str, t.Any]:
        out = {}
        def _collect(item: BaseItem, prefix: str):
            for n, v in vars(item).items():
                p = f"{prefix}{n}"
                if isinstance(v, Param):
                    if train: out[p] = v.val
                elif isinstance(v, State):
                    if runtime: out[p] = v.val
                elif isinstance(v, BaseItem) and recurse:
                    _collect(v, p + '.')
        _collect(self, '')
        return out

    def load_state_dict(self, sd: dict[str, t.Any], *, recurse: bool=True):
        if recurse:
            buckets: dict[BaseItem, dict[str, t.Any]] = {}
            for k, v in sd.items():
                if '.' in k:
                    head, rest = k.split('.', 1)
                    child = getattr(self, head, None)
                    if isinstance(child, BaseItem):
                        buckets.setdefault(child, {})[rest] = v
                        continue
                self._assign_state(k, v)
            for ch, sub in buckets.items():
                ch.load_state_dict(sub, recurse=True)
        else:
            for k, v in sd.items():
                self._assign_state(k, v)

    def _assign_state(self, key: str, value: t.Any):
        tgt = getattr(self, key)
        if isinstance(tgt, (Param, State)):
            tgt.val = value
        else:
            raise KeyError(f"{key!r} does not reference Param/State")

# SharableItem()
#    Shared()
#    Param()
#    State()
#  1) make them pydantic.BaseModel
#  2) Require to use something easily serializable?
#  3) val can be TypedDict, Pydantic.BaseModel, primitive
#     Must keep simple
#     Best to put limitations on them 
#     use a Generic Pydantic Model for them
#     make suggestions for how to  update

# BaseProcess => 

"""shared.py – alias wrapper that guarantees single instantiation via BuildContext.

Assumes that BaseItem, BaseSpec, BuildContext, register_kind and _KINDS are
importable from the core runtime module (here we reference them relatively).
"""
T = TypeVar("T")

@register_kind
class Shared(BaseItem, Generic[T]):
    """Wraps an arbitrary value `val` so the same `id` in a spec maps to a single
    runtime object.
    """

    val: T  # public attribute, included in BaseItem.__item_fields__

    # BaseItem already provides __init__, post_init, spec(), etc.

    # ------------------------------------------------------------------
    #  Deserialisation helper that respects BuildContext aliasing
    # ------------------------------------------------------------------
    @classmethod
    def from_spec(cls, spec: BaseSpec, ctx: BuildContext) -> "Shared[T]":
        cached = ctx.fetch(spec)
        if cached is not None:
            return cast("Shared[T]", cached)

        inner_val = spec.val  # type: ignore[attr-defined]
        if isinstance(inner_val, BaseSpec):
            # nested BaseItem: delegate to its class loader
            inner_cls = _KINDS[inner_val.kind]
            inner_val = inner_cls.from_spec(inner_val, ctx)  # type: ignore[arg-type]

        obj = cls(val=inner_val)  # type: ignore[arg-type]
        ctx.store(spec, obj)
        return obj

    # Shared is a leaf for state_dict / parameters – nothing extra needed.

    def __repr__(self) -> str:  # pragma: no cover
        return f"Shared(id={id(self):x}, val={self.val!r})"


class State(Shared):

    pass



class Param(Shared):

    training: bool = True

