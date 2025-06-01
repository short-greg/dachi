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

import inspect
from typing import Callable, Any, Dict, Optional, Union, List
from dataclasses import InitVar
import inspect
import typing as t
from uuid import uuid4

try:  # 3.12+
    from typing import dataclass_transform
except ImportError:  # 3.8–3.11
    from typing_extensions import dataclass_transform

from pydantic import BaseModel, Field, ConfigDict, create_model, field_validator

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

    # @field_validator("data")
    # @classmethod
    # def check_data_type(cls, v):
    #     # Extract expected type T from __orig_bases__
    #     expected_type = None
    #     for base in cls.__orig_bases__:
    #         if getattr(base, '__origin__', None) is ShareableItem:
    #             expected_type = get_args(base)[0]
    #             break
    #     if expected_type and not isinstance(v, expected_type):
    #         raise TypeError(f"Expected data of type {expected_type}, got {type(v)}")
    #     return v

    def __hash__(self):
        return id(self) 

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
class BaseModule:
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

    @classmethod
    def __build_schema__(cls) -> None:

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

        for n, typ, dflt, _ in fields:
            if isinstance(typ, type) and issubclass(typ, ShareableItem):
                origin = typ.__base__ if typ is not ShareableItem else ShareableItem
            else:
                origin = typ
                if isinstance(origin, type) and issubclass(origin, BaseModule):
                    origin = origin.schema()  # recurse for nested BaseItems

            spec_fields[n] = (typ, ... if dflt is inspect._empty else dflt)
        cls.__spec__ = create_model(
            f"{cls.__name__}Spec",
            __base__=BaseSpec,
            model_config=ConfigDict(arbitrary_types_allowed=True),
            **spec_fields,
        )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls is BaseModule:
            return
        if not hasattr(cls, "__spec__"):
            cls.__build_schema__()

    # ---------------------------------------------------
    # generic __init__
    # ---------------------------------------------------

    def __init__(self, **kwargs: t.Any):
        self._children = []
        self._init_vars = {}
        self._parameters: dict[str, Param] = {}
        self._states: dict[str, State] = {}
        self._modules: dict[str, BaseModule] = {}

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
            if isinstance(v, BaseModule):
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

    # ---- sub-module traversal ----------------------------------------
    def modules(self, *, recurse: bool = True):
        """Yield **self** first, then all sub-items depth-first."""
        yield self
        if recurse:
            for child in self._modules.values():
                yield from child.modules(recurse=True)

    def named_modules(self, *, recurse: bool = True, prefix: str = ""):
        """Yield ``(dotted_name, module)`` pairs."""
        yield prefix.rstrip("."), self
        if recurse:
            for name, child in self._modules.items():
                child_prefix = f"{prefix}{name}."
                yield from child.named_modules(recurse=True, prefix=child_prefix)

    def named_parameters(
        self, *, recurse: bool = True, train_only: bool | None = None, prefix: str = ""
    ) -> t.Generator[tuple[str, Param]]:
        for name, p in self._parameters.items():
            if train_only is None or p.training is train_only:
                yield f"{prefix}{name}", p
        if recurse:
            for cname, child in self._modules.items():
                child_prefix = f"{prefix}{cname}."
                yield from child.named_parameters(recurse=True, train_only=train_only, prefix=child_prefix)

    def children(self):
        """Immediate child modules (non-recursive)."""
        return self._modules.values()

    def named_states(self, *, recurse: bool = True, prefix: str = ""):
        for name, s in self._states.items():
            yield f"{prefix}{name}", s
        if recurse:
            for cname, child in self._modules.items():
                child_prefix = f"{prefix}{cname}."
                yield from child.named_states(recurse=True, prefix=child_prefix)

    def apply(self, fn, *, filter_type: type | None = None):
        """
        Recursively apply *fn* to every registered object.

        If *filter_type* is given, only objects satisfying
        ``isinstance(obj, filter_type)`` are passed to *fn*.
        """
        targets = [self, *self._parameters.values(), *self._states.values()]
        for obj in targets:
            if filter_type is None or isinstance(obj, filter_type):
                fn(obj)
        for child in self._modules.values():
            child.apply(fn, filter_type=filter_type)
    def eval(self):
        """Alias for ``train(False)``."""
        return self.train(False)

    def train(self, mode: bool = True):
        """Recursively set ``Param.training`` for all parameters."""
        for p in self._parameters.values():
            p.training = mode
        for child in self._modules.values():
            child.train(mode)
        return self               # for chaining

    @property
    def training(self) -> bool:
        # True if ANY param is in training mode; False when all frozen
        return any(p.training for p in self._parameters.values())

    def named_children(self):
        return self._modules.items()

    def spec(self) -> BaseSpec:
        data: dict[str, t.Any] = {}
        for n, is_init in self.__class__.__is_initvar__.items():
            if is_init:
                data[n] = self._init_vars[n]
            else:
                v = getattr(self, n)
                if isinstance(v, BaseModule):
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

    def __setattr__(self, name, value):
        if isinstance(value, Param):
            self.register_parameter(name, value)
        elif isinstance(value, State):
            self.register_state(name, value)
        elif isinstance(value, BaseModule):
            self.register_module(name, value)
        else:
            super().__setattr__(name, value)

    def register_parameter(self, name: str, param: Param):
        self._parameters[name] = param
        super().__setattr__(name, param)

    def register_state(self, name: str, state: State):
        self._states[name] = state
        super().__setattr__(name, state)

    def register_module(self, name: str, module: BaseModule):
        self._modules[name] = module
        super().__setattr__(name, module)

    def parameters(self, *, recurse: bool = True, train_only: bool | None = None, _seen: t.Optional[set[int]] = None):
        if _seen is None:
            _seen = set()

        for param in self._parameters.values():
            if id(param) not in _seen:
                if train_only is None or param.training is train_only:
                    _seen.add(id(param))
                    yield param

        if recurse:
            for child in self._modules.values():
                yield from child.parameters(recurse=True, train_only=train_only, _seen=_seen)

    # def parameters(self, *, recurse: bool = True, train_only: bool | None = None):
    #     seen = set()

    #     # Collect local parameters
    #     for param in self._parameters.values():
    #         if id(param) not in seen:
    #             if train_only is None or param.training is train_only:
    #                 seen.add(id(param))
    #                 yield param

    #     # Recurse into submodules
    #     if recurse:
    #         for child in self._modules.values():
    #             yield from child.parameters(recurse=True, train_only=train_only)

    def state_dict(
        self,
        *,
        recurse: bool = True,
        train: bool = True,
        runtime: bool = True,
    ) -> dict[str, t.Any]:
        out: dict[str, t.Any] = {}

        # Collect Params
        if train:
            for name, param in self._parameters.items():
                out[name] = param.data

        # Collect States
        if runtime:
            for name, state in self._states.items():
                out[name] = state.data

        # Recurse into child BaseItems
        if recurse:
            for name, child in self._modules.items():
                child_sd = child.state_dict(recurse=True, train=train, runtime=runtime)
                for sub_name, value in child_sd.items():
                    out[f"{name}.{sub_name}"] = value

        return out

    def state_keys(
        self,
        *,
        recurse: bool = True,
        train: bool = True,
        runtime: bool = True
    ) -> set[str]:
        """
        Returns a set of dotted keys representing the structure of the state_dict.
        """
        keys = set()

        def _collect(obj: BaseModule, prefix: str):
            for name, v in vars(obj).items():
                path = f"{prefix}{name}"
                if isinstance(v, Param) and train:
                    keys.add(path)
                elif isinstance(v, State) and runtime:
                    keys.add(path)
                elif isinstance(v, BaseModule) and recurse:
                    _collect(v, path + ".")
        _collect(self, "")
        return keys

    def load_state_dict(
        self,
        sd: dict[str, t.Any],
        *,
        recurse: bool = True,
        train: bool = True,
        runtime: bool = True,
        strict: bool = True
    ):
        found = set()

        # Load Params
        if train:
            for name, param in self._parameters.items():
                if name in sd:
                    param.data = sd[name]
                    found.add(name)

        # Load States
        if runtime:
            for name, state in self._states.items():
                if name in sd:
                    state.data = sd[name]
                    found.add(name)

        # Recurse into submodules
        if recurse:
            for name, child in self._modules.items():
                child_sd = {k[len(name)+1:]: v for k, v in sd.items() if k.startswith(f"{name}.")}
                child.load_state_dict(child_sd, recurse=True, train=train, runtime=runtime, strict=False)
                found.update(f"{name}.{k}" for k in child_sd.keys())

        if strict:
            expected_keys = self.state_keys(recurse=recurse, train=train, runtime=runtime)
            passed_keys = set(sd.keys())
            missing_keys = expected_keys - passed_keys
            extra_keys = passed_keys - expected_keys

            if missing_keys:
                raise KeyError(f"Missing keys in load_state_dict: {sorted(missing_keys)}")
            if extra_keys:
                raise KeyError(f"Unexpected keys in load_state_dict: {sorted(extra_keys)}")


class RegistryEntry:
    def __init__(self,
                 obj: Union[type, Callable],
                 obj_type: str,
                 tags: Dict[str, Any],
                 package: str,
                 description: Optional[str] = None):
        self.obj = obj
        self.type = obj_type
        self.tags = tags
        self.package = package
        self.description = description


class Registry:
    def __init__(self):
        self._entries: Dict[str, RegistryEntry] = {}

    def register(self,
                 name: Optional[str] = None,
                 tags: Optional[Dict[str, Any]] = None,
                 description: Optional[str] = None) -> Callable[[Union[type, Callable]], Union[type, Callable]]:
        def decorator(obj: Union[type, Callable]) -> Union[type, Callable]:
            key: str = name or obj.__name__
            obj_type: str = "class" if inspect.isclass(obj) else "function"
            module: str = obj.__module__

            if key in self._entries:
                print(f"Warning: Overwriting existing entry '{key}'")

            self._entries[key] = RegistryEntry(
                obj=obj,
                obj_type=obj_type,
                tags=tags or {},
                package=module,
                description=description
            )
            return obj
        return decorator

    def filter(self,
               obj_type: Optional[str] = None,
               tags: Optional[Dict[str, Any]] = None,
               package: Optional[str] = None) -> Dict[str, RegistryEntry]:
        results: Dict[str, RegistryEntry] = {}
        for k, v in self._entries.items():
            if obj_type and v.type != obj_type:
                continue
            if tags and not all(item in v.tags.items() for item in tags.items()):
                continue
            if package and v.package != package:
                continue
            results[k] = v
        return results

    def __getitem__(self, key: Union[str, List[str]]) -> Union[RegistryEntry, Dict[str, RegistryEntry]]:
        if isinstance(key, list):
            return {k: self._entries[k] for k in key if k in self._entries}
        return self._entries[key]

    def deregister(self, key: str) -> None:
        if key in self._entries:
            del self._entries[key]

    def list_entries(self) -> List[str]:
        return list(self._entries.keys())

    def list_types(self) -> List[str]:
        return list(set(v.type for v in self._entries.values()))

    def list_packages(self) -> List[str]:
        return list(set(v.package for v in self._entries.values()))

    def list_tags(self) -> List[str]:
        tags: set[str] = set()
        for v in self._entries.values():
            tags.update(v.tags.keys())
        return list(tags)


registry = Registry()
