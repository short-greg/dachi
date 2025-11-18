# 1st party
from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Union, Generic, Callable, Any, Dict, Optional, Union, List
from pydantic.fields import FieldInfo
import typing as t
import pydantic
import itertools
from typing import Generic, Union
import inspect
import json
from enum import Enum, auto
from dataclasses import dataclass

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

# try:  # 3.12+
#     from typing import dataclass_transform
# except ImportError:  # 3.8–3.11
#     from typing_extensions import dataclass_transform

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
J = t.TypeVar("J", bound=t.Union[BaseModel, dict, str, int, float, bool])



def to_kind(cls): 
    """Convert a class to its kind."""
    
    return cls.__qualname__


class ShareableItem(pydantic.BaseModel, t.Generic[J]):
    """Serializable leaf object with a ``data`` field."""

    data: J | None
    _callbacks = t.List[Callable[[J | None, J | None]]] = pydantic.PrivateAttr(default_factory=list)

    # def get(self) -> J | None:

    #     return self._data
    
    # def set(self, data: J | None):
    #     expected_type = self._get_expected_type()
    #     if (
    #         expected_type is not None and not isinstance(data, expected_type)
    #     ):
    #         raise TypeError(f"Expected data of type {expected_type}, got {type(data)}")
    #     self._data = data
    #     self.update_data_hook(data)

    def get(self) -> J | None:
        return self.data

    def set(self, value: J | None):
        
        old_val = self.data
        super().__setattr__('data', value)
        self.update_data_hook(old_val, value)
        return value

    def __setattr__(self, name, value):
        
        if name == "data":
            return self.set(value)
        return super().__setattr__(name, value)

    def empty(self) -> bool:

        return self.data is None

    # @property
    # def data(self) -> J | None:
    #     """Get the data value."""
    #     return self._data

    # @data.setter
    # def data(self, value: J):
    #     """Set the data value and trigger update hook."""
    #     self.set(value)
    #     return value

    def update_data_hook(self, old_val: J | None, val: J | None) -> J | None:
        # override for any hooks / logic here for data
        # e.g. log, trigger dirty flag, coerce type
        for callback in self._callbacks:
            callback(old_val, val)

    def __hash__(self):
        return id(self) 
    
    def load(self, data) -> "ShareableItem[J]":
        """
        Rebuild a ShareableItem from a spec or dict.
        """
        if isinstance(self._data, BaseModel):
            # If data is a BaseModel, use its model_validate method
            self._data = self._data.model_validate(data)
        else:
            self._data = data

    def dump(self) -> dict:
        """
        Dump the ShareableItem to a dictionary.
        """
        if isinstance(self.data, BaseModel):
            # If data is a BaseModel, use its model_dump method
            data = self.data.model_dump()
        else:
            data = self.data
        return data

    def schema(self) -> dict:

        return self.model_json_schema()

    # def schema(self) -> dict:
    #     """
    #     Get the JSON schema dict for the ShareableItem's data.

    #     Returns:
    #         JSON schema dictionary
    #     """
    #     if isinstance(self._data, BaseModel):
    #         return type(self._data).model_json_schema()
    #     else:
    #         python_type = type(self._data).__name__
    #         json_type = python_type_to_json_schema_type(python_type)
    #         return {"type": json_type}

    def has_callback(self, callback: Callable[[J | None, J | None], None]) -> bool:
        return callback in self._callbacks

    def register_callback(self, callback: Callable[[J | None, J | None], None]) -> None:
        """Register a callback to be called when the data is updated."""
        self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[J | None, J | None], None]) -> bool:
        """Unregister a previously registered callback. 
        If callback does not exist will return False"""
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False

    # def _get_expected_type(self):
    #     """Resolve expected type from the immediate generic base (Param, State, Shared)."""
    #     cls = self.__class__
    #     for base in getattr(cls, "__orig_bases__", []):
    #         origin = getattr(base, "__origin__", None)
    #         if origin in {Param, Attr, Shared}:
    #             return t.get_args(base)[0]
    #     return None
    
    def __eq__(self, other):
        if isinstance(other, ShareableItem):
            return self.data == other.data
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.data)})"

    def __call__(self, value: J):
        self.data = value
        return self.data

    # Arithmetic dunder methods
    def __add__(self, other):
        result = self.data + (other.data if isinstance(other, ShareableItem) else other)
        self.data = result
        return self

    def __sub__(self, other):
        result = self.data - (other.data if isinstance(other, ShareableItem) else other)
        self.data = result
        return self

    def __mul__(self, other):
        result = self.data * (other.data if isinstance(other, ShareableItem) else other)
        self.data = result
        return self

    def __truediv__(self, other):
        result = self.data / (other.data if isinstance(other, ShareableItem) else other)
        self.data = result
        return self

    def __floordiv__(self, other):
        result = self.data // (other.data if isinstance(other, ShareableItem) else other)
        self.data = result
        return self

    def __mod__(self, other):
        result = self.data % (other.data if isinstance(other, ShareableItem) else other)
        self.data = result
        return self

    def __pow__(self, other):
        result = self.data ** (other.data if isinstance(other, ShareableItem) else other)
        self.data = result
        return self

    # Reverse arithmetic dunder methods
    def __radd__(self, other):
        result = (other.data if isinstance(other, ShareableItem) else other) + self.data
        self.data = result
        return self

    def __rsub__(self, other):
        result = (other.data if isinstance(other, ShareableItem) else other) - self.data
        self.data = result
        return self

    def __rmul__(self, other):
        result = (other.data if isinstance(other, ShareableItem) else other) * self.data
        self.data = result
        return self

    def __rtruediv__(self, other):
        result = (other.data if isinstance(other, ShareableItem) else other) / self.data
        self.data = result
        return self

    def __rfloordiv__(self, other):
        result = (other.data if isinstance(other, ShareableItem) else other) // self.data
        self.data = result
        return self

    def __rmod__(self, other):
        result = (other.data if isinstance(other, ShareableItem) else other) % self.data
        self.data = result
        return self

    def __rpow__(self, other):
        result = (other.data if isinstance(other, ShareableItem) else other) ** self.data
        self.data = result
        return self

    # In-place arithmetic dunder methods
    def __iadd__(self, other):
        self.data = self.data + (other.data if isinstance(other, ShareableItem) else other)
        return self

    def __isub__(self, other):
        self.data = self.data - (other.data if isinstance(other, ShareableItem) else other)
        return self

    def __imul__(self, other):
        self.data = self.data * (other.data if isinstance(other, ShareableItem) else other)
        return self

    def __itruediv__(self, other):
        self.data = self.data / (other.data if isinstance(other, ShareableItem) else other)
        return self

    def __ifloordiv__(self, other):
        self.data = self.data // (other.data if isinstance(other, ShareableItem) else other)
        return self

    def __imod__(self, other):
        self.data = self.data % (other.data if isinstance(other, ShareableItem) else other)
        return self

    def __ipow__(self, other):
        self.data = self.data ** (other.data if isinstance(other, ShareableItem) else other)
        return self

    def __lt__(self, other):
        return self.data < (other.data if isinstance(other, ShareableItem) else other)

    def __le__(self, other):
        return self.data <= (other.data if isinstance(other, ShareableItem) else other)

    def __gt__(self, other):
        return self.data > (other.data if isinstance(other, ShareableItem) else other)

    def __ge__(self, other):
        return self.data >= (other.data if isinstance(other, ShareableItem) else other)


class Param(ShareableItem[J]):
    """Trainable parameter; ``training`` may be toggled to freeze it."""

    fixed: bool = False
    
    def set(self, data):
        if self._fixed:
            raise RuntimeError(
                'Cannot set parameter that is fixed.'
            )
        data = super().set(data)

    def is_fixed(self) -> bool:
        """
        Check if the parameter is fixed.
        """
        return self.fixed
    
    def fix(self):
        """
        Fix the parameter, making it unmodifiable.
        """
        self.fixed = True

    def unfix(self):
        """
        Unfix the parameter, making it modifiable.
        """
        self.fixed = False


class Runtime(ShareableItem[J]):
    """Mutable runtime state (e.g. counters, RNG seeds, rolling averages).

    Example:

    attr = Attr[float](data=0.0)
    """
    pass


class Shared(ShareableItem[J]):
    """Pointer‑like wrapper whose value should *not* enter ``state_dict``.
    
    Example:

    shared = Shared[float](data=0.0)
    """
    pass


class Renderable(ABC):
    """Mixin for classes that implement the render()
    method. Render is used to determine how to represent an
    object as a string to send to thte LLM
    """

    @abstractmethod
    def render(self) -> str:
        """Convert an object to a string representation for 
        an llm

        Returns:
            str: the string representation of the object
        """
        pass


class Trainable(ABC):
    """
    """

    @abstractmethod
    def parameters(self) -> t.Iterator['Param']:
        pass


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


def ParamField(
    *, default=..., frozen=True, default_factory=..., **kwargs
):
    """Create a Field for Param with default value.
    Frozen is set to True by default 
    """
    f = Field(default, frozen=frozen, default_factory=default_factory, **kwargs)
    # mark this field as a "param"
    meta = getattr(f, "json_schema_extra", None) or {}
    meta["is_param"] = True
    f.json_schema_extra = meta
    return f


class SelfInit:

    def __init__(self, fn: t.Callable[['Module'], t.Any]):
        self.fn = fn

    def __call__(self, module: 'Module'):
        return self.fn(module)


def PrivateRuntime(
    default=None, 
    default_factory=None, 
    instance_factory=None
):
    """Create a PrivateAttr for Attr with default value."""
    if instance_factory is not None and default is None and default_factory is None:
        return pydantic.PrivateAttr(
            default=SelfInit(
                instance_factory, Runtime
            )
        )
    if default is None and default_factory is not None:
        return pydantic.PrivateAttr(
            default_factory=lambda: Runtime(data=default_factory())
        )
    return pydantic.PrivateAttr(
        default_factory=lambda: Runtime(data=default)
    )


def PrivateParam(
    default=None, 
    default_factory=None, 
    instance_factory=None
):
    """Create a PrivateAttr for Param with default value."""
    if instance_factory is not None and default is None and default_factory is None:
        return pydantic.PrivateAttr(
            default=SelfInit(
                instance_factory, Param
            )
        )
    if default is None and default_factory is not None:
        return pydantic.PrivateAttr(
            default_factory=lambda: Param(data=default_factory())
        )
    return pydantic.PrivateAttr(
        default_factory=lambda: Param(data=default)
    )


class StateType(Enum):

    MODULE: str = auto()
    ATTR: str = auto()
    PARAM: str = auto()


class Module(pydantic.BaseModel):
    # Pydantic v2 style config
    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        ignored_types=(Param, Runtime),  # do not treat Param/Runtime annotations as fields
    )

    kind: str
    # registry: name -> StateType (PARAM / ATTR / MODULE)
    _registry: t.Dict[str, StateType] = pydantic.PrivateAttr(default_factory=dict)

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if cls is Module:
            return

        # Set constant kind: Literal["ClsName"]
        cls.__annotations__["kind"] = t.Literal[cls.__qualname__]
        if "kind" not in cls.__dict__:
            cls.kind = cls.__qualname__

        #
        # --- Handle ParamFields declared as normal fields ---
        #
        # Detect which fields were declared as ParamField(...)
        param_field_names: list[str] = []
        for name, field in cls.__dict__.items():
            if isinstance(field, pydantic.fields.FieldInfo):
                extra = field.json_schema_extra or {}
                if extra.get("is_param"):
                    param_field_names.append(name)

        # For each ParamField:
        #   1. Remove it from annotations (no longer a pydantic field)
        #   2. Add a PrivateAttr placeholder instead
        for name in param_field_names:
            cls.__annotations__.pop(name, None)
            private = pydantic.PrivateAttr(default=None)
            setattr(cls, name, private)
            cls.__private_attributes__[name] = private

        for name, attr in list(cls.__dict__.items()):
            if not isinstance(attr, FieldInfo):
                continue

            extra = attr.json_schema_extra or {}
            if not extra.get("is_param"):
                continue

            # Original user annotation type, e.g. `float` in `x: float = ParamField(...)`
            orig_type = cls.__annotations__.get(name, t.Any)

            # Annotate as Param[orig_type] so later code can inspect __annotations__[name].__args__[0]
            cls.__annotations__[name] = Param[orig_type]

            # Build a PrivateParam default based on the Field default
            default = attr.default
            if isinstance(default, SelfInit):
                private_attr = PrivateParam(factory=default.fn)
            else:
                private_attr = PrivateParam(default=default)

            # Replace the FieldInfo on the class with the PrivateAttr
            setattr(cls, name, private_attr)

        # Rebuild model so pydantic recognizes changes
        cls.model_rebuild(force=True)

    def model_post_init(self, __context):
        super().model_post_init(__context)

        # 1) Param *fields* (ParamField)
        for name, field in self.model_fields.items():
            extra = getattr(field, "json_schema_extra", None) or {}
            if not extra.get("is_param"):
                continue

            raw_value = getattr(self, name)  # validated user value (T)

            # build Param[T] using the annotation if available
            ann = self.__annotations__.get(name, t.Any)
            try:
                ParamT = Param[ann]          # Param[T]
            except TypeError:
                ParamT = Param               # fallback

            param_obj = ParamT(raw_value)    # Param[T](data) or similar
            setattr(self, name, param_obj)   # now attribute `name` is a Param
            if name in self._registry:
                raise RuntimeError(
                    f"Parameter '{name}' already registered in module '{self.__class__.__name__}'"
                )
            self._registry[name] = StateType.PARAM

        # 2) Private attributes (ignore the registry itself)
        for name in self.__private_attributes__.keys():
            if name == "_registry":
                continue

            value = getattr(self, name)

            # SelfInit → compute from self, then re-read
            if isinstance(value, SelfInit):
                computed = value.fn(self)
                setattr(self, name, computed)
                value = computed

            if isinstance(value, Param):
                if name in self._registry:
                    raise RuntimeError(
                        f"Parameter '{name}' already registered in module '{self.__class__.__name__}'"
                    )
                self._registry[name] = StateType.PARAM

            elif isinstance(value, Runtime):
                if name in self._registry:
                    raise RuntimeError(
                        f"Runtime attribute '{name}' already registered in module '{self.__class__.__name__}'"
                    )
                self._registry[name] = StateType.ATTR

            elif isinstance(value, Module):
                if name in self._registry:
                    raise RuntimeError(
                        f"Module '{name}' already registered in module '{self.__class__.__name__}'"
                    )
                self._registry[name] = StateType.MODULE
        
    def parameters(
        self,
        *,
        recurse: bool = True,
        _seen: t.Optional[set[int]] = None,
        with_annotations: bool = False,
    ) -> t.Iterator[Param | tuple[Param, t.Any]]:
        if _seen is None:
            _seen = set()

        # local params
        for name, state_type in self._registry.items():
            if state_type is not StateType.PARAM:
                continue

            param = getattr(self, name)
            if not isinstance(param, Param):
                continue

            if id(param) in _seen:
                continue
            _seen.add(id(param))

            if with_annotations:
                ann = self.__annotations__.get(name, t.Any)
                yield (param, ann)
            else:
                yield param

        # recurse into child modules
        if recurse:
            for name, state_type in self._registry.items():
                if state_type is not StateType.MODULE:
                    continue
                child = getattr(self, name)
                if isinstance(child, Module):
                    yield from child.parameters(
                        recurse=True, _seen=_seen, with_annotations=with_annotations
                    )

    def modules(
        self,
        *,
        recurse: bool = True,
        f: t.Callable[['Module'], bool] | None = None,
    ):
        if f is None or f(self):
            yield self
        if recurse:
            for name, state_type in self._registry.items():
                if state_type is not StateType.MODULE:
                    continue
                child = getattr(self, name)
                if isinstance(child, Module):
                    yield from child.modules(recurse=True, f=f)

    def named_modules(
        self,
        *,
        recurse: bool = True,
        prefix: str = "",
        f: t.Callable[['Module'], bool] | None = None,
    ):
        if f is None or f(self):
            yield prefix.rstrip("."), self
        if recurse:
            for name, state_type in self._registry.items():
                if state_type is not StateType.MODULE:
                    continue
                child = getattr(self, name)
                if isinstance(child, Module):
                    child_prefix = f"{prefix}{name}."
                    yield from child.named_modules(
                        recurse=True, prefix=child_prefix, f=f
                    )

    def named_parameters(
        self,
        *,
        recurse: bool = True,
        prefix: str = "",
    ) -> t.Generator[tuple[str, Param], None, None]:
        for name, state_type in self._registry.items():
            if state_type is not StateType.PARAM:
                continue
            param = getattr(self, name)
            if isinstance(param, Param):
                yield f"{prefix}{name}", param

        if recurse:
            for name, state_type in self._registry.items():
                if state_type is not StateType.MODULE:
                    continue
                child = getattr(self, name)
                if isinstance(child, Module):
                    child_prefix = f"{prefix}{name}."
                    yield from child.named_parameters(
                        recurse=True, prefix=child_prefix
                    )

    def named_states(
        self,
        *,
        recurse: bool = True,
        prefix: str = "",
    ):
        """Yield all state names and their Runtime objects."""
        for name, state_type in self._registry.items():
            if state_type is not StateType.ATTR:
                continue
            state = getattr(self, name)
            if isinstance(state, Runtime):
                yield f"{prefix}{name}", state

        if recurse:
            for name, state_type in self._registry.items():
                if state_type is not StateType.MODULE:
                    continue
                child = getattr(self, name)
                if isinstance(child, Module):
                    child_prefix = f"{prefix}{name}."
                    yield from child.named_states(recurse=True, prefix=child_prefix)

    def children(self):
        """Immediate child modules (non-recursive)."""
        return [
            getattr(self, name)
            for name, state_type in self._registry.items()
            if state_type is StateType.MODULE
        ]

    def named_children(self):
        """Immediate child modules (name, module) pairs."""
        for name, state_type in self._registry.items():
            if state_type is not StateType.MODULE:
                continue
            child = getattr(self, name)
            if isinstance(child, Module):
                yield name, child

    def apply(
        self,
        fn: t.Callable[[t.Any], None],
        *,
        include: t.Callable[[t.Any], bool] | t.Type | None = None,
    ):
        """
        Recursively apply *fn* to self and all registered objects.
        """
        targets: list[t.Any] = [self]
        for name in self._registry:
            targets.append(getattr(self, name))

        for obj in targets:
            if include is None:
                fn(obj)
            elif isinstance(include, type) and isinstance(obj, include):
                fn(obj)
            elif not isinstance(include, type) and include(obj):
                fn(obj)

        for name, state_type in self._registry.items():
            if state_type is not StateType.MODULE:
                continue
            child = getattr(self, name)
            if isinstance(child, Module):
                child.apply(fn, include=include)

    def train(self, mode: bool = True):
        """Recursively set Param.training for all parameters."""
        self.training = mode
        for name, state_type in self._registry.items():
            if state_type is not StateType.MODULE:
                continue
            child = getattr(self, name)
            if isinstance(child, Module):
                child.train(mode)
        return self

    def eval(self):
        """Alias for ``train(False)``."""
        return self.train(False)

    def named_children(self):
        """
        Yield all child module names and their corresponding modules.
        """
        return self._modules.items()

    # TODO: figure out how to deal with Shareables
    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        # Avoid registering internal / pydantic internals
        if name.startswith("_"):
            return value

        if isinstance(value, Param):
            self._registry[name] = StateType.PARAM
        elif isinstance(value, Runtime):
            self._registry[name] = StateType.ATTR
        elif isinstance(value, Module):
            self._registry[name] = StateType.MODULE

        return value

    def state_dict(
        self,
        *,
        recurse: bool = True,
        train: bool = True,
        runtime: bool = True,
    ) -> dict[str, t.Any]:
        out: dict[str, t.Any] = {}

        # local
        for name, state_type in self._registry.items():
            obj = getattr(self, name)
            if state_type is StateType.PARAM and train and isinstance(obj, Param):
                out[name] = obj.data
            elif state_type is StateType.ATTR and runtime and isinstance(obj, Runtime):
                out[name] = obj.data

        # recurse into modules
        if recurse:
            for name, state_type in self._registry.items():
                if state_type is not StateType.MODULE:
                    continue
                child = getattr(self, name)
                if not isinstance(child, Module):
                    continue
                child_sd = child.state_dict(
                    recurse=True, train=train, runtime=runtime
                )
                for sub_name, value in child_sd.items():
                    out[f"{name}.{sub_name}"] = value

        return out

    def state_keys(
        self,
        *,
        recurse: bool = True,
        train: bool = True,
        runtime: bool = True,
    ) -> set[str]:
        keys: set[str] = set()

        for name, state_type in self._registry.items():
            if state_type is StateType.PARAM and train:
                keys.add(name)
            elif state_type is StateType.ATTR and runtime:
                keys.add(name)

        if recurse:
            for name, state_type in self._registry.items():
                if state_type is not StateType.MODULE:
                    continue
                child = getattr(self, name)
                if not isinstance(child, Module):
                    continue
                for sub_name in child.state_keys(
                    recurse=True, train=train, runtime=runtime
                ):
                    keys.add(f"{name}.{sub_name}")

        return keys

    def load_state_dict(
        self,
        sd: dict[str, t.Any],
        *,
        recurse: bool = True,
        train: bool = True,
        runtime: bool = True,
        strict: bool = True,
    ):
        if not isinstance(sd, dict):
            raise TypeError(f"StateDict must be of type dict, not {type(sd)}")

        found: set[str] = set()

        # local
        for name, state_type in self._registry.items():
            if name not in sd:
                continue
            obj = getattr(self, name)
            if state_type is StateType.PARAM and train and isinstance(obj, Param):
                obj.data = sd[name]
                found.add(name)
            elif state_type is StateType.ATTR and runtime and isinstance(obj, Runtime):
                obj.data = sd[name]
                found.add(name)

        # recurse
        if recurse:
            for name, state_type in self._registry.items():
                if state_type is not StateType.MODULE:
                    continue
                child = getattr(self, name)
                if not isinstance(child, Module):
                    continue
                prefix = f"{name}."
                child_sd = {
                    k[len(prefix):]: v
                    for k, v in sd.items()
                    if k.startswith(prefix)
                }
                child.load_state_dict(
                    child_sd,
                    recurse=True,
                    train=train,
                    runtime=runtime,
                    strict=False,
                )
                found.update(k for k in sd.keys() if k.startswith(prefix))

        if strict:
            expected_keys = self.state_keys(
                recurse=recurse, train=train, runtime=runtime
            )
            passed_keys = set(sd.keys())
            missing = expected_keys - passed_keys
            extra = passed_keys - expected_keys
            if missing:
                raise KeyError(f"Missing keys in load_state_dict: {sorted(missing)}")
            if extra:
                raise KeyError(f"Unexpected keys in load_state_dict: {sorted(extra)}")

    # def children(self):
    #     """Immediate child modules (non-recursive)."""
    #     return [child for (child, state_type) in self._registry.values() if state_type == StateType.MODULE]
    # def eval_args(self):
    #     """Alias for ``train(False)``."""
    #     return self.train(False)

    # def train(self, mode: bool = True):
    #     """Recursively set ``Param.training`` for all parameters."""
    #     self.training = mode
    #     for child in self._modules.values():
    #         child.train(mode)
    #     return self

    # def __setattr__(self, name, value):
        
    #     # TODO: confirm that name does 
    #     # not exist in more than one of these 
    #     # dictionaries
    #     # this currently has some issues
    #     # have to debug
    #     super().__setattr__(name, value)
    #     if isinstance(value, Param):
    #         if name in self.__annotations__:
    #             value = Param[self.__annotations__[name]](data=value)
    #         self._registry[name] = value, StateType.PARAM
    #     elif isinstance(value, Runtime):
    #         if name in self.__annotations__:
    #             value = Runtime[self.__annotations__[name]](data=value)
    #         value = Runtime[self.__annotations__[name]](data=value)
    #         self._registry[name] = value, StateType.ATTR
    #     elif isinstance(value, Module):
    #         self._registry[name] = value, StateType.MODULE

        # if name in self._states:
        #     self._states[name] = 
        #     self._parameters[name] = value
        # elif name in self._states:
        #     self._states[name] = value
        # elif name in self._modules:
        #     self._modules[name] = value
        # return value

    # def __getattribute__(self, name):
    #     val = super().__getattribute__(name)
    #     return val
    
    # def register_parameter(self, name: str, param: Param):
    #     """Register a parameter with the given name."""
    #     self._parameters[name] = param
    #     super().__setattr__(name, param)

    # def register_state(self, name: str, state: Runtime):
    #     """Register a state with the given name."""
    #     self._registry[name] = state
    #     super().__setattr__(name, state)

    # def register_module(self, name: str, module: 'Module'):
    #     """Register a submodule with the given name."""
    #     self._modules[name] = module
    #     super().__setattr__(name, module)



T = t.TypeVar("T")


class RegistryEntry(t.Generic[T]):
    def __init__(self,
                 obj: T,
                 obj_type: str,
                 tags: Dict[str, Any],
                 package: str,
                 description: Optional[str] = None):
        self.obj = obj
        self.type = obj_type
        self.tags = tags
        self.package = package
        self.description = description


V = t.TypeVar("V", bound=Module)


class Checkpoint(pydantic.BaseModel, t.Generic[V]):
    """Checkpoint for BaseModle objects, containing spec and state_dict."""
    
    spec: t.Dict[str, t.Any] = Field(
        description="The specification for the module."
    )
    state: t.Dict[str, Dict[t.Any]] = Field(
        description="The state dict for the module"
    )

    def save(self, path: str):
        """Save the checkpoint to a file."""
        with open(path, 'w') as f:
            data = {
                "spec": self.spec,
                "state": self.state

            }
            f.write(json.dumps(data, indent=2))
            # f.write(self.model_dump_json(indent=2))
    
    @classmethod
    def load(cls, path: str) -> "Checkpoint":
        """Load a checkpoint from a file."""
        with open(path, 'r') as f:
            data = f.read()
        data = json.loads(data)

        load_cls = mod_registry[data['spec']['kind']]

        spec = load_cls.obj.__spec__.model_validate(data['spec'])
        state = data['state']
        return cls(spec=spec, state=state)

    # TODO: Update the following methods

    # @classmethod  
    # def load_module(cls, path: str, ctx: Optional[dict] = None) -> V:
    #     """Reconstruct the BaseModule from the checkpoint."""
    #     if ctx is None:
    #         ctx = {}
    #     obj = cls.load(path)
    #     module_cls = obj.spec.load_cls()
    #     module = module_cls.from_spec(obj.spec, ctx=ctx)
    #     module.load_state_dict(obj.state_dict)
    #     return module
    
    # @classmethod
    # def save_module(self, module: Module, path: str):
    #     """Save the BaseModule as a checkpoint."""
    #     spec = module.spec(to_dict=False)
    #     state_dict = module.state_dict(
    #         recurse=True, train=True, runtime=True
    #     )
    #     checkpoint = Checkpoint(
    #         spec=spec,
    #         state_dict=state_dict
    #     )
    #     checkpoint.save(path)


# ============================================================================
# Module Field Descriptors
# ============================================================================

class Registry(t.Generic[T]):
    """Registry for BaseModule classes and functions.
    Allows registration, filtering, and retrieval of objects
    by name, type, tags, and package.
    """
    def __init__(self):
        self._entries: Dict[str, RegistryEntry[T]] = {}

    def register(
        self,
        name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None
    ) -> Callable[[Union[type, Callable]], Union[type, Callable]]:
        """
        Register a Module in the registry.

        Args:
            name: The name of the module.
            tags: A dictionary of tags associated with the module.
            description: A description of the module.
        """
        def decorator(obj: Union[type, Callable]) -> Union[type, Callable]:

            key: str = name or to_kind(obj)
            obj_type: str = "class" if inspect.isclass(obj) else "function"
            module: str = obj.__module__

            if key in self._entries:
                print(f"Warning: Overwriting existing entry '{key}'")

            self._entries[key] = RegistryEntry[T](
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
               package: Optional[str] = None) -> Dict[str, RegistryEntry[T]]:
        """
        Filter the registry entries based on the given criteria.
        Args:
            obj_type: The type of the object to filter by.
            tags: A dictionary of tags to filter by.
            package: The package to filter by.

        Returns:
            A dictionary of matching registry entries.
        """
        results: Dict[str, RegistryEntry[T]] = {}
        for k, v in self._entries.items():
            if obj_type and v.type != obj_type:
                continue
            if tags and not all(item in v.tags.items() for item in tags.items()):
                continue
            if package and v.package != package:
                continue
            results[k] = v
        return results

    def __getitem__(self, key: Union[str, List[str]]) -> Union[RegistryEntry[T], Dict[str, RegistryEntry[T]]]:
        """Retrieve a single entry by key or a list of entries by keys."""
        try: 
            if isinstance(key, list):
                return {k: self._entries[k] for k in key if k in self._entries}
            return self._entries[key]
        except KeyError:
            raise KeyError(f"Registry entry '{key}' not found. Available entries: {list(self._entries.keys())}")

    def deregister(self, key: str) -> None:
        """Remove an entry from the registry by key."""
        if key in self._entries:
            del self._entries[key]

    def list_entries(self) -> List[str]:
        """
        List all registered entries in the registry.
        """
        return list(self._entries.keys())

    def list_types(self) -> List[str]:
        """
        List all unique types of registered entries.
        """
        return list(set(v.type for v in self._entries.values()))

    def list_packages(self) -> List[str]:
        """
        List all unique packages of registered entries.
        """
        return list(set(v.package for v in self._entries.values()))

    def list_tags(self) -> List[str]:
        """
        List all unique tags of registered entries.
        """
        tags: set[str] = set()
        for v in self._entries.values():
            tags.update(v.tags.keys())
        return list(tags)
    
    def __call__(self,
                 name: Optional[str] = None,
                 tags: Optional[Dict[str, Any]] = None,
                 description: Optional[str] = None) -> Callable[[Union[type, Callable]], Union[type, Callable]]:
        """
        Create a decorator to register a module.
        Args:
            name: The name of the module.
            tags: A dictionary of tags associated with the module.
            description: A description of the module.

        Returns:
            A decorator that registers the module.
        """
        return self.register(
            name=name,
            tags=tags,
            description=description
        )

mod_registry = Registry[Module]()



P = t.TypeVar("P", bound=Param)

class ParamSet(pydantic.BaseModel, t.Generic[J]):
    """ParamSet is a 
    """

    params: t.Tuple[P] = Field(default_factory=tuple)

    @classmethod
    def build(cls, module: Module) -> "ParamSet":
        """Build a ParamSet from a BaseModule, collecting all parameters."""
        params, annotations = list(
            module.parameters(recurse=True, with_annotations=True)
        )
        return cls[annotations](params=params)

    def update(self, param_set: Dict):
        """Update the parameters from a dictionary.

        Args:
            param_set: Dictionary with param_0, param_1, etc. keys
            flat: If True, expects flat dict like {"param_0": "value"}.
                  If False (default), expects schema-compliant structure.
        """
        updated = ParamSet.model_validate(
            param_set
        )
        for old_param, new_param in zip(self, updated):
            old_param.set(new_param.data)


class AdaptModule(
    Module, 
    Generic[V]
):
    """A *module‑as‑parameter* wrapper.

    • Appears as a **single** :class:`~dachi.core.Param` to any optimiser
      (``self.adapted_param``).
    • Holds a live ``self.adapted`` sub‑module that is rebuilt automatically
      whenever the underlying spec changes.
    • Optionally *restricts* which sub‑module classes are legal, both in the
      JSON‑schema exposed to an LLM and at **runtime**.

    Extra features vs the original implementation:
    ------------------------------------------------
    * Runtime whitelist enforcement (``allowed`` kw‑arg)
    * Fixed state‑dict duplication bug
    * Optional gradient isolation of inner parameters (``train_submods``)
    * Hook ``on_swap`` fired after every rebuild for observers / logging
    * ``schema(mapping=None)`` mirrors BT / DAG helper so callers can patch
      the JSON‑Schema in one line.
    """
    # set of *kind* strings allowed
    # allowed:   InitVar[t.Dict[str | t.Type, t.Type[BaseSpec] | t.Type[BaseModule] | t.Dict] | None]  = None 

    # fixed: bool = False
    # train_submods: bool = True   # expose inner numeric params?
    _adapted: V | None = PrivateParam(
        None
    )
    _train_submods: bool = pydantic.PrivateAttr(default=True)
    _fixed: bool = pydantic.PrivateAttr(default=False)

    @pydantic.field_validator('adapted', mode='before')
    def validate_adapted(self, val: J | None | Param[J | None]) -> t.Set[str]:
        """Validate the allowed set."""

        if isinstance(val, Param):
            return val
        
        return Param[J](data=val)
    
    def fix(self):
        """Collapse to spec‑blob so only *adapted_param* remains trainable."""
        self._fixed = True

    def unfix(self, *, ctx: dict | None = None):
        self._fixed = False

    def parameters(self, *, recurse=True, _seen=None, with_annotation: bool=False):  # noqa: D401
        if _seen is None:
            _seen = set()
        if id(self) in _seen:
            return
        _seen.add(id(self))

        # always expose the *spec* parameter itself unless frozen
        if not self._fixed:
            if with_annotation:
                yield (self._adapted, V)
            else:
                yield self._adapted

        # inner numeric params – optional
        if recurse and self._train_submods and not self.fixed:
            yield from self._adapted.parameters(recurse=True, _seen=_seen, with_annotations=with_annotation)

    def render(self) -> str:  # for LLM debugging
        return f"AdaptModule(adapted={self._adapted.__class__.__name__}, fixed={self.fixed})"

    @classmethod
    def build(cls, 
        adapted: V | None = None,
        *,
        train_submods: bool = True,
        fixed: bool = False,
        **kwargs
    ) -> "AdaptModule[V]":
        """Build an AdaptModule wrapping *adapted*."""
        adapt_mod = cls(
            **kwargs
        )
        adapt_mod._adapted.set(adapted)
        adapt_mod._train_submods = train_submods
        adapt_mod._fixed = fixed
        return adapt_mod

    @property
    def adapted(self) -> Module:
        """Get the adapted module
        """
        return self._adapted

    @adapted.setter
    def adapted(self, val: V):
        if self._fixed:
            raise RuntimeError("Cannot update adapted on a frozen AdaptModule")
        self._adapted.set(val)

    def state_dict(self) -> dict:
        """Get the state dict, including adapted sub-module."""
        out = super().state_dict()
        if self._adapted is not None:
            adapted_sd = self._adapted.state_dict()
            for k, v in adapted_sd.items():
                out[f"adapted.{k}"] = v
        return out

    def load_state_dict(self, sd, *, recurse = True, train = True, runtime = True, strict = True):
        super().load_state_dict(sd, recurse=recurse, train=train, runtime=runtime, strict=strict)
        # Load adapted sub-module state dict
        if self._adapted is None:
            return
        adapted_sd = {k[len("adapted.") :]: v for k, v in sd.items() if k.startswith("adapted.")}
        self._adapted.load_state_dict(adapted_sd, recurse=recurse, train=train, runtime=runtime, strict=strict)

    # @allowed.setter
    # def allowed(self, allowed: t.List):
    #     """
    #     Set the allowed sub-module classes.
    #     """
    #     if allowed is None:
    #         self._allowed = None
    #         return
    #     allowed_set = set()
        
    #     for item in sorted(set(allowed), key=str):
    #         if inspect.isclass(item) and issubclass(item, BaseModule):
    #             allowed_set.add(to_kind(item))                      # fully-qualified
    #         elif isinstance(item, str):
    #             obj = resolve_name(item, namespace={**globals(), **locals()}, search_sys_modules=True)
    #             if not (inspect.isclass(obj) and issubclass(obj, BaseModule)):
    #                 raise TypeError(
    #                     f"'{item}' is not a BaseModule subclass"
    #                 )
    #             allowed_set.add(to_kind(obj))
    #         else:
    #             raise TypeError(f"Invalid entry {item!r}")
    #     self._allowed = list(allowed_set)

    # def restricted_schema(
    #     self, *, _profile = "shared", _seen = None, **kwargs
    # ) -> dict:
    #     # TODO: Implement

    #     if isinstance(self._adapted, RestrictedSchemaMixin):
    #         return self._adapted.restricted_schema(
    #             _profile=_profile,
    #             _seen=_seen,
    #             **kwargs
    #         )
    #     return self._adapted.schema_model()

    # @classmethod
    # def restricted_schema(
    #     cls,
    #     mapping: Mapping[type[BaseModule], Iterable[type[BaseModule]]] | None = None,
    # ) -> dict:
    #     if mapping is None:
    #         return super().schema()
    #     # canonicalise order so schema output is deterministic regardless of
    #     # caller list ordering – fixes equality test failures.
    #     canonical: dict[type[BaseModule], tuple[type[BaseModule], ...]] = {}
    #     for ph, allowed in mapping.items():
    #         # remove dups then sort by class name for a stable order
    #         unique_sorted = tuple(sorted(set(allowed), key=lambda c: c.__name__))
    #         canonical[ph] = unique_sorted
    #     return cls._restricted_schema(canonical)
    #     # self, mapping: t.Optional[dict[type[BaseModule], t.Iterable[type[BaseModule]]]] = None):
    #     # if mapping is None:
    #     #     return super().schema()
    #     # return self._restricted_schema(mapping)
    #     # if mapping is None:
    #     #     return super().schema()
    #     # return cls._restricted_schema(mapping)

    # def update_adapted(self, new_spec: BaseSpec):
    #     """Callback fired when *adapted_param* changes."""

    #     if self.fixed:
    #         raise RuntimeError("Cannot update adapted on a frozen AdaptModule")
        
    #     sub_cls = mod_registry[new_spec.kind].obj
    #     self._adapted = sub_cls.from_spec(new_spec, ctx={})

    # @mod.setter
    # def mod(self, val: V):
    #     self._adapted = val
    #     # 1) Create a Param that holds the *spec* of the sub‑module
    #     if val is not None:
    #         self._adapted_param.unregister_callback(self.update_adapted)
    #         self._adapted_param.set(data=val.spec())
    #         self._adapted_param.register_callback(self.update_adapted)
    #     else:
    #         self._adapted_param.unregister_callback(self.update_adapted)
    #         self._adapted_param.set(data=None)
    #         self._adapted_param.register_callback(self.update_adapted)

    # @property
    # def adapted_param(self) -> Param:
    #     return self._adapted_param


# P = t.TypeVar("P", bound=Param)

# class ParamSet(pydantic.BaseModel, t.Generic[J]):
#     """ParamSet is a 
#     """

#     params: t.Tuple[P] = Field(default_factory=tuple)

#     @classmethod
#     def build(cls, module: Module) -> "ParamSet":
#         """Build a ParamSet from a BaseModule, collecting all parameters."""
#         params, annotations = list(
#             module.parameters(recurse=True, with_annotations=True)
#         )
#         return cls[annotations](params=params)

#     def update(self, param_set: Dict):
#         """Update the parameters from a dictionary.

#         Args:
#             param_set: Dictionary with param_0, param_1, etc. keys
#             flat: If True, expects flat dict like {"param_0": "value"}.
#                   If False (default), expects schema-compliant structure.
#         """
#         updated = ParamSet.model_validate(
#             param_set
#         )
#         for old_param, new_param in zip(self, updated):
#             old_param.set(new_param.data)
        #    old_param.load(new_param)
        
        # for i, param in enumerate(self.params):
        #     key = f"param_{i}"
        #     if key in param_set:
        #         value = param_set[key]
        #         if not flat and isinstance(param._data, BaseModel):
        #             # For BaseModel data, validate the dict using Pydantic
        #             param.data = type(param._data).model_validate(value)
        #         else:
        #             # For primitives or flat format, direct assignment
        #             param.data = value

    def __iter__(self) -> t.Iterator[P]:
        yield from self.params

    def schema(self) -> dict:
        return self.model_json_schema()

    # def schema(self) -> dict:
    #     """
    #     Return the JSON schema for all parameters in the set.
    #     """
    #     properties = {}
    #     for i, param in enumerate(self.params):
    #         properties[f"param_{i}"] = param.schema()

    #     return {
    #         "type": "object",
    #         "properties": properties,
    #         "required": list(properties.keys()),
    #         "additionalProperties": False
    #     }

    # def to_dict(self) -> dict:
    #     """
    #     Dump all parameters' data to a dictionary.
    #     """
    #     return {
    #         f"param_{i}": param.dump() 
    #         for i, param in enumerate(self.params)
    #     }

    # def state_dict(
    #     self, *, 
    #     recurse: bool = True, 
    #     train: bool = True, 
    #     runtime: bool = True):
    #     sd = {}
    #     # sd = super().state_dict()
    #     # spec Param
    #     # nested params / attrs
    #     if recurse:
    #         for k, v in self._adapted.state_dict(recurse=True, train=train, runtime=runtime).items():
    #             sd[f"_adapted.{k}"] = v
    #     sd["_adapted_param"] = self.adapted_param.dump()
    #     print(list(sd.keys()))
    #     return sd

    # def load_state_dict(self, sd: dict[str, t.Any], *, recurse: bool = True, train: bool = True, runtime: bool = True, strict: bool = True):
    #     # 1) restore spec first (this rebuilds `adapted` via callback)
    #     # super().load_state_dict(
    #     #     sd, recurse=recurse, train=train,
    #     #     runtime=runtime, strict=strict
    #     # )
    #     if "_adapted_param" in sd:
    #         # pass
    #         cur_cls = mod_registry[sd['_adapted_param']['kind']].obj
    #         spec = cur_cls.schema().model_validate(sd['_adapted_param'])
    #         print(spec)
    #         self._adapted_param.data = spec
    #         # self.adapted_param.load(sd["adapted_param"])
    #     # 2) pass nested keys to adapted module
    #     nested = {k[len("_adapted."):]: v for k, v in sd.items() if k.startswith("_adapted.")}
    #     if nested:
    #         self.adapted.load_state_dict(nested, recurse=True, train=train, runtime=runtime, strict=strict)
    #     # strict checking
    #     if strict:
    #         expected = self.state_keys(recurse=True, train=train, runtime=runtime)
    #         missing = expected - sd.keys()
    #         extra = sd.keys() - expected
    #         if missing:
    #             raise KeyError(f"Missing keys in load_state_dict: {sorted(missing)}")
    #         if extra:
    #             raise KeyError(f"Unexpected keys in load_state_dict: {sorted(extra)}")


# class RestrictedSchemaMixin:
#     """
#     Provide `_restricted_schema(mapping)` where **mapping** is
#     {placeholder_cls: iterable_of_allowed_module_classes}.
#     Patches the JSON-Schema so every "$ref" to each placeholder's *spec*
#     is replaced by a `oneOf` union of the allowed spec classes.

#     Purely cosmetic – runtime validation is unchanged.
#     """
#     @classmethod
#     def _restricted_schema(
#         cls,
#         mapping: t.Mapping[
#             type["BaseModule"],              # placeholder  (e.g. Task)
#             Iterable[type["BaseModule"]]     # allowed mods (e.g. Action1…)
#         ],
#     ) -> dict:

#         # normalise & freeze for cache key
#         norm = tuple((ph, tuple(allowed)) for ph, allowed in mapping.items())
#         return cls.__rs_cache(norm)

#     @classmethod
#     @lru_cache
#     def __rs_cache(
#         cls,
#         norm: tuple[tuple[type["BaseModule"], tuple[type["BaseModule"], ...]], ...],
#     ) -> dict:

#         # 0) Build patch-tables for every placeholder
#         union_schemas   = {}    # placeholder_spec_name → dict(oneOf=…)
#         placeholder_refs = {}   # placeholder_spec_name → full "$ref" str

#         for placeholder_cls, allowed in norm:
#             allowed_specs = [m.schema() for m in allowed]
#             union         = Union[tuple(allowed_specs)]
#             union_schema  = TypeAdapter(union).json_schema()

#             # union_schema *is* the JSON of oneOf already
#             union_schemas[placeholder_cls.schema().__name__] = union_schema
#             placeholder_refs[placeholder_cls.schema().__name__] = (
#                 f"#/$defs/{placeholder_cls.schema().__name__}"
#             )

#         # 1) For convenience, make a *root* union of all first-level allowed specs
#         #    (not strictly required but matches earlier behaviour)
#         top_specs = [s for _, allowed in norm for s in allowed]
#         root_schema = TypeAdapter(Union[tuple(m.schema() for m in top_specs)]
#                                   ).json_schema()

#         # 2) Walk & patch
#         patched = copy.deepcopy(root_schema)

#         def _walk(obj):
#             if isinstance(obj, dict):
#                 ref = obj.get("$ref")
#                 if ref:
#                     # check each placeholder
#                     for spec_name, target_ref in placeholder_refs.items():
#                         if ref == target_ref:
#                             obj.clear()
#                             obj.update(union_schemas[spec_name])
#                             break
#                 else:
#                     for v in obj.values():
#                         _walk(v)
#             elif isinstance(obj, list):
#                 for v in obj:
#                     _walk(v)

#         _walk(patched)
#         return patched


# def get_class_annotations(cls: type) -> dict[str, type]:
#     """Safely get annotations with fallback to __annotations__"""
#     try:
#         hints = t.get_type_hints(cls)
#     except Exception as e:
#         # Log or handle if needed
#         hints = {}

#     raw = getattr(cls, '__annotations__', {})
#     for k, v in raw.items():
#         if k not in hints:
#             hints[k] = v  # fallback type, maybe str or ForwardRef
#     return hints


# @dataclass_transform(kw_only_default=True, field_specifiers=(Field,))
# class BaseModule:
#     """Dataclass‑like runtime object without exec‑generated ``__init__``.
    
#     A Pydantic BaseModel will be created for BaseModules
#     automatically so all fields specified in the
#     class header must be serializable by Pydantic to
#     create the Spec.

#     Use InitVar to indicate that a variable should be initialized in post_init. Other fields will automatically be included as member variables.

#     Example:
#     class MyModule(BaseModule):
    
#         # The attributes defined here must
#         # be serializable by Pydantic to create
#         # MyModuleSpec. Don't include Param, Attr, Shared
#         # in the header; instead, define them in __post_init__.

#         val: int = 1
#         name: InitVar[str]

#         def __post_init__(self, name: str):
#             self.name = name
#             self.weight = Param(data=0.5)
#             self.cur_index = Attr(data=0)
#     m = MyModule(name="test")

#     """

#     # populated by __init_subclass__
#     __spec__: t.ClassVar[type[BaseSpec]]
#     __spec_hooks__: t.ClassVar[t.List[str]] = []
#     __item_fields__: t.ClassVar[list[tuple[str, t.Any, t.Any, bool]]]
#     __is_initvar__: t.ClassVar[dict[str, bool]]
#     training: bool = True  # True if any Module is in training mode; False when not

#     def __post_init__(self):
#         pass
    
#     @classmethod
#     def _spec_model_name(cls: type) -> str:
#         """
#         Return a deterministic, collision-free name for the spec model.

#         Example:
#             pkg_a.models.Leaf            ->  pkg_a_models_LeafSpec
#             pkg_b.sub.Leaf.Inner         ->  pkg_b_sub_Leaf_InnerSpec
#         """
#         path = f"{cls.__module__}.{cls.__qualname__}" # .replace('.', '_')
#         return f"{path}Spec"
    
#     @classmethod
#     def __build_schema_hook__(
#         cls, 
#         name: str, 
#         type_: t.Any, 
#         default: t.Any
#     ) -> t.Any:
#         """
#         Hook for custom schema building logic.
#         This is called for each field in the class.
#         """
#         raise ValueError(
#             f"Unknown build schema hook name: {name}. "
#             "This should be implemented in the subclass."
#         )
    
#     @classmethod
#     def __convert_type_to_spec__(cls, typ: t.Any) -> t.Any:
#         """Convert a type annotation to its spec equivalent.

#         Handles Union types (both typing.Union and types.UnionType) by converting
#         each Union member that is a BaseModule to its spec model.

#         Args:
#             typ: Type annotation to convert

#         Returns:
#             Converted type annotation with BaseModule types replaced by their spec models
#         """
#         origin = t.get_origin(typ)

#         if origin is t.Union or isinstance(origin, type) and issubclass(origin, types.UnionType):
#             union_args = t.get_args(typ)
#             converted_args = []
#             for arg in union_args:
#                 if isinstance(arg, type) and issubclass(arg, BaseModule):
#                     converted_args.append(arg.schema_model())
#                 else:
#                     converted_args.append(arg)
#             return t.Union[tuple(converted_args)]

#         base_type = origin if origin is not None else typ
#         if isinstance(base_type, type) and issubclass(base_type, BaseModule):
#             return base_type.schema_model()

#         return typ

#     @classmethod
#     def __build_schema__(cls) -> None:
#         """
#         Collect fields from *all* ancestors, then merge/override with annotations
#         found on *cls* itself.  When an InitVar is later replaced by a method or
#         property of the same name, we treat that as 'no default supplied' rather
#         than letting the callable leak into runtime initialisation.
#         """

#         # Check if class inherits from Generic - if so, require explicit Spec definition
#         for base in getattr(cls, '__orig_bases__', []):
#             if t.get_origin(base) is t.Generic:
#                 raise TypeError(
#                     f"{cls.__name__} inherits from Generic and must define its Spec class explicitly. "
#                     f"Add a nested class like: class __spec__(BaseSpec, {base}): ..."
#                 )

#         parent_fields: dict[str, tuple[t.Any, t.Any, bool]] = {}
#         for base in cls.__mro__[1:]:
#             if hasattr(base, "__item_fields__"):
#                 for n, typ, dflt, is_init in base.__item_fields__:
#                     parent_fields.setdefault(n, (typ, dflt, is_init))


#         ann = t.get_type_hints(cls, include_extras=True)
#         for name, typ in ann.items():
#             if t.get_origin(typ) is t.ClassVar:
#                 continue

#             dflt  = getattr(cls, name, inspect._empty)
#             is_iv = isinstance(typ, InitVar)

#             if is_iv and (
#                 inspect.isfunction(dflt) or isinstance(dflt, property)
#             ):
#                 # A callable of the same name replaced the field; treat as if the
#                 # user supplied *no* default.  We fall back to None so that
#                 # __init__ accepts a missing keyword and __post_init__ can decide
#                 # what to do.
#                 dflt = None

#             if is_iv:
#                 typ = (
#                     t.get_args(typ)[0]
#                     if t.get_origin(typ) is InitVar else t.Any
#                 )

#             parent_fields[name] = (typ, dflt, is_iv)

#         # 3⃣  write back to canonical structures -------------------------------
#         cls.__item_fields__ = [
#             (n, *parent_fields[n]) for n in parent_fields
#         ]
#         cls.__is_initvar__ = {n: iv for n, (_, _, iv) in parent_fields.items()}

#         # 4⃣  build / rebuild the pydantic spec --------------------------------
#         spec_fields: dict[str, tuple[t.Any, t.Any]] = {}
#         for n, typ, dflt, _ in cls.__item_fields__:
#             # Check if field has a custom schema hook
#             if n in cls.__spec_hooks__:
#                 origin = cls.__build_schema_hook__(n, typ, dflt)
#             # Check if field is a BaseFieldDescriptor (modfield/modlistfield/moddictfield)
#             elif isinstance(dflt, BaseFieldDescriptor):
#                 # Get spec annotation from descriptor (already validated in __set_name__)
#                 origin = dflt.get_spec_annotation()
#                 dflt = ...  # modfields are required unless explicitly marked optional
#             # Check if field is a dataclasses.Field (from dataclasses.field())
#             elif hasattr(dflt, 'default') and hasattr(dflt, 'default_factory'):
#                 # This is a dataclasses.Field - extract the actual default
#                 if isinstance(dflt, DataclassField):
#                     if dflt.default is not MISSING:
#                         origin = cls.__convert_type_to_spec__(typ)
#                         dflt = dflt.default
#                     elif dflt.default_factory is not MISSING:
#                         origin = cls.__convert_type_to_spec__(typ)
#                         dflt = dflt.default_factory
#                     else:
#                         origin = cls.__convert_type_to_spec__(typ)
#                         dflt = ...
#                 else:
#                     origin = cls.__convert_type_to_spec__(typ)
#             else:
#                 origin = cls.__convert_type_to_spec__(typ)

#             spec_fields[n] = (origin, ... if dflt is inspect._empty else dflt)

#         # Find all parent spec classes for proper inheritance
#         parent_specs = []
#         for base in cls.__bases__:
#             if hasattr(base, '__spec__') and issubclass(base, BaseModule):
#                 parent_specs.append(base.__spec__)

#         # Use parent specs if found, otherwise BaseSpec
#         # If multiple parent specs, use tuple for multiple inheritance
#         if len(parent_specs) > 1:
#             spec_base = tuple(parent_specs)
#         elif len(parent_specs) == 1:
#             spec_base = parent_specs[0]
#         else:
#             spec_base = BaseSpec

#         cls.__spec__ = create_model(
#             f"{cls._spec_model_name()}",
#             __base__       = spec_base,
#             kind           = (t.Literal[cls.__qualname__], cls.__qualname__),
#             # model_config   = ConfigDict(arbitrary_types_allowed=True),
#             **spec_fields,
#         )

#     def __init_subclass__(cls, **__kwd):
#         super().__init_subclass__(**__kwd)
#         if cls is BaseModule:
#             return
        
#         if '__spec__' not in cls.__dict__:
#             cls.__build_schema__()

#     def __init__(self, **__kwd: t.Any):
#         self._children = []
#         self._init_vars = {}
#         self._parameters: dict[str, Param] = {}
#         self._states: dict[str, Attr] = {}
#         self._modules: dict[str, BaseModule] = {}

#         for name, _typ, default, is_init in self.__class__.__item_fields__:
#             if name in __kwd:
#                 val = __kwd.pop(name)
#             elif default is not inspect._empty:
#                 # Handle BaseFieldDescriptor (modfield/modlistfield/moddictfield)
#                 if isinstance(default, BaseFieldDescriptor):
#                     try:
#                         val = default.get_default()
#                     except TypeError:
#                         raise TypeError(f"Missing required keyword argument: {name!r}")
#                 # Handle dataclasses.Field with default_factory
#                 elif isinstance(default, DataclassField):
#                     if default.default_factory is not MISSING:
#                         val = default.default_factory()
#                     elif default.default is not MISSING:
#                         val = default.default
#                     else:
#                         raise TypeError(f"Missing required keyword argument: {name!r}")
#                 else:
#                     val = default
#             else:
#                 raise TypeError(f"Missing required keyword argument: {name!r}")

#             if is_init:
#                 self._init_vars[name] = val
#             else:
#                 setattr(self, name, val)

#         if __kwd:
#             raise TypeError(
#                 f"Unexpected keyword arguments: {', '.join(__kwd)}"
#             )

#         # child registration
#         for v in vars(self).values():
#             if isinstance(v, BaseModule):
#                 self._children.append(v)

#         if hasattr(self, "__post_init__"):
#             # Check if __post_init__ accepts all InitVars before calling

#             # TODO: Consider whether
#             # to move this to __init_subclass__
#             sig = inspect.signature(self.__post_init__)
#             accepted_params = set(sig.parameters.keys())
#             unexpected = set(self._init_vars.keys()) - accepted_params
#             if unexpected:
#                 raise RuntimeError(
#                     f"__post_init__ does not accept InitVars passed in: {unexpected}. "
#                     f"Accepted parameters: {accepted_params}. InitVars: {self._init_vars}"
#                 )
#             self.__post_init__(**self._init_vars)
#         elif len(self._init_vars) > 0:
#             raise RuntimeError(
#                 'InitVars have been defined but there is no __post_init__ defined.'
#             )

#     @classmethod
#     def schema(cls) -> dict:
#         """Return the Pydantic schema dict for the Spec."""
#         return cls.schema_model().model_json_schema()

#     # schema & spec helpers
#     @classmethod
#     def schema_model(cls) -> type[BaseSpec]:
#         return cls.__spec__

#     # ---- sub-module traversal ----------------------------------------
#     def modules(
#         self, *, 
#         recurse: bool = True, 
#         f: t.Callable[['BaseModule'], bool] | None = None):
#         """Yield **self** first, then all sub-items depth-first."""
#         if f is None or f(self):
#             yield self
#             if recurse:
#                 for child in self._modules.values():
#                     yield from child.modules(recurse=True, f=f)

#     def named_modules(
#         self, *, 
#         recurse: bool = True, 
#         prefix: str = "", 
#         f: t.Callable[['BaseModule'], bool] | None = None
#     ):
#         """Yield ``(dotted_name, module)`` pairs."""
#         if f is None or f(self):
#             yield prefix.rstrip("."), self
#             if recurse:
#                 for name, child in self._modules.items():
#                     child_prefix = f"{prefix}{name}."
#                     yield from child.named_modules(recurse=True, prefix=child_prefix)

#     def named_parameters(
#         self, *, recurse: bool = True, prefix: str = ""
#     ) -> t.Generator[tuple[str, Param], None, None]:
#         """
#         Yield all parameter names and their corresponding Param objects.
#         """
#         for name, p in self._parameters.items():
#             #if train_only is None or p.training is train_only:
#             # if train_only and isinstance(p, Param) or not train_only:
#                 # yield only if training is True or train_only is None
#                 # (i.e. we want all parameters)
#             yield f"{prefix}{name}", p
#         if recurse:
#             for cname, child in self._modules.items():
#                 child_prefix = f"{prefix}{cname}."
#                 yield from child.named_parameters(recurse=True, prefix=child_prefix)

#     def children(self):
#         """Immediate child modules (non-recursive)."""
#         return self._modules.values()

#     def named_states(self, *, recurse: bool = True, prefix: str = ""):
#         """
#         Yield all states names and their corresponding Attr objects.
#         """
#         for name, s in self._states.items():
#             yield f"{prefix}{name}", s
#         if recurse:
#             for cname, child in self._modules.items():
#                 child_prefix = f"{prefix}{cname}."
#                 yield from child.named_states(recurse=True, prefix=child_prefix)

#     def apply(self, fn, *, include: t.Callable[[t.Any], bool] | t.Type | None = None):
#         """
#         Recursively apply *fn* to every registered object.

#         If *filter_type* is given, only objects satisfying
#         ``isinstance(obj, filter_type)`` are passed to *fn*.
#         """
#         targets = [self, *self._parameters.values(), *self._states.values()]
#         for obj in targets:
#             print(obj, include)
#             if include is None:
#                 print('Including None')
#                 fn(obj)
#             elif isinstance(include, t.Type) and isinstance(obj, include):
#                 print('Including type')
#                 fn(obj)
#             elif not isinstance(include, t.Type) and include(obj):
#                 print('Including f')
#                 fn(obj)
#         for child in self._modules.values():
#             child.apply(fn, include=include)

#     def eval_args(self):
#         """Alias for ``train(False)``."""
#         return self.train(False)

#     def train(self, mode: bool = True):
#         """Recursively set ``Param.training`` for all parameters."""
#         self.training = mode
#         for child in self._modules.values():
#             child.train(mode)
#         return self

#     def named_children(self):
#         """
#         Yield all child module names and their corresponding modules.
#         """
#         return self._modules.items()

#     def spec_hook(
#         self, *, 
#         name: str,
#         val: t.Any,
#         to_dict: bool = False,
#     ):
#         """
#         Serialise *this* runtime object → its spec counterpart.

#         Nested `BaseModule` instances are recursively converted.
#         `ModuleList` containers are converted element-wise.
#         """
#         raise ValueError(
#             f"Unknown from_spec_hook name: {name}. "
#             "This should be implemented in the subclass."
#         )

#     @classmethod
#     def from_spec_hook(
#         cls,
#         name: str,
#         val: t.Any,
#         ctx: "dict | None" = None,
#     ) -> t.Any:
#         """
#         Hook for the registry to call when a spec is encountered.
#         This is used to create a ModuleList from a spec.
#         """
#         raise ValueError(
#             f"Unknown from_spec_hook name: {name}. "
#             "This should be implemented in the subclass."
#         )

#     def spec(
#         self, *, 
#         to_dict: bool = False
#     ):
#         """
#         Serialise *this* runtime object → its spec counterpart.

#         Nested `BaseModule` instances are recursively converted.
#         `ModuleList` containers are converted element-wise.
#         """
#         data: dict[str, t.Any] = {}

#         for name, is_init in self.__class__.__is_initvar__.items():
#             if is_init:
#                 val = self._init_vars[name]
#                 if name in self.__spec_hooks__:
#                     val = self.spec_hook(
#                         name=name,
#                         val=val,
#                         to_dict=to_dict
#                     )
#                 data[name] = val
#                 continue

#             val = getattr(self, name)
#             if name in self.__spec_hooks__:
#                 # run custom spec hook if defined
#                 data[name] = self.spec_hook(
#                     name=name,
#                     val=val,
#                     to_dict=to_dict
#                 )
#             elif isinstance(val, BaseModule):
#                 data[name] = val.spec(to_dict=False)
            
#             else:
#                 data[name] = val

#         spec_obj = self.__class__.__spec__(
#             kind=self.__class__.__qualname__,
#             id=str(id(self)),
#             **data
#         )
#         return spec_obj.model_dump() if to_dict else spec_obj

#     @classmethod
#     def from_spec(
#         cls,
#         spec:  BaseSpec | dict,
#         ctx:   "dict | None" = None,
#     ):
#         """
#         Rebuild a runtime `BaseModule` from *spec*.

#         • `ctx` caches already-created objects so identical specs
#         resolve to the same instance.
#         • Works for nested BaseModules       (key = spec['id'])
#         • Works for Param / State / Shared   (key = spec['ref_name'])
#         """
#         ctx = ctx or {}

#         # ---- 1) normalise input -----------------------------------------
        
#         if isinstance(spec, dict) and "kind" in spec:
#             spec_obj: BaseSpec = cls.__spec__.model_validate(spec)
#         else:                                       # already a BaseSpec
#             spec_obj = spec

#         if isinstance(spec_obj, BaseSpec):
#             key = spec_obj.id                       # BaseModule path
#         # elif isinstance(spec_obj, dict) and "ref_name" in spec_obj:
#         #     key = spec_obj["ref_name"]              # Shared / Param / State path
#         else:
#             key = None                              # primitives → no dedup

#         if key and (hit := ctx.get(key)) is not None:
#             return hit                              # reuse existing object

#         kwargs: dict[str, t.Any] = {}

#         for name, is_init in cls.__is_initvar__.items():
#             val = getattr(spec_obj, name)

#             cls_val = cls.__dict__.get(name)
#             if (
#                 (isinstance(val, dict) 
#                  and "kind" in val 
#                  and cls_val is not None 
#                  and issubclass(
#                 cls_val, BaseModule)) 
#                 or isinstance(val, BaseSpec)
#                 or name in cls.__spec_hooks__
#             ):
#             #    pass
#             # (a) Nested BaseModule spec  -----------------------------
#             # if isinstance(val, (BaseSpec, dict)):

#                 if isinstance(val, BaseSpec) and val.id in ctx:
#                     val = ctx.get(val.id)  # reuse existing module

#                 elif name in cls.__spec_hooks__:
#                     # run custom spec hook if defined
#                     # id = val.id
#                     val = cls.from_spec_hook(
#                         name=name,
#                         val=val,
#                         ctx=ctx
#                     )
#                     # ctx[id] = val
#                 elif isinstance(val, BaseSpec):
#                     id = val.id
#                     sub_cls = mod_registry[val.kind].obj
#                     val = sub_cls.from_spec(val, ctx)
#                     ctx[id] = val
#                 else:
#                     # allow dicts with 'kind' to be parsed as BaseSpec
#                     id = val['id']
#                     sub_cls = mod_registry[val["kind"]].obj
#                     val = sub_cls.from_spec(val, ctx)
#                     ctx[id] = val
                
#             kwargs[name] = val

#         # ---- 4) construct this module ----------------------------------
#         obj = cls(**kwargs)

#         # ---- 5) cache for future duplicates ----------------------------
#         if key:
#             ctx[key] = obj
#             # ctx.put(key, obj)

#         return obj

#     def __setattr__(self, name, value):
#         if isinstance(value, Param):
#             self.register_parameter(name, value)
#         elif isinstance(value, Attr):
#             self.register_state(name, value)
#         elif isinstance(value, BaseModule):
#             self.register_module(name, value)
#         else:
#             super().__setattr__(name, value)

#     def register_parameter(self, name: str, param: Param):
#         """Register a parameter with the given name."""
#         self._parameters[name] = param
#         super().__setattr__(name, param)

#     def register_state(self, name: str, state: Attr):
#         """Register a state with the given name."""
#         self._states[name] = state
#         super().__setattr__(name, state)

#     def register_module(self, name: str, module: 'BaseModule'):
#         """Register a submodule with the given name."""
#         self._modules[name] = module
#         super().__setattr__(name, module)

#     def parameters(self, *, recurse: bool = True, _seen: t.Optional[set[int]] = None, with_annotations: bool=False) -> t.Iterator[Param | tuple[Param, t.Any]]:
#         """
#         Yield all parameter names and their corresponding Param objects.
#         """
#         if _seen is None:
#             _seen = set()

#         for name, (param, state_type) in self._states.items():
#             if state_type != StateType.PARAM:
#                 continue
#             if id(param) not in _seen:
#                 # if train_only is None or param.training is train_only:
#                 _seen.add(id(param))
#                 if with_annotations:
#                     annotation = self.__annotations__.get(name, t.Any)
#                     yield (param, annotation)
#                 else:
#                     yield param

#         if recurse:
#             for child in self._modules.values():
#                 yield from child.parameters(recurse=True, _seen=_seen, with_annotations=with_annotations)

#     def state_dict(
#         self,
#         *,
#         recurse: bool = True,
#         train: bool = True,
#         runtime: bool = True,
#     ) -> dict[str, t.Any]:
#         """
#         Returns a dictionary representation of the module's state.

#         Args:
#             recurse: Whether to recurse into child modules.
#             train: Whether to include training parameters (Param).
#             runtime: Whether to include runtime states (Attr).
#         """

#         out: dict[str, t.Any] = {}

#         if train:
#             for name, param in self._parameters.items():
#                 out[name] = param.data

#         if runtime:
#             for name, state in self._states.items():
#                 out[name] = state.data

#         # Recurse into child BaseItems
#         if recurse:
#             for name, child in self._modules.items():
#                 child_sd = child.state_dict(recurse=True, train=train, runtime=runtime)
#                 for sub_name, value in child_sd.items():
#                     out[f"{name}.{sub_name}"] = value

#         return out

#     def state_keys(
#         self,
#         *,
#         recurse: bool = True,
#         train: bool = True,
#         runtime: bool = True
#     ) -> set[str]:
#         """
#         Returns a set of dotted keys representing the structure of the state_dict.
#         """
#         keys = set()

#         def _collect(obj: BaseModule, prefix: str):
#             for name, v in vars(obj).items():
#                 path = f"{prefix}{name}"
#                 if isinstance(v, Param) and train:
#                     keys.add(path)
#                 elif isinstance(v, Attr) and runtime:
#                     keys.add(path)
#                 elif isinstance(v, BaseModule) and recurse:
#                     _collect(v, path + ".")
#         _collect(self, "")
#         return keys

#     def load_state_dict(
#         self,
#         sd: dict[str, t.Any],
#         *,
#         recurse: bool = True,
#         train: bool = True,
#         runtime: bool = True,
#         strict: bool = True
#     ):
#         """
#         Load the state dictionary into the module.

#         Args:
#             sd: The state dictionary to load.
#             recurse: Whether to recurse into child modules.
#             train: Whether to include training parameters (Param).
#             runtime: Whether to include runtime states (Attr).
#             strict: Whether to enforce strict loading (i.e., all keys must match).
#         """
#         if not isinstance(sd, dict):
#             raise TypeError(
#                 f"StateDict must be of type dict not {type(sd)}"
#             )
#         found = set()

#         # Load Params
#         if train:
#             for name, param in self._parameters.items():
#                 if name in sd:
#                     param.data = sd[name]
#                     found.add(name)

#         # Load States
#         if runtime:
#             for name, state in self._states.items():
#                 if name in sd:
#                     state.data = sd[name]
#                     found.add(name)

#         # Recurse into submodules
#         if recurse:
#             for name, child in self._modules.items():
#                 child_sd = {k[len(name)+1:]: v for k, v in sd.items() if k.startswith(f"{name}.")}
#                 child.load_state_dict(child_sd, recurse=True, train=train, runtime=runtime, strict=False)
#                 found.update(f"{name}.{k}" for k in child_sd.keys())

#         if strict:
#             expected_keys = self.state_keys(recurse=recurse, train=train, runtime=runtime)
#             passed_keys = set(sd.keys())
#             missing_keys = expected_keys - passed_keys
#             extra_keys = passed_keys - expected_keys

#             if missing_keys:
#                 raise KeyError(f"Missing keys in load_state_dict: {sorted(missing_keys)}")
#             if extra_keys:
#                 raise KeyError(
#                     f"Unexpected keys in load_state_dict: {sorted(extra_keys)}"
#                 )


# class BaseSpec(BaseModel):
#     """Base class for Specs
#     Specs are automatically subclassed by BaseModule 
#     to create a Spec for that Module. It can
#     manually be subclassed if needed.
#     """

#     kind : str
#     id : str = Field(
#         default_factory=lambda: str(uuid4())
#     )
#     style: t.Literal['structured'] = 'structured'

#     model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)

#     @classmethod
#     def class_kind(cls) -> str:
#         """
#         Return the constant literal value of the `kind` field that was
#         set by create_model().  Raises if it hasn’t been frozen.
#         """
#         default = cls.model_fields['kind'].default
#         if default is None:
#             raise RuntimeError(f"{cls.__name__} has no fixed `kind` default")
#         return default

#     @classmethod
#     def load_cls(cls):
#         kind = cls.class_kind()
#         if kind not in mod_registry.list_entries():
#             raise ValueError(f"Class kind '{kind}' not registered in registry.")
#         return mod_registry[kind].obj


# class RestrictedSchemaMixin(ABC):
#     """
#     Base mixin for creating restricted JSON schemas.

#     Subclasses must implement:
#     - restricted_schema(**kwargs) -> dict
#     - _schema_process_variants(variants, mixin_class, ...) -> list[dict]
#     """

#     @classmethod
#     @abstractmethod
#     def restricted_schema(
#         cls,
#         *,
#         _profile: str = "shared",
#         _seen: dict | None = None,
#         **kwargs
#     ) -> dict:
#         """Generate restricted schema. Must be implemented by subclasses."""
#         raise NotImplementedError()

#     @classmethod
#     def _schema_process_variants(
#         cls,
#         variants: list,
#         *,
#         filter_fn: t.Callable | None = None,
#         restricted_schema_cls: t.Type['RestrictedSchemaMixin'],
#         _seen: dict | None = None,
#         **recursive_kwargs
#     ) -> list[dict]:
#         """
#         Process variants and return their schema dicts.

#         For each variant:
#         - Apply filter_fn if provided
#         - If variant is a class subclassing restricted_schema_cls, call restricted_schema() as classmethod
#         - If variant is an instance of restricted_schema_cls, call its restricted_schema() recursively
#         - Else use normalize_schema_type_variants to convert it to a schema

#         Args:
#             variants: List of task/state instances or classes
#             filter_fn: Optional filter (e.g., lambda t: isinstance(t, Action))
#             restricted_schema_cls: The domain-specific mixin class to check for recursion
#             _seen: Cycle detection dict
#             **recursive_kwargs: Passed to nested restricted_schema() calls (e.g., tasks=...)

#         Returns:
#             List of schema dicts

#         Raises:
#             TypeError: If variant cannot be converted to a schema
#         """
#         if filter_fn is not None:
#             variants = [v for v in variants if filter_fn(v)]

#         schemas = []
#         for variant in variants:
#             # Check if variant is a CLASS that subclasses the domain-specific mixin
#             if isinstance(variant, type) and issubclass(variant, restricted_schema_cls):
#                 # Call restricted_schema as classmethod - no instance needed!
#                 schemas.append(variant.restricted_schema(_seen=_seen, **recursive_kwargs))

#             # Check if variant is an INSTANCE of the domain-specific mixin
#             elif isinstance(variant, restricted_schema_cls):
#                 # Call restricted_schema on the instance (delegates to classmethod)
#                 schemas.append(variant.__class__.restricted_schema(_seen=_seen, **recursive_kwargs))

#             # Otherwise, use normalize_schema_type_variants for regular cases
#             else:
#                 # This handles: module classes, spec classes, spec instances, schema dicts
#                 entries = cls.normalize_schema_type_variants([variant])
#                 if entries:
#                     schemas.append(entries[0][1])  # Extract the schema dict
#                 else:
#                     raise TypeError(f"Variant {variant} could not be normalized to a schema")

#         return schemas

#     # @classmethod
#     # def normalize_schema_type_variants(
#     #     cls,
#     #     objs: t.Iterable[t.Any],
#     # ) -> list[tuple[str, dict]]:
#     #     """
#     #     Convert allowed type variants into standardized (name, schema_dict) entries.

#     #     Supported input formats:
#     #         • Module class:        MyTask            -> ("MyTaskSpec", {...schema...})
#     #         • Spec model class:    MyTaskSpec        -> ("MyTaskSpec", {...schema...})
#     #         • Spec instance:       my_task_spec_obj  -> ("MyTaskSpec", {...schema...})
#     #         • Raw schema dict:     {"title": "X", ...} or {"$id": ".../X", ...} -> ("X", {...schema...})

#     #     Args:
#     #         objs: Iterable of allowed types in any supported format

#     #     Returns:
#     #         List of (spec_name, schema_dict) tuples, de-duplicated and sorted by name
#     #     """
#     #     entries: list[tuple[str, dict]] = []

#     #     for o in objs:
#     #         # Module class
#     #         if isinstance(o, type) and issubclass(o, Module):
#     #             schema_dict = o.schema()
#     #             name = cls._schema_name_from_dict(schema_dict)
#     #             entries.append((name, schema_dict))
#     #             continue
#     #         # Spec model class
#     #         if isinstance(o, type) and issubclass(o, Module):
#     #             entries.append((o.__name__, o.model_json_schema()))
#     #             continue
#     #         # Spec instance
#     #         if not isinstance(o, dict) and hasattr(o, "__class__") and isinstance(o.__class__, type) and issubclass(o.__class__, Module):
#     #             sm = o.__class__
#     #             entries.append((sm.__name__, sm.model_json_schema()))
#     #             continue
#     #         # Raw dict schema
#     #         if isinstance(o, dict):
#     #             name = cls._schema_name_from_dict(o)
#     #             entries.append((name, o))
#     #             continue
#     #         raise TypeError(f"Unsupported variant type: {type(o)!r}")

#     #     # de-dupe (last wins), then deterministic order
#     #     dedup: dict[str, dict] = {}
#     #     for name, doc in entries:
#     #         dedup[name] = doc
#     #     return [(name, dedup[name]) for name in sorted(dedup.keys())]

#     # =========================
#     # 3 Update Helpers (implemented in base)
#     # =========================

#     @classmethod
#     def _schema_update_list_field(
#         cls,
#         schema: dict,
#         *,
#         field_name: str,
#         placeholder_name: str,
#         variant_schemas: list[dict],
#         profile: str = "shared"
#     ) -> dict:
#         """
#         Update a ModuleList field in the schema.

#         Path: ["properties", field_name, "items"]

#         Handles nullable fields: If field is "ModuleList[T] | None", wraps in anyOf.

#         Args:
#             schema: The base schema dict to update
#             field_name: Name of the ModuleList field (e.g., "tasks")
#             placeholder_name: Name of placeholder spec (e.g., "TaskSpec")
#             variant_schemas: List of schema dicts for allowed variants
#             profile: "shared" (use $defs/Allowed_*) or "inline" (use oneOf)

#         Returns:
#             Updated schema dict
#         """
#         entries = [(cls._schema_name_from_dict(s), s) for s in variant_schemas]
#         cls._schema_require_defs_for_entries(schema, entries)

#         # Build the union
#         if profile == "shared":
#             union_ref = cls._schema_ensure_shared_union(
#                 schema,
#                 placeholder_name=placeholder_name,
#                 entries=entries
#             )
#             replacement = {"$ref": union_ref}
#         else:
#             replacement = cls._schema_make_union_inline(entries)

#         # Check if field is nullable (has anyOf with null)
#         field_schema = schema.get("properties", {}).get(field_name, {})
#         if "anyOf" in field_schema:
#             # Nullable field: update the items in the array part of anyOf
#             # Structure: {"anyOf": [{"type": "array", "items": {...}}, {"type": "null"}]}
#             for option in field_schema["anyOf"]:
#                 if isinstance(option, dict) and option.get("type") == "array":
#                     option["items"] = replacement
#                     break
#         else:
#             # Non-nullable field: directly update items
#             cls._schema_replace_at_path(schema, ["properties", field_name, "items"], replacement)

#         return schema

#     @classmethod
#     def _schema_update_dict_field(
#         cls,
#         schema: dict,
#         *,
#         field_name: str,
#         placeholder_name: str,
#         variant_schemas: list[dict],
#         profile: str = "shared"
#     ) -> dict:
#         """
#         Update a ModuleDict field in the schema.

#         Path: ["properties", field_name, "additionalProperties"]

#         Args:
#             schema: The base schema dict to update
#             field_name: Name of the ModuleDict field (e.g., "states")
#             placeholder_name: Name of placeholder spec (e.g., "BaseStateSpec")
#             variant_schemas: List of schema dicts for allowed variants
#             profile: "shared" or "inline"

#         Returns:
#             Updated schema dict
#         """
#         # Merge $defs from all variant schemas (Pattern A: pass-through schemas may have nested $defs)
#         for variant_schema in variant_schemas:
#             if "$defs" in variant_schema:
#                 if "$defs" not in schema:
#                     schema["$defs"] = {}
#                 schema["$defs"].update(variant_schema["$defs"])

#         entries = [(cls._schema_name_from_dict(s), s) for s in variant_schemas]
#         cls._schema_require_defs_for_entries(schema, entries)

#         if profile == "shared":
#             union_ref = cls._schema_ensure_shared_union(
#                 schema,
#                 placeholder_name=placeholder_name,
#                 entries=entries
#             )
#             replacement = {"$ref": union_ref}
#         else:
#             replacement = cls._schema_make_union_inline(entries)

#         cls._schema_replace_at_path(
#             schema,
#             ["properties", field_name, "additionalProperties"],
#             replacement
#         )

#         return schema

#     @classmethod
#     def _schema_update_single_field(
#         cls,
#         schema: dict,
#         *,
#         field_name: str,
#         placeholder_name: str,
#         variant_schemas: list[dict],
#         profile: str = "shared"
#     ) -> dict:
#         """
#         Update a single module field in the schema.

#         Path: ["properties", field_name]

#         Args:
#             schema: The base schema dict to update
#             field_name: Name of the single field (e.g., "root")
#             placeholder_name: Name of placeholder spec (e.g., "TaskSpec")
#             variant_schemas: List of schema dicts for allowed variants
#             profile: "shared" or "inline"

#         Returns:
#             Updated schema dict
#         """
#         # Merge $defs from all variant schemas (Pattern A: pass-through schemas may have nested $defs)
#         for variant_schema in variant_schemas:
#             if "$defs" in variant_schema:
#                 if "$defs" not in schema:
#                     schema["$defs"] = {}
#                 schema["$defs"].update(variant_schema["$defs"])

#         entries = [(cls._schema_name_from_dict(s), s) for s in variant_schemas]
#         cls._schema_require_defs_for_entries(schema, entries)

#         if profile == "shared":
#             union_ref = cls._schema_ensure_shared_union(
#                 schema,
#                 placeholder_name=placeholder_name,
#                 entries=entries
#             )
#             replacement = {"$ref": union_ref}
#         else:
#             replacement = cls._schema_make_union_inline(entries)

#         cls._schema_replace_at_path(schema, ["properties", field_name], replacement)

#         return schema

#     @classmethod
#     def _schema_merge_defs(
#         cls,
#         target_schema: dict,
#         *source_schemas: dict
#     ) -> dict:
#         """
#         Merge $defs from source schemas into target schema.

#         This is specifically for Pattern A (pass-through) where a child's restricted
#         schema has its own $defs that need to be hoisted to the parent's root-level $defs.

#         JSON Schema $ref with "#/$defs/X" always looks at the document root, not nested
#         $defs, so all definitions must be at the root level.

#         Args:
#             target_schema: Schema to merge definitions into
#             *source_schemas: One or more schemas whose $defs to merge

#         Returns:
#             The target_schema (modified in place)

#         Example:
#             # Region has restricted states in its $defs
#             region_schema = Region.restricted_schema(states=[StateA, StateB])

#             # Merge Region's $defs into CompositeState's $defs
#             composite_schema = CompositeState.schema()
#             cls._schema_merge_defs(composite_schema, region_schema)
#             # Now composite_schema["$defs"] contains Allowed_BaseStateSpec, etc.
#         """
#         if "$defs" not in target_schema:
#             target_schema["$defs"] = {}

#         for source_schema in source_schemas:
#             if "$defs" in source_schema:
#                 target_schema["$defs"].update(source_schema["$defs"])

#         return target_schema

#     # =========================
#     # Low-Level Schema Helpers
#     # =========================

#     @staticmethod
#     def _schema_name_from_dict(schema_dict: dict) -> str:
#         """
#         Extract spec name from schema dict.

#         Tries 'title' first, then tail of '$id'.

#         Args:
#             schema_dict: JSON schema dict

#         Returns:
#             Spec name (e.g., "TaskSpec")

#         Raises:
#             TypeError: If no title or $id found
#         """
#         if "title" in schema_dict and isinstance(schema_dict["title"], str):
#             return schema_dict["title"].strip()

#         if "$id" in schema_dict and isinstance(schema_dict["$id"], str):
#             _id = schema_dict["$id"].strip()
#             # Remove trailing # first, then get tail after last /
#             tail = _id.rstrip("#").rsplit("/", 1)[-1]
#             if tail:
#                 return tail

#         raise TypeError("Schema dict must have 'title' or '$id' to derive spec name")

#     @staticmethod
#     def _schema_require_defs_for_entries(schema: dict, entries: list[tuple[str, dict]]) -> None:
#         """
#         Add entries to $defs if not already present.

#         Args:
#             schema: Schema dict to update
#             entries: List of (name, schema_dict) tuples
#         """
#         defs = schema.setdefault("$defs", {})
#         for name, entry_schema in entries:
#             defs.setdefault(name, entry_schema)

#     @staticmethod
#     def _schema_build_refs(entries: list[tuple[str, dict]]) -> list[dict]:
#         """
#         Convert entries to $ref list for oneOf.

#         Args:
#             entries: List of (name, schema_dict) tuples

#         Returns:
#             List of {"$ref": "#/$defs/<name>"} dicts
#         """
#         return [{"$ref": f"#/$defs/{name}"} for name, _ in entries]

#     @staticmethod
#     def _schema_make_union_inline(entries: list[tuple[str, dict]]) -> dict:
#         """
#         Create inline oneOf union.

#         Args:
#             entries: List of (name, schema_dict) tuples

#         Returns:
#             {"oneOf": [...]} dict
#         """
#         return {"oneOf": RestrictedSchemaMixin._schema_build_refs(entries)}

#     @staticmethod
#     def _schema_allowed_union_name(placeholder_name: str) -> str:
#         """
#         Generate name for shared union in $defs.

#         Args:
#             placeholder_name: Original placeholder name (e.g., "TaskSpec")

#         Returns:
#             Allowed union name (e.g., "Allowed_TaskSpec")
#         """
#         return f"Allowed_{placeholder_name}"

#     @classmethod
#     def _schema_ensure_shared_union(
#         cls,
#         schema: dict,
#         *,
#         placeholder_name: str,
#         entries: list[tuple[str, dict]]
#     ) -> str:
#         """
#         Ensure shared union exists in $defs and return its $ref.

#         Args:
#             schema: Schema dict to update
#             placeholder_name: Original placeholder name
#             entries: List of (name, schema_dict) tuples

#         Returns:
#             Reference string (e.g., "#/$defs/Allowed_TaskSpec")
#         """
#         defs = schema.setdefault("$defs", {})
#         allowed_name = cls._schema_allowed_union_name(placeholder_name)

#         if allowed_name not in defs:
#             defs[allowed_name] = cls._schema_make_union_inline(entries)

#         return f"#/$defs/{allowed_name}"

#     @staticmethod
#     def _schema_node_at(schema: dict, path: list[str]) -> t.Any:
#         """
#         Navigate to node at path in schema.

#         Args:
#             schema: Schema dict
#             path: List of keys (e.g., ["properties", "tasks", "items"])

#         Returns:
#             Node at path, or None if not found
#         """
#         cur = schema
#         for key in path:
#             if isinstance(cur, dict) and key in cur:
#                 cur = cur[key]
#             else:
#                 return None
#         return cur

#     @staticmethod
#     def _schema_replace_at_path(schema: dict, path: list[str], replacement: t.Any) -> None:
#         """
#         Replace node at path with replacement.

#         Args:
#             schema: Schema dict to update
#             path: List of keys to navigate to
#             replacement: New value to set

#         Raises:
#             ValueError: If path is empty
#             KeyError: If path is invalid
#         """
#         if not path:
#             raise ValueError("Path cannot be empty")

#         *parent_path, last_key = path
#         parent = RestrictedSchemaMixin._schema_node_at(schema, parent_path) if parent_path else schema

#         if not isinstance(parent, dict):
#             raise KeyError(f"Invalid path: {'.'.join(path)}")

#         parent[last_key] = replacement


    # def restricted_schema(
    #     self,
    #     *,
    #     filter_schema_cls: t.Type[RestrictedSchemaMixin] = type,
    #     variants: list | None = None,
    #     _profile: str = "shared",
    #     _seen: dict | None = None,
    #     **kwargs
    # ) -> tuple[dict, dict]:
    #     """Generate restricted schema for single module field."""
    #     if variants is None:
    #         base_schema = self._owner.schema()
    #         return (base_schema["properties"][self._name], {})

    #     # Check if typ is a BaseFieldTypeDescriptor - delegate to it
    #     if isinstance(self.typ, BaseFieldTypeDescriptor):
    #         variant_schemas = self._schema_process_variants(
    #             variants,
    #             restricted_schema_cls=filter_schema_cls,
    #             _seen=_seen,
    #             **kwargs
    #         )
    #         entries = [(self._schema_name_from_dict(s), s) for s in variant_schemas]
    #         return self.typ.restricted_schema(
    #             variants=entries,
    #             field_name=self._name,
    #             profile=_profile,
    #             schema_processor=None  # Already processed
    #         )

    #     # Check if typ is list with BaseFieldTypeDescriptor as first item
    #     if isinstance(self.typ, list) and len(self.typ) > 0:
    #         if isinstance(self.typ[0], BaseFieldTypeDescriptor):
    #             variant_schemas = self._schema_process_variants(
    #                 variants,
    #                 restricted_schema_cls=filter_schema_cls,
    #                 _seen=_seen,
    #                 **kwargs
    #             )
    #             entries = [(self._schema_name_from_dict(s), s) for s in variant_schemas]
    #             return self.typ[0].restricted_schema(
    #                 variants=entries,
    #                 field_name=self._name,
    #                 profile=_profile,
    #                 schema_processor=None
    #             )

    #     # Default: build simple union for plain fields
    #     variant_schemas = self._schema_process_variants(
    #         variants,
    #         restricted_schema_cls=filter_schema_cls,
    #         _seen=_seen,
    #         **kwargs
    #     )

    #     entries = [(self._schema_name_from_dict(s), s) for s in variant_schemas]

    #     if _profile == "shared":
    #         union_name = self._schema_allowed_union_name(self._name)
    #         defs = {union_name: {"oneOf": self._schema_build_refs(entries)}}
    #         for name, schema in entries:
    #             defs[name] = schema

    #         field_schema = {"$ref": f"#/$defs/{union_name}"}
    #         return (field_schema, defs)
    #     else:
    #         field_schema = self._schema_make_union_inline(entries)
    #         return (field_schema, {})

# # ============================================================================
# # Annotation Utilities
# # ============================================================================

# def flatten_annotation(annotation) -> list:
#     """Flatten Union/Optional at current level only.

#     Unwraps Union and Optional types to extract alternatives, but does not
#     recurse into generic type arguments like ModuleList[...].

#     Examples:
#         Task → [Task]
#         Task | State → [Task, State]
#         Optional[Task] → [Task, None]
#         ModuleList[Task | State] → [ModuleList[Task | State]]  # Not flattened inside

#     Args:
#         annotation: Type annotation to flatten

#     Returns:
#         List of type alternatives
#     """
#     origin = t.get_origin(annotation)
#     # Handle both typing.Union and types.UnionType (Python 3.10+ | syntax)
#     if origin is t.Union or origin is types.UnionType:
#         return list(t.get_args(annotation))
#     return [annotation]


# def is_generic_annotation(annotation) -> bool:
#     """Check if annotation is a generic type like SomeClass[...].

#     Args:
#         annotation: Type annotation to check

#     Returns:
#         True if annotation is a parameterized generic type

#     Examples:
#         is_generic_annotation(ModuleList[Task]) → True
#         is_generic_annotation(Task) → False
#         is_generic_annotation(Task | State) → True (Union is generic)
#     """
#     return t.get_origin(annotation) is not None


# def extract_generic_parts(annotation) -> tuple:
#     """Extract container class and type arguments from generic annotation.

#     Args:
#         annotation: Generic type annotation

#     Returns:
#         (origin_class, type_args) tuple, or (None, ()) if not generic

#     Examples:
#         extract_generic_parts(ModuleList[Task]) → (ModuleList, (Task,))
#         extract_generic_parts(ModuleDict[str, State]) → (ModuleDict, (str, State))
#         extract_generic_parts(Task) → (None, ())
#     """
#     origin = t.get_origin(annotation)
#     if origin is None:
#         return (None, ())
#     return (origin, t.get_args(annotation))


# class ModFieldDescriptor(BaseFieldDescriptor):
#     """Descriptor for single module field."""

#     def schema_model(self) -> t.Type:

#         sub_schemas = []
#         has_none = False
#         for typ in self.typ:
#             if typ is None:
#                 has_none = True
#             elif isinstance(typ, BaseFieldTypeDescriptor):
#                 sub_schemas.append(typ.schema_model())
#             elif isinstance(typ, type) and issubclass(typ, BaseModule):
#                 sub_schemas.append(typ.schema_model())
#         if has_none:
#             if len(sub_schemas) == 0:
#                 return type(None)  # only None
#             elif len(sub_schemas) == 1:
#                 return t.Optional[sub_schemas[0]]
#             else:
#                 return t.Optional[t.Union[tuple(sub_schemas)]]
#         else:
#             if len(sub_schemas) == 1:
#                 return sub_schemas[0]
#             else:
#                 return t.Union[tuple(sub_schemas)]

#     def schema(self) -> dict:
#         """Generate JSON schema for this field."""
#         sub_schemas = []
#         has_none = False
#         for typ in self.typ:
#             if typ is None:
#                 has_none = True
#             elif isinstance(typ, BaseFieldTypeDescriptor):
#                 sub_schemas.append(typ.schema())
#             elif isinstance(typ, type) and issubclass(typ, BaseModule):
#                 sub_schemas.append(typ.schema())

#         # Build schema from sub_schemas
#         if has_none:
#             if len(sub_schemas) == 0:
#                 return {"type": "null"}
#             elif len(sub_schemas) == 1:
#                 # Optional[T] - allow null or the schema
#                 return {"anyOf": [{"type": "null"}, sub_schemas[0]]}
#             else:
#                 # Optional[Union[...]] - allow null or any of the schemas
#                 return {"anyOf": [{"type": "null"}] + sub_schemas}
#         else:
#             if len(sub_schemas) == 1:
#                 return sub_schemas[0]
#             else:
#                 return {"anyOf": sub_schemas}

#     def restricted_schema(
#         self,
#         *,
#         filter_schema_cls: t.Type['RestrictedSchemaMixin'] = type,
#         variants: list | None = None,
#         _profile: str = "shared",
#         _seen: dict | None = None,
#         **kwargs
#     ) -> tuple[dict, dict]:
#         """Generate restricted schema for this field."""
#         if variants is None:
#             base_schema = self._owner.schema()
#             return (base_schema["properties"][self._name], {})

#         # Loop over self.typ and delegate to BaseFieldTypeDescriptor if present

#         schemas = []
#         defs = []
#         for typ in self.typ:
#             if isinstance(typ, BaseFieldTypeDescriptor):
#                 schema, def_ = typ.restricted_schema(
#                     filter_schema_cls=filter_schema_cls,
#                     variants=variants,
#                     field_name=self._name,
#                     _profile=_profile
#                 )
#                 schemas.append(schema)
#                 defs.append(def_)
#             elif isinstance(typ, type) and issubclass(typ, filter_schema_cls):
#                 # Domain mixins return just schema dicts, not tuples
#                 schema = typ.restricted_schema(
#                     variants=variants,
#                     _profile=_profile
#                 )
#                 schemas.append(schema)
#                 defs.append({})
#             elif typ is None:
#                 # None type - add null schema
#                 schemas.append({"type": "null"})
#                 defs.append({})

#             else:
#                 schema = typ.schema()
#                 schemas.append(schema)
#                 defs.append({})

#         # Merge all defs
#         merged_defs = {}
#         for def_dict in defs:
#             merged_defs.update(def_dict)

#         # Combine schemas based on count
#         if self.single_typ and len(schemas) == 1:
#             # Single type - return schema directly
#             return (schemas[0], merged_defs)
#         elif len(schemas) > 1:
#             # Multiple types - create union using anyOf
#             return ({"anyOf": schemas}, merged_defs)
#         elif len(schemas) == 0:
#             # No schemas collected - shouldn't normally happen
#             # Fall back to processing variants directly
#             variant_schemas = self._schema_process_variants(
#                 variants,
#                 restricted_schema_cls=filter_schema_cls,
#                 _seen=_seen,
#                 **kwargs
#             )
#             entries = [(self._schema_name_from_dict(s), s) for s in variant_schemas]

#             if _profile == "shared":
#                 union_name = self._schema_allowed_union_name(self._name)
#                 defs_dict = {union_name: {"oneOf": self._schema_build_refs(entries)}}
#                 for name, schema in entries:
#                     defs_dict[name] = schema
#                 field_schema = {"$ref": f"#/$defs/{union_name}"}
#                 return (field_schema, defs_dict)
#             else:
#                 defs_dict = {name: schema for name, schema in entries}
#                 field_schema = self._schema_make_union_inline(entries)
#                 return (field_schema, defs_dict)
#         else:
#             # Single schema but not single_typ? Return it anyway
#             return (schemas[0] if schemas else {"type": "null"}, merged_defs)


# def generictype(container_type: type, *type_args, **metadata):
#     """Create a GenericFieldType for use with modfield().

#     Convenience function for explicit, readable generic type specifications.

#     Args:
#         container_type: Container class (ModuleList, ModuleDict, list, dict, etc.)
#         *type_args: Type arguments
#         **metadata: Optional metadata

#     Returns:
#         GenericFieldType instance

#     Examples:
#         typ=generictype(ModuleList, Task)
#         typ=generictype(ModuleDict, str, State)
#         typ=generictype(ModuleList, Task, max_items=10)
#         typ=generictype(ModuleDict, str, generictype(ModuleList, Task))
#     """
#     return GenericFieldType(container_type, *type_args, **metadata)


# def modfield(typ=UNDEFINED, default=UNDEFINED, default_factory=UNDEFINED) -> ModFieldDescriptor:
#     """Mark field as containing a BaseModule instance.

#     Args:
#         typ: Type specification. Can be:
#             - A type (Task)
#             - A generic type (ModuleList[Task]) - auto-wrapped in GenericFieldType
#             - A BaseFieldTypeDescriptor (GenericFieldType(...))
#             - A list of types ([Task1, Task2]) - legacy
#         default: Default value
#         default_factory: Factory function for default value

#     Returns:
#         ModFieldDescriptor instance

#     Examples:
#         field: Task = modfield()
#         field: ModuleList[Task] = modfield()  # Auto-wrapped in GenericFieldType
#         field = modfield(typ=generictype(ModuleList, Task))
#     """
#     # Auto-wrap generic types in GenericFieldType
#     if typ is not UNDEFINED and not isinstance(typ, (list, BaseFieldTypeDescriptor)):
#         if is_generic_type(typ):
#             typ = GenericFieldType.from_annotation(typ)

#     # Handle list of types: typ=[ModList[Task], ...] -> convert first to GenericFieldType
#     if isinstance(typ, list) and len(typ) > 0:
#         if is_generic_type(typ[0]):
#             typ = [GenericFieldType.from_annotation(typ[0])] + typ[1:]

#     return ModFieldDescriptor(typ=typ, default=default, default_factory=default_factory)



# UNDEFINED = inspect._empty


# class BaseFieldTypeDescriptor(ABC):
#     """Abstract base class for field type descriptors.

#     Type descriptors encapsulate knowledge about how to handle specific type patterns
#     (e.g., generics, unions, custom containers). They can be passed to modfield() to
#     specify how types should be extracted, validated, and converted to schemas.

#     This makes the system extensible - users can create custom type descriptors.
#     """

#     @abstractmethod
#     def schema(self) -> dict:
#         """Generate JSON schema for this type descriptor.

#         Returns:
#             Schema dict
#         """
#         raise NotImplementedError

#     @abstractmethod
#     def restricted_schema(
#         self,
#         *,
#         filter_schema_cls: t.Type['RestrictedSchemaMixin'] = type,
#         variants: list,
#         field_name: str,
#         _profile: str = "shared",
#         _seen: dict | None = None,
#     ) -> tuple[dict, dict]:
#         """Build restricted schema for this type pattern.

#         Args:
#             filter_schema_cls: Class to filter variants by
#             variants: List of allowed types/schemas
#             field_name: Name of the field (for union naming)
#             profile: "shared" or "inline"

#         Returns:
#             (field_schema, defs_dict) tuple
#         """
#         raise NotImplementedError


# class GenericFieldType(BaseFieldTypeDescriptor):
#     """Represents a generic container type with element type constraints.

#     Delegates schema generation to the parameterized container type.

#     Examples:
#         GenericFieldType(ModuleList, Task) → ModuleList[Task]
#         GenericFieldType(ModuleList, Task1, Task2) → ModuleList[Task1 | Task2]
#         GenericFieldType(ModuleDict, str, Task) → ModuleDict[str, Task]
#         GenericFieldType(ModuleDict, str, Task1, Task2) → ModuleDict[str, Task1 | Task2]
#     """

#     def __init__(self, origin: type, *typs, **metadata):
#         """Initialize GenericFieldType.

#         Args:
#             origin: Container type (ModuleList, ModuleDict, etc.)
#             *typs: Type constraints
#                    For list-like: (Task,) or (Task1, Task2, ...) for union
#                    For dict-like: (key_type, value_type1, value_type2, ...) where values form union
#             **metadata: Optional metadata (for future extensibility)
#         """
#         self.origin = origin
#         self.typs = []
#         self.single_typ = []
#         for typ in typs:
#             if isinstance(typ, list):
#                 self.typs.append(typ)
#                 self.single_typ.append(False)
#             else:
#                 self.typs.append([typ])
#                 self.single_typ.append(True)

#         self.metadata = metadata

#     @classmethod
#     def from_annotation(cls, annotation) -> 'GenericFieldType':
#         """Build GenericFieldType from annotation like ModuleList[Task | State].

#         Recursively processes nested generics and unions to build the internal
#         representation with consistent list-based storage.

#         Args:
#             annotation: Generic type annotation (e.g., ModuleList[Task | State])

#         Returns:
#             GenericFieldType instance with normalized internal storage

#         Examples:
#             from_annotation(ModuleList[Task])
#             → GenericFieldType(ModuleList, [Task])

#             from_annotation(ModuleList[Task | State])
#             → GenericFieldType(ModuleList, [Task, State])

#             from_annotation(ModuleDict[str, Task | State])
#             → GenericFieldType(ModuleDict, [str], [Task, State])

#             from_annotation(ModuleList[ModuleList[Task]])
#             → GenericFieldType(ModuleList, [GenericFieldType(ModuleList, [Task])])
#         """
#         origin, type_args = extract_generic_parts(annotation)

#         if origin is None:
#             raise ValueError(f"Expected generic type, got {annotation}")

#         # Process each type argument position
#         positions = []
#         for type_arg in type_args:
#             # Flatten union at this position
#             flattened = flatten_annotation(type_arg)

#             # Process each alternative in this position
#             processed = []
#             for item in flattened:
#                 if item is None:
#                     processed.append(None)
#                 elif is_generic_annotation(item):
#                     # Nested generic - recurse
#                     nested = cls.from_annotation(item)
#                     processed.append(nested)
#                 else:
#                     # Concrete type
#                     processed.append(item)

#             positions.append(processed)

#         # Build GenericFieldType with positions as lists
#         return cls(origin, *positions)

#     def get_parameterized_type(self):
#         """Create the parameterized type like ModuleList[Task] or ModuleDict[str, Task]."""
#         idx = []
#         for typ, single_typ in zip(self.typs, self.single_typ):
#             if single_typ:
#                 if isinstance(typ[0], BaseFieldTypeDescriptor):
#                     idx.append(typ[0].get_parameterized_type())
#                 else:
#                     if isinstance(typ[0], type) and issubclass(typ[0], BaseModule):
#                         idx.append(typ[0].schema_model())
#                     else:
#                         idx.append(typ[0])
#             else:
#                 # do the same as for the single_typ
#                 cur_typ = []
#                 for typ_i in typ:
#                     if isinstance(typ_i, BaseFieldTypeDescriptor):
#                         cur_typ.append(typ_i.get_parameterized_type())
#                     else:
#                         if isinstance(typ_i, type) and issubclass(typ_i, BaseModule):
#                             cur_typ.append(typ_i.schema_model())
#                         else:
#                             cur_typ.append(typ_i)

#                 value_union = t.Union[tuple(cur_typ)]
#                 idx.append(value_union)

#         # Use Spec type for the origin if it's a BaseModule (e.g., ModuleDict -> ModuleDictSpec)
#         origin = self.origin
#         if isinstance(origin, type) and issubclass(origin, BaseModule):
#             origin = origin.schema_model()

#         return origin[tuple(idx)]

#     def schema_model(self) -> t.Type[BaseSpec]:
#         """Get the Pydantic schema model by delegating to the parameterized container type."""
#         parameterized = self.get_parameterized_type()
#         return parameterized.schema_model()

#     def schema(self) -> dict:
#         """Generate JSON schema by delegating to the parameterized container type."""
#         parameterized = self.get_parameterized_type()
#         return parameterized.schema()

#     def restricted_schema(
#         self,
#         *,
#         filter_schema_cls: t.Type['RestrictedSchemaMixin'] = type,
#         variants: list,
#         field_name: str,
#         _profile: str = "shared",
#         _seen: dict | None = None,
#     ) -> tuple[dict, dict]:
#         """Generate restricted schema by delegating to the parameterized container type."""

#         schemas = []
#         defs = []
#         for typ in self.typs:

#             cur_schemas = []
#             cur_defs = []
#             for typ_i in typ:

#                 if isinstance(typ_i, BaseFieldTypeDescriptor):
#                     schema, def_ = typ_i.restricted_schema(
#                         filter_schema_cls=filter_schema_cls,
#                         variants=variants,
#                         field_name=field_name,
#                         _profile=_profile,
#                         _seen=_seen,
#                     )
#                 elif isinstance(typ_i, type) and issubclass(typ_i, filter_schema_cls):

#                     schema = typ_i.restricted_schema(
#                         filter_schema_cls=filter_schema_cls,
#                         variants=variants,
#                         field_name=field_name,
#                         _profile=_profile,
#                         _seen=_seen,
#                     )
#                     def_ = {}
#                 elif isinstance(typ_i, type) and issubclass(typ_i, BaseModule):
#                     schema = typ_i.schema()
#                     def_ = {}
#                 else:
#                     # Non-module type (e.g., str, int for dict keys)
#                     # Use a simple type schema
#                     if typ_i == str:
#                         schema = {"type": "string"}
#                     elif typ_i == int:
#                         schema = {"type": "integer"}
#                     else:
#                         schema = {"type": "null"}
#                     def_ = {}

#                 cur_schemas.append(schema)
#                 cur_defs.append(def_)

#             schemas.append(cur_schemas)
#             defs.append(cur_defs)

#         # Merge all defs
#         merged_defs = {}
#         for pos_defs in defs:
#             for def_dict in pos_defs:
#                 merged_defs.update(def_dict)

#         # Combine schemas for each position (combining multiple types with anyOf if needed)
#         combined_schemas = []
#         for cur_schemas in schemas:
#             if len(cur_schemas) == 1:
#                 combined_schemas.append(cur_schemas[0])
#             elif len(cur_schemas) > 1:
#                 combined_schemas.append({"anyOf": cur_schemas})
#             else:
#                 combined_schemas.append({"type": "null"})

#         # Return regular schema with merged defs
#         # The schema structure is handled by get_parameterized_type()
#         return (self.schema(), merged_defs)

#     def __repr__(self):
#         typs_str = ', '.join(repr(typ) for typ in self.typs)
#         origin_name = getattr(self.origin, '__name__', str(self.origin))
#         return f"GenericFieldType({origin_name}, {typs_str})"

#     def __eq__(self, other):
#         if not isinstance(other, GenericFieldType):
#             return False
#         return self.origin == other.origin and self.typs == other.typs


# class BaseFieldDescriptor(RestrictedSchemaMixin):
#     """Base descriptor for module fields with restricted schema support."""

#     # TODO: IF default = UNDEFINED, then field is required
#     def __init__(self, typ=UNDEFINED, default=UNDEFINED, default_factory=UNDEFINED):
        
#         self.default = default
#         self.default_factory = default_factory
#         self._name = None
#         self._owner = None
#         self._types = None
#         self.single_typ = []
#         if isinstance(typ, list):
#             self.typ = typ
#             self.single_typ = False
#         else:
#             self.typ = [typ]
#             self.single_typ = True

#     def __set_name__(self, owner, name):
#         """Called when descriptor is assigned to class attribute."""
#         self._name = name
#         self._owner = owner
#         # Use get_type_hints to resolve string annotations (from __future__ import annotations)
#         try:
#             type_hints = t.get_type_hints(owner)
#             annotation = type_hints.get(name)
#         except Exception:
#             # Fall back to __annotations__ if get_type_hints fails
#             annotation = owner.__annotations__.get(name)
#         self.validate_annotation(annotation)

#     def __get__(self, obj, objtype=None):
#         """Get field value from instance."""
#         if obj is None:
#             return self
#         return obj.__dict__.get(self._name)

#     def __set__(self, obj, value):
#         """Set field value on instance."""
#         obj.__dict__[self._name] = value

#     def get_default(self):
#         """Get the default value for this field.

#         Returns the result of default_factory() if set, otherwise default value.
#         """
#         if self.default_factory is not UNDEFINED:
#             items = self.default_factory()

#             # Check if typ is a BaseFieldTypeDescriptor and wrap items
#             # if isinstance(self.typ, BaseFieldTypeDescriptor):
#             #     return self.typ.wrap_items(items)

#             # # Check if typ is a list with BaseFieldTypeDescriptor as first item
#             # if isinstance(self.typ, list) and len(self.typ) > 0:
#             #     if isinstance(self.typ[0], BaseFieldTypeDescriptor):
#             #         return self.typ[0].wrap_items(items)

#             return items
#         elif self.default is not UNDEFINED:
#             return self.default
#         else:
#             raise TypeError(f"No default value for field")

#     @abstractmethod
#     def schema(self) -> dict:
#         """Generate JSON schema for this field.

#         Delegates to the typ descriptor if it's a BaseFieldTypeDescriptor.
#         Otherwise, returns an empty schema (to be filled in by Pydantic).
#         """
#         pass

#     @abstractmethod
#     def restricted_schema(
#         self,
#         *,
#         filter_schema_cls: t.Type['RestrictedSchemaMixin'] = type,
#         variants: list | None = None,
#         _profile: str = "shared",
#         _seen: dict | None = None,
#         **kwargs
#     ) -> dict:
#         """Generate restricted schema for this field.

#         Delegates to the typ descriptor if it's a BaseFieldTypeDescriptor.
#         Otherwise, raises NotImplementedError (to be handled by subclasses).
#         """
#         pass

#     def schema_model(self) -> t.Type[BaseSpec]:
#         """Get the Pydantic schema model for this field.

#         Delegates to the typ descriptor if it's a BaseFieldTypeDescriptor.
#         Otherwise, raises NotImplementedError (to be handled by subclasses).
#         """
#         pass

#     def validate_annotation(self, annotation) -> None:
#         """Validate annotation and populate self.typ if not already set.

#         Extracts type information from annotations, handling:
#         - Union types and Optional
#         - Generic containers (ModuleList[...], ModuleDict[...])
#         - Nested generics (ModuleList[ModuleList[Task]])
#         - Concrete BaseModule types

#         Args:
#             annotation: Type annotation from class definition

#         Raises:
#             RuntimeError: If no annotation and no explicit typ parameter
#             TypeError: If annotation contains non-BaseModule types
#         """
#         # If typ not provided, extract from annotation
#         if self.typ == [UNDEFINED]:
#             if annotation is None:
#                 raise RuntimeError(
#                     f"Field '{self._name}' has modfield() but no annotation "
#                     f"and no explicit typ parameter"
#                 )

#             # Flatten and process annotation
#             flattened = flatten_annotation(annotation)
#             result = []
#             for item in flattened:
#                 if item is None:
#                     result.append(None)
#                 elif is_generic_annotation(item):
#                     result.append(GenericFieldType.from_annotation(item))
#                 else:
#                     result.append(item)

#             self.typ = result

#         # Validate all types in self.typ
#         for typ_item in self.typ:
#             if typ_item is None:
#                 continue  # None is OK for Optional
#             if isinstance(typ_item, GenericFieldType):
#                 continue  # GenericFieldType is OK
#             if isinstance(typ_item, type) and issubclass(typ_item, BaseModule):
#                 continue  # BaseModule subclass is OK

#             # Invalid type
#             raise TypeError(
#                 f"Field '{self._name}' must be a BaseModule subclass, got {typ_item}"
#             )

#     def get_spec_annotation(self) -> type:
#         """Convert runtime types to Spec types for schema building.

#         Converts BaseModule types to their corresponding Spec models, and handles
#         GenericFieldType instances by getting their parameterized types.

#         Returns:
#             Type annotation suitable for schema generation
#         """
#         if self.typ is UNDEFINED or not self.typ:
#             return type(None)

#         # Convert each type to Spec
#         spec_types = []
#         for typ in self.typ:
#             if typ is None:
#                 # Skip None in spec annotation (handled separately in schema)
#                 continue
#             elif isinstance(typ, GenericFieldType):
#                 # GenericFieldType provides parameterized type
#                 spec_types.append(typ.get_parameterized_type())
#             elif isinstance(typ, type) and issubclass(typ, BaseModule):
#                 # Convert BaseModule to its Spec model
#                 spec_types.append(typ.schema_model())
#             else:
#                 # Other types (primitives, etc.)
#                 spec_types.append(typ)

#         # Return single type or Union
#         if len(spec_types) == 0:
#             return type(None)
#         elif len(spec_types) == 1:
#             return spec_types[0]
#         else:
#             return t.Union[tuple(spec_types)]

#     # def validate_annotation(self, annotation) -> None:
#     #     """Validate annotation and extract types into self._types."""
#     #     if annotation is None and self.typ is UNDEFINED:
#     #         raise RuntimeError(
#     #             f"Field '{self._name}' has modfield() but no annotation "
#     #             f"and no explicit typ parameter"
#     #         )

#     #     # Extract types from annotation or typ parameter
#     #     if self.typ is not UNDEFINED:
#     #         if isinstance(self.typ, list):
#     #             # Legacy or contains type descriptors: typ=[GenericFieldType(...), Type2, ...]
#     #             extracted = self.typ
#     #         else:
#     #             # Single type - could be plain or already wrapped BaseFieldTypeDescriptor
#     #             extracted = [self.typ]
#     #     else:
#     #         # Extract types from annotation
#     #         extracted = self._extract_types_from_annotation(annotation)

#     #     # Process extracted types: check for BaseFieldTypeDescriptor instances
#     #     processed = []
#     #     for item in extracted:
#     #         if isinstance(item, BaseFieldTypeDescriptor):
#     #             # Extract the actual types from the descriptor
#     #             processed.extend(item.extract_types())
#     #         else:
#     #             processed.append(item)

#     #     # Compare or set
#     #     if self._types is not None:
#     #         if self._types != processed:
#     #             raise TypeError(
#     #                 f"Field '{self._name}' type mismatch: "
#     #                 f"annotation {processed} != typ parameter {self._types}"
#     #             )
#     #     else:
#     #         self._types = processed

#     #     # Validate that all types (except None) are BaseModule subclasses
#     #     for typ in self._types:
#     #         if typ is None:
#     #             continue  # None is allowed for Optional types
#     #         if not (isinstance(typ, type) and issubclass(typ, BaseModule)):
#     #             raise TypeError(
#     #                 f"modfield() type must be a BaseModule subclass, got {typ}"
#     #             )

#     # def _extract_types_from_annotation(self, annotation) -> list:
#     #     """Extract list of types from annotation.

#     #     For ModFieldDescriptor:
#     #         Union[A, B] -> [A, B]
#     #         Optional[A] -> [A, None]
#     #         A -> [A]

#     #     Subclasses may override to handle ModuleList[T], ModuleDict[K,V], etc.
#     #     """
#     #     origin = t.get_origin(annotation)

#     #     # Handle Union (both typing.Union and types.UnionType for PEP 604)
#     #     if origin in (t.Union, types.UnionType):
#     #         return list(t.get_args(annotation))

#     #     # Single type
#     #     return [annotation]

#     # def get_types(self) -> list:
#     #     """Get types as list."""
#     #     return self._types

#     # def _to_spec_type(self, typ: type) -> type:
#     #     """Convert a single type to its spec equivalent."""
#     #     if typ is None:
#     #         return None

#     #     if isinstance(typ, type) and issubclass(typ, BaseModule):
#     #         return typ.schema_model()

#     #     return typ

#     # def get_spec_annotation(self) -> type:
#     #     """Convert types to spec annotation for schema building."""
#     #     # Check if typ is a BaseFieldTypeDescriptor - use its conversion
#     #     if isinstance(self.typ, BaseFieldTypeDescriptor):
#     #         spec_type = self.typ.to_spec_annotation(self._to_spec_type)
#     #         has_none = None in self._types
#     #         return t.Optional[spec_type] if has_none else spec_type

#     #     # Check if typ is list with BaseFieldTypeDescriptor as first item
#     #     if isinstance(self.typ, list) and len(self.typ) > 0:
#     #         if isinstance(self.typ[0], BaseFieldTypeDescriptor):
#     #             spec_type = self.typ[0].to_spec_annotation(self._to_spec_type)
#     #             has_none = None in self._types
#     #             return t.Optional[spec_type] if has_none else spec_type

#     #     # Default: convert each type
#     #     spec_types = [self._to_spec_type(t) for t in self._types if t is not None]
#     #     has_none = None in self._types

#     #     if len(spec_types) == 0:
#     #         return type(None) if has_none else None
#     #     elif len(spec_types) == 1:
#     #         return t.Optional[spec_types[0]] if has_none else spec_types[0]
#     #     else:
#     #         union = t.Union[tuple(spec_types)]
#     #         return t.Optional[union] if has_none else union

#     # @abstractmethod
#     # def restricted_schema(
#     #     self,
#     #     *,
#     #     filter_schema_cls: t.Type['RestrictedSchemaMixin'] = type,
#     #     variants: list | None = None,
#     #     _profile: str = "shared",
#     #     _seen: dict | None = None,
#     #     **kwargs
#     # ) -> tuple[dict, dict]:
#     #     """Generate restricted schema for this field.

#     #     Returns:
#     #         (field_schema, defs_dict) tuple
#     #     """
#     #     raise NotImplementedError
    
#     # @property
#     # def required(self) -> bool:
#     #     return self.default is UNDEFINED


# @dataclass
# class Checkpoint(Generic[V]):
#     """Checkpoint for BaseModle objects, containing spec and state_dict."""
#     spec: BaseSpec = Field(
#         description="Specification of the object, including its kind and id."
#     )
#     state_dict: Dict[str, Any] = Field(
#         description="State dictionary containing parameters and states."
#     )

#     def save(self, path: str):
#         """Save the checkpoint to a file."""
#         with open(path, 'w') as f:
#             spec_data = self.spec.model_dump()
#             data = {
#                 "spec": spec_data,
#                 "state_dict": self.state_dict

#             }
#             f.write(json.dumps(data, indent=2))
#             # f.write(self.model_dump_json(indent=2))
    
#     @classmethod
#     def load(cls, path: str) -> "Checkpoint":
#         """Load a checkpoint from a file."""
#         with open(path, 'r') as f:
#             data = f.read()
#         data = json.loads(data)

#         load_cls = mod_registry[data['spec']['kind']]

#         spec = load_cls.obj.__spec__.model_validate(data['spec'])
#         state_dict = data['state_dict']
#         return cls(spec=spec, state_dict=state_dict)

#     @classmethod  
#     def load_module(cls, path: str, ctx: Optional[dict] = None) -> V:
#         """Reconstruct the BaseModule from the checkpoint."""
#         if ctx is None:
#             ctx = {}
#         obj = cls.load(path)
#         module_cls = obj.spec.load_cls()
#         module = module_cls.from_spec(obj.spec, ctx=ctx)
#         module.load_state_dict(obj.state_dict)
#         return module
    
#     @classmethod
#     def save_module(self, module: BaseModule, path: str):
#         """Save the BaseModule as a checkpoint."""
#         spec = module.spec(to_dict=False)
#         state_dict = module.state_dict(
#             recurse=True, train=True, runtime=True
#         )
#         checkpoint = Checkpoint(
#             spec=spec,
#             state_dict=state_dict
#         )
#         checkpoint.save(path)


# Assumes these exist in your package:
# - BaseModule, BaseSpec, registry (kind -> {obj: BaseModule subclass})
# - TypeAdapter from pydantic.v2

# from __future__ import annotations

# import typing as t
# from abc import ABC, abstractmethod

# Assume BaseModule, BaseSpec are available and use Pydantic v2-style APIs:
# - BaseModule.schema_model() -> Type[BaseSpec]
# - BaseSpec.model_json_schema() -> dict

# from __future__ import annotations

# import typing as t
# from abc import ABC, abstractmethod

# Assume BaseModule, BaseSpec are available and use Pydantic v2-style APIs:
# - BaseModule.schema_model() -> Type[BaseSpec]
# - BaseSpec.model_json_schema() -> dict

# def filter_class_variants(
#     target: t.Type | list[t.Type],
#     variants: t.Iterable[t.Any],
#     registry_instance: 'Registry' = None
# ) -> t.Iterator[t.Any]:
#     """
#     Filter variants to only those that are subclasses of target class(es).

#     Args:
#         target: Single class or list of classes to filter by
#         variants: Iterable of variants (module classes, spec classes, instances, dicts, etc.)
#         registry_instance: Optional registry to use for lookups

#     Returns:
#         Iterator of variants that match the target class(es)

#     Examples:
#         >>> filter_class_variants(Leaf, [Action(), Sequence(), Condition()])
#         # Returns only Action() and Condition() (both are Leaf subclasses)

#         >>> filter_class_variants([Condition, Action], task_list)
#         # Returns only Conditions and Actions from task_list
#     """

#     # Normalize target to a tuple
#     targets = (target,) if not isinstance(target, list) else tuple(target)

#     def matches(variant):
#         mod_cls = lookup_module_class(variant, registry_instance)
#         return mod_cls is not None and any(issubclass(mod_cls, t) for t in targets)

#     return itertools.filterfalse(lambda v: not matches(v), variants)


# def lookup_module_class(variant: t.Any, registry_instance: 'Registry' = None) -> t.Type['Module'] | None:
#     """
#     Look up the BaseModule class for a given variant.

#     This utility accepts multiple formats (module class, spec class, spec instance, schema dict, or string name)
#     and returns the corresponding BaseModule class by looking it up in the registry.

#     Useful for filtering variants by type (e.g., only accepting Condition or Leaf tasks).

#     Args:
#         variant: Can be:
#             - BaseModule class (e.g., ActionA) - returns itself
#             - BaseSpec class (e.g., ActionASpec) - looks up ActionA
#             - BaseSpec instance (e.g., ActionASpec()) - looks up ActionA
#             - Schema dict (e.g., {"title": "ActionASpec"}) - looks up ActionA
#             - String name (e.g., "ActionA" or "ActionASpec") - looks up ActionA
#         registry_instance: Optional registry instance. If None, uses global registry.

#     Returns:
#         The BaseModule class, or None if not found

#     Examples:
#         >>> lookup_module_class(ActionA)  # Already a module class
#         <class 'ActionA'>

#         >>> lookup_module_class(ActionASpec)  # Spec class
#         <class 'ActionA'>

#         >>> lookup_module_class("ActionA")  # String name
#         <class 'ActionA'>

#         >>> lookup_module_class({"title": "ActionASpec"})  # Schema dict
#         <class 'ActionA'>
#     """
#     if registry_instance is None:
#         registry_instance = mod_registry

#     # String name - look up directly
#     if isinstance(variant, str):
#         module_name = variant.replace("Spec", "")
#         try:
#             entry = registry_instance[module_name]
#             return entry.obj if entry else None
#         except KeyError:
#             return None

#     # Already a BaseModule class
#     if isinstance(variant, type) and issubclass(variant, Module):
#         return variant

#     # Spec class - look up in registry
#     if isinstance(variant, type) and issubclass(variant, Module):
#         # Get just the class name without module path
#         spec_name = variant.__name__.rsplit(".", 1)[-1]
#         module_name = spec_name.replace("Spec", "")
#         try:
#             entry = registry_instance[module_name]
#             return entry.obj if entry else None
#         except KeyError:
#             return None

#     # Spec instance - look up in registry
#     if not isinstance(variant, dict) and hasattr(variant, "__class__") and isinstance(variant.__class__, type) and issubclass(variant.__class__, Module):
#         # Get just the class name without module path
#         spec_name = variant.__class__.__name__.rsplit(".", 1)[-1]
#         module_name = spec_name.replace("Spec", "")
#         try:
#             entry = registry_instance[module_name]
#             return entry.obj if entry else None
#         except KeyError:
#             return None

#     # Schema dict - extract name and look up
#     if isinstance(variant, dict):
#         if "title" in variant and isinstance(variant["title"], str):
#             spec_name = variant["title"].strip()
#             # Remove module path if present
#             spec_name = spec_name.rsplit(".", 1)[-1]
#         elif "$id" in variant and isinstance(variant["$id"], str):
#             _id = variant["$id"].strip()
#             spec_name = _id.rstrip("#").rsplit("/", 1)[-1]
#         else:
#             return None

#         module_name = spec_name.replace("Spec", "")
#         try:
#             entry = registry_instance[module_name]
#             return entry.obj if entry else None
#         except KeyError:
#             return None

#     return None


# @dataclass
# class Checkpoint(Generic[V]):
#     """Checkpoint for BaseModle objects, containing spec and state_dict."""
#     spec: BaseSpec = Field(
#         description="Specification of the object, including its kind and id."
#     )
#     state_dict: Dict[str, Any] = Field(
#         description="State dictionary containing parameters and states."
#     )

#     def save(self, path: str):
#         """Save the checkpoint to a file."""
#         with open(path, 'w') as f:
#             spec_data = self.spec.model_dump()
#             data = {
#                 "spec": spec_data,
#                 "state_dict": self.state_dict

#             }
#             f.write(json.dumps(data, indent=2))
#             # f.write(self.model_dump_json(indent=2))
    
#     @classmethod
#     def load(cls, path: str) -> "Checkpoint":
#         """Load a checkpoint from a file."""
#         with open(path, 'r') as f:
#             data = f.read()
#         data = json.loads(data)

#         load_cls = mod_registry[data['spec']['kind']]

#         spec = load_cls.obj.__spec__.model_validate(data['spec'])
#         state_dict = data['state_dict']
#         return cls(spec=spec, state_dict=state_dict)

#     @classmethod  
#     def load_module(cls, path: str, ctx: Optional[dict] = None) -> V:
#         """Reconstruct the BaseModule from the checkpoint."""
#         if ctx is None:
#             ctx = {}
#         obj = cls.load(path)
#         module_cls = obj.spec.load_cls()
#         module = module_cls.from_spec(obj.spec, ctx=ctx)
#         module.load_state_dict(obj.state_dict)
#         return module
    
#     @classmethod
#     def save_module(self, module: BaseModule, path: str):
#         """Save the BaseModule as a checkpoint."""
#         spec = module.spec(to_dict=False)
#         state_dict = module.state_dict(
#             recurse=True, train=True, runtime=True
#         )
#         checkpoint = Checkpoint(
#             spec=spec,
#             state_dict=state_dict
#         )
#         checkpoint.save(path)


# Assumes these exist in your package:
# - BaseModule, BaseSpec, registry (kind -> {obj: BaseModule subclass})
# - TypeAdapter from pydantic.v2

# from __future__ import annotations

# import typing as t
# from abc import ABC, abstractmethod

# Assume BaseModule, BaseSpec are available and use Pydantic v2-style APIs:
# - BaseModule.schema_model() -> Type[BaseSpec]
# - BaseSpec.model_json_schema() -> dict

# from __future__ import annotations

# import typing as t
# from abc import ABC, abstractmethod

# Assume BaseModule, BaseSpec are available and use Pydantic v2-style APIs:
# - BaseModule.schema_model() -> Type[BaseSpec]
# - BaseSpec.model_json_schema() -> dict


# def filter_class_variants(
#     target: t.Type | list[t.Type],
#     variants: t.Iterable[t.Any],
#     registry_instance: 'Registry' = None
# ) -> t.Iterator[t.Any]:
#     """
#     Filter variants to only those that are subclasses of target class(es).

#     Args:
#         target: Single class or list of classes to filter by
#         variants: Iterable of variants (module classes, spec classes, instances, dicts, etc.)
#         registry_instance: Optional registry to use for lookups

#     Returns:
#         Iterator of variants that match the target class(es)

#     Examples:
#         >>> filter_class_variants(Leaf, [Action(), Sequence(), Condition()])
#         # Returns only Action() and Condition() (both are Leaf subclasses)

#         >>> filter_class_variants([Condition, Action], task_list)
#         # Returns only Conditions and Actions from task_list
#     """

#     # Normalize target to a tuple
#     targets = (target,) if not isinstance(target, list) else tuple(target)

#     def matches(variant):
#         mod_cls = lookup_module_class(variant, registry_instance)
#         return mod_cls is not None and any(issubclass(mod_cls, t) for t in targets)

#     return itertools.filterfalse(lambda v: not matches(v), variants)


# def lookup_module_class(variant: t.Any, registry_instance: 'Registry' = None) -> t.Type['Module'] | None:
#     """
#     Look up the BaseModule class for a given variant.

#     This utility accepts multiple formats (module class, spec class, spec instance, schema dict, or string name)
#     and returns the corresponding BaseModule class by looking it up in the registry.

#     Useful for filtering variants by type (e.g., only accepting Condition or Leaf tasks).

#     Args:
#         variant: Can be:
#             - BaseModule class (e.g., ActionA) - returns itself
#             - BaseSpec class (e.g., ActionASpec) - looks up ActionA
#             - BaseSpec instance (e.g., ActionASpec()) - looks up ActionA
#             - Schema dict (e.g., {"title": "ActionASpec"}) - looks up ActionA
#             - String name (e.g., "ActionA" or "ActionASpec") - looks up ActionA
#         registry_instance: Optional registry instance. If None, uses global registry.

#     Returns:
#         The BaseModule class, or None if not found

#     Examples:
#         >>> lookup_module_class(ActionA)  # Already a module class
#         <class 'ActionA'>

#         >>> lookup_module_class(ActionASpec)  # Spec class
#         <class 'ActionA'>

#         >>> lookup_module_class("ActionA")  # String name
#         <class 'ActionA'>

#         >>> lookup_module_class({"title": "ActionASpec"})  # Schema dict
#         <class 'ActionA'>
#     """
#     if registry_instance is None:
#         registry_instance = mod_registry

#     # String name - look up directly
#     if isinstance(variant, str):
#         module_name = variant.replace("Spec", "")
#         try:
#             entry = registry_instance[module_name]
#             return entry.obj if entry else None
#         except KeyError:
#             return None

#     # Already a BaseModule class
#     if isinstance(variant, type) and issubclass(variant, Module):
#         return variant

#     # Spec class - look up in registry
#     if isinstance(variant, type) and issubclass(variant, Module):
#         # Get just the class name without module path
#         spec_name = variant.__name__.rsplit(".", 1)[-1]
#         module_name = spec_name.replace("Spec", "")
#         try:
#             entry = registry_instance[module_name]
#             return entry.obj if entry else None
#         except KeyError:
#             return None

#     # Spec instance - look up in registry
#     if not isinstance(variant, dict) and hasattr(variant, "__class__") and isinstance(variant.__class__, type) and issubclass(variant.__class__, Module):
#         # Get just the class name without module path
#         spec_name = variant.__class__.__name__.rsplit(".", 1)[-1]
#         module_name = spec_name.replace("Spec", "")
#         try:
#             entry = registry_instance[module_name]
#             return entry.obj if entry else None
#         except KeyError:
#             return None

#     # Schema dict - extract name and look up
#     if isinstance(variant, dict):
#         if "title" in variant and isinstance(variant["title"], str):
#             spec_name = variant["title"].strip()
#             # Remove module path if present
#             spec_name = spec_name.rsplit(".", 1)[-1]
#         elif "$id" in variant and isinstance(variant["$id"], str):
#             _id = variant["$id"].strip()
#             spec_name = _id.rstrip("#").rsplit("/", 1)[-1]
#         else:
#             return None

#         module_name = spec_name.replace("Spec", "")
#         try:
#             entry = registry_instance[module_name]
#             return entry.obj if entry else None
#         except KeyError:
#             return None

#     return None
