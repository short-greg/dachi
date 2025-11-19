# 1st party
from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Union, Generic, Callable, Any, Dict, Optional, Union, List
from pydantic.fields import FieldInfo
import typing as t
import pydantic
from typing import Generic, Union
import inspect
import json
from enum import Enum, auto

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
    _callbacks: t.List[
        Callable[[J | None, J | None], None]
    ] = pydantic.PrivateAttr(default_factory=list)

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


# def ParamField(
#     *, default=..., frozen=True, default_factory=..., **kwargs
# ):
#     """Create a Field for Param with default value.
#     Frozen is set to True by default 
#     """
#     f = Field(default, frozen=frozen, default_factory=default_factory, **kwargs)
#     # mark this field as a "param"
#     meta = getattr(f, "json_schema_extra", None) or {}
#     meta["is_param"] = True
#     f.json_schema_extra = meta
#     return f


# def SharedField(
#     *, default=..., frozen=True, default_factory=..., **kwargs
# ):
#     """Create a Field for Param with default value.
#     Frozen is set to True by default 
#     """
#     f = Field(default, frozen=frozen, default_factory=default_factory, **kwargs)
#     # mark this field as a "param"
#     meta = getattr(f, "json_schema_extra", None) or {}
#     meta["is_param"] = True
#     f.json_schema_extra = meta
#     return f


def RuntimeField(
    *, default=..., default_factory=..., **kwargs
):
    """Create a Field for Runtime with default value.
    Frozen is always True for this. It can be treated as a way to initialize a variable that will be used at runtime.
    """
    f = Field(default, default_factory=default_factory,
    frozen=True, **kwargs)
    # mark this field as a "runtime"
    meta = getattr(f, "json_schema_extra", None) or {}
    meta["is_runtime"] = True
    f.json_schema_extra = meta
    return f


class SelfInit:

    def __init__(self, fn: t.Callable[['Module'], t.Any]):
        self.fn = fn

    def __call__(self, module: 'Module'):
        return self.fn(module)


def _PrivateType(
    cls: t.Type[ShareableItem],
    default=None, 
    default_factory=None, 
    instance_factory=None,
    instance_field=None
):
    """Create a PrivateAttr for ShareableItem subclass with default value."""
    if instance_field is not None and default is None and default_factory is None and instance_factory is None:
        return pydantic.PrivateAttr(
            default=SelfInit(
                lambda module: cls(data=getattr(module, instance_field))
            )
        )
    if instance_factory is not None and default is None and default_factory is None:
        return pydantic.PrivateAttr(
            default=SelfInit(
                instance_factory
            )
        )
    if default is None and default_factory is not None:
        return pydantic.PrivateAttr(
            default_factory=lambda: cls(data=default_factory())
        )
    return pydantic.PrivateAttr(
        default_factory=lambda: cls(data=default)
    )


def PrivateRuntime(
    default=None, 
    default_factory=None, 
    instance_factory=None,
    instance_field=None
):
    """Create a PrivateAttr for Attr with default value."""
    return _PrivateType(
        Runtime,
        default=default,
        default_factory=default_factory,
        instance_factory=instance_factory,
        instance_field=instance_field,
    )


def PrivateParam(
    default=None, 
    default_factory=None, 
    instance_factory=None,
    instance_field=None
):
    """Create a PrivateAttr for Param with default value."""
    return _PrivateType(
        Param,
        default=default,
        default_factory=default_factory,
        instance_factory=instance_factory,
        instance_field=instance_field,
    )


def PrivateShared(
    default=None, 
    default_factory=None, 
    instance_factory=None,
    instance_field=None
):
    """Create a PrivateAttr for Shared with default value."""
    return _PrivateType(
        Shared,
        default=default,
        default_factory=default_factory,
        instance_factory=instance_factory,
        instance_field=instance_field,
    )


class StateType(Enum):

    MODULE: str = auto()
    ATTR: str = auto()
    PARAM: str = auto()


class Module(pydantic.BaseModel):
    # Pydantic v2 style config
    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        ignored_types=(ShareableItem,),  # do not treat Param/Runtime annotations as fields
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
        runtime_field_names: list[str] = []
        for name, field in cls.__dict__.items():
            if isinstance(field, pydantic.fields.FieldInfo):
                extra = field.json_schema_extra or {}
                if extra.get("is_param"):
                    param_field_names.append(name)
                elif extra.get("is_runtime"):
                    runtime_field_names.append(name)

        # For each ParamField:
        #   1. Remove it from annotations (no longer a pydantic field)
        #   2. Add a PrivateAttr placeholder instead
        for name in param_field_names:
            cls.__annotations__.pop(name, None)
            private = pydantic.PrivateAttr(default=None)
            setattr(cls, name, private)
            cls.__private_attributes__[name] = private

        for name in runtime_field_names:
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


class Checkpoint(pydantic.BaseModel):
    """Checkpoint for BaseModle objects, containing spec and state_dict."""
    
    spec: t.Dict[str, t.Any] = Field(
        description="The specification for the module."
    )
    state: t.Dict[str, Dict[str, t.Any]] = Field(
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
    _adapted: V | None = PrivateParam(None)
    _train_submods: bool = pydantic.PrivateAttr(default=True)
    _fixed: bool = pydantic.PrivateAttr(default=False)

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
    def adapted(self, val: V | Param[V]):
        if self._fixed:
            raise RuntimeError("Cannot update adapted on a frozen AdaptModule")

        if not isinstance(val, Param):
            val = Param[J](data=val)
        
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

    # @pydantic.field_validator('adapted', mode='before')
    # def validate_adapted(self, val: J | None | Param[J | None]) -> t.Set[str]:
    #     """Validate the allowed set."""

    #     if isinstance(val, Param):
    #         return val
        
    #     return Param[J](data=val)
    
