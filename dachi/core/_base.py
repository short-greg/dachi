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
from dachi.utils import get_all_private_attr_annotations
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

    def update_data_hook(self, old_val: J | None, val: J | None) -> J | None:
        # override for any hooks / logic here for data
        # e.g. log, trigger dirty flag, coerce type
        for callback in self._callbacks:
            callback(old_val, val)

    def __hash__(self):
        return id(self) 
    
    def spec_schema(self) -> dict:

        return self.model_json_schema()

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
        result_data = self.data + (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __sub__(self, other):
        result_data = self.data - (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __mul__(self, other):
        result_data = self.data * (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __truediv__(self, other):
        result_data = self.data / (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __floordiv__(self, other):
        result_data = self.data // (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __mod__(self, other):
        result_data = self.data % (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __pow__(self, other):
        result_data = self.data ** (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    # Reverse arithmetic dunder methods
    def __radd__(self, other):
        result_data = (other.data if isinstance(other, ShareableItem) else other) + self.data
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __rsub__(self, other):
        result_data = (other.data if isinstance(other, ShareableItem) else other) - self.data
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __rmul__(self, other):
        result_data = (other.data if isinstance(other, ShareableItem) else other) * self.data
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __rtruediv__(self, other):
        result_data = (other.data if isinstance(other, ShareableItem) else other) / self.data
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __rfloordiv__(self, other):
        result_data = (other.data if isinstance(other, ShareableItem) else other) // self.data
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __rmod__(self, other):
        result_data = (other.data if isinstance(other, ShareableItem) else other) % self.data
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __rpow__(self, other):
        result_data = (other.data if isinstance(other, ShareableItem) else other) ** self.data
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    # In-place arithmetic dunder methods
    def __iadd__(self, other):
        result_data = self.data + (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __isub__(self, other):
        result_data = self.data - (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __imul__(self, other):
        result_data = self.data * (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __itruediv__(self, other):
        result_data = self.data / (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __ifloordiv__(self, other):
        result_data = self.data // (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __imod__(self, other):
        result_data = self.data % (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __ipow__(self, other):
        result_data = self.data ** (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __lt__(self, other):
        return self.data < (other.data if isinstance(other, ShareableItem) else other)

    def __le__(self, other):
        return self.data <= (other.data if isinstance(other, ShareableItem) else other)

    def __gt__(self, other):
        return self.data > (other.data if isinstance(other, ShareableItem) else other)

    def __ge__(self, other):
        return self.data >= (other.data if isinstance(other, ShareableItem) else other)

    # def _get_expected_type(self):
    #     """Resolve expected type from the immediate generic base (Param, State, Shared)."""
    #     cls = self.__class__
    #     for base in getattr(cls, "__orig_bases__", []):
    #         origin = getattr(base, "__origin__", None)
    #         if origin in {Param, Attr, Shared}:
    #             return t.get_args(base)[0]
    #     return None

    def load(self, data):
        """
        Rebuild a ShareableItem from a spec or dict.
        """
        print('Loading ', data, self.__class__)
        loaded = self.__class__.model_validate(data)
        print('Loaded: ', self.data.__class__)
        self.data = loaded.data

    def dump(self) -> dict:
        """
        Dump the ShareableItem to a dictionary.
        """
        # if isinstance(self.data, BaseModel):
        #     # If data is a BaseModel, use its model_dump method
        #     data = self.data.model_dump()
        # else:
        #     data = self.data

        return self.model_dump()

    def __repr__(self):
        
        return f"{self.__class__.__name__}(data={repr(self.data)})"
    
    def __str__(self):
        return str(self.data)


class Param(ShareableItem[J]):
    """Trainable parameter; ``training`` may be toggled to freeze it."""

    _fixed: bool = False
    
    def set(self, data):
        if self._fixed:
            raise RuntimeError(
                'Cannot set parameter that is fixed.'
            )
        data = super().set(data)
        return data

    def is_fixed(self) -> bool:
        """
        Check if the parameter is fixed.
        """
        return self._fixed
    
    def fix(self):
        """
        Fix the parameter, making it unmodifiable.
        """
        self._fixed = True

    def unfix(self):
        """
        Unfix the parameter, making it modifiable.
        """
        self._fixed = False   



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
    def parameters(
        self,
        *,
        recurse: bool = True,
        _seen: t.Optional[set[int]] = None,
        with_annotations: bool = False,
    ) -> t.Iterator[Param | tuple[Param, t.Any]]:
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


class StateType(Enum):

    MODULE: str = auto()
    RUNTIME: str = auto()
    PARAM: str = auto()


class ObjInit(t.Generic[T]):

    def __init__(
        self, fn: t.Callable[['Module'], t.Any], 
        base_cls: t.Type
    ):
        self.fn = fn
        self.base_cls = base_cls

    def __call__(self, module: 'Module', anno=None):
        # anno = self.anno or anno
        if anno is None:
            return self.base_cls(data=self.fn(module))
        return self.base_cls[anno](data=self.fn(module))


class FuncInit(t.Generic[T]):
    
    def __init__(
        self, fn: t.Callable[[], t.Any], 
        base_cls: t.Type
    ):
        self.fn = fn
        self.base_cls = base_cls

    def __call__(self, anno=None):
        # anno = self.anno or anno
        if anno is None:
            return self.base_cls(data=self.fn())
        return self.base_cls[anno](data=self.fn())


def _PrivateType(
    cls: t.Type[ShareableItem],
    default=None, 
    default_factory=None, 
    instance_factory=None,
    instance_field=None
):
    """Create a PrivateAttr for ShareableItem subclass with default value."""
    if (
        instance_field is not None 
        and default is None 
        and default_factory is None 
        and instance_factory is None
    ):
        return pydantic.PrivateAttr(
            default=ObjInit(
                lambda module: getattr(module, instance_field), cls
            )
        )
    if instance_factory is not None and default is None and default_factory is None:
        return pydantic.PrivateAttr(
            default=ObjInit(
                instance_factory, cls
            )
        )
    if default is None and default_factory is not None:
        return pydantic.PrivateAttr(
            default=FuncInit(
                default_factory, cls
            )
        )
    return pydantic.PrivateAttr(
        default=FuncInit(
            lambda: default, cls
        )
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
        instance_field=instance_field
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
        instance_field=instance_field
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
        instance_field=instance_field
    )


# import sys
# from typing import Any
# from pydantic import BaseModel


# def _resolve_raw_annotation(defining_cls: type[BaseModel], raw: Any) -> Any:
#     """Best-effort resolution of a single annotation value."""
#     if raw is None:
#         return None

#     # Already a real type or typing object
#     if not isinstance(raw, str):
#         return raw

#     module_globals = vars(sys.modules[defining_cls.__module__])
#     localns = dict(vars(defining_cls))

#     try:
#         return eval(raw, module_globals, localns)
#     except Exception:
#         # NameError, SyntaxError, whatever – don't blow up, just keep the string
#         return raw


# def get_all_private_attr_annotations(model_cls: type[BaseModel]) -> dict[str, Any]:
#     """
#     Return {name: annotation_or_type} for all private attributes on this
#     Pydantic model class, including those declared on base classes.

#     - If the annotation is a real type (int, SomeModel, list[str], ...),
#       you'll get that object.
#     - If it's a string and resolvable in the defining class's namespace,
#       you'll get the resolved type.
#     - If resolution fails, you'll get the raw string.
#     """
#     private_attrs = getattr(model_cls, "__private_attributes__", {})
#     if not private_attrs:
#         return {}

#     result: dict[str, Any] = {}

#     for name in private_attrs.keys():
#         raw = None
#         defining_cls: type[BaseModel] | None = None

#         # Find the class in the MRO that actually defines the annotation
#         for cls in model_cls.__mro__:
#             anns = getattr(cls, "__annotations__", {})
#             if name in anns:
#                 raw = anns[name]
#                 defining_cls = cls
#                 break

#         if defining_cls is None:
#             # No annotation found anywhere in the MRO
#             result[name] = None
#         else:
#             result[name] = _resolve_raw_annotation(defining_cls, raw)

#     return result


class Module(pydantic.BaseModel, StorableState, Trainable):
    # Pydantic v2 style config
    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        ignored_types=(ShareableItem,),  # do not treat Param/Runtime annotations as fields
    )

    KIND: str = Field(default="Module")
    # registry: name -> StateType (PARAM / ATTR / MODULE)
    _registry: t.Dict[str, StateType] = pydantic.PrivateAttr(default_factory=dict)
    _training: Runtime[bool] = PrivateRuntime(default=True)

    @classmethod
    def __init_subclass__(cls, **kwargs):
        if cls is Module:
            super().__init_subclass__(**kwargs)
            return

        # Set constant kind: Literal["ClsName"]
        # Must modify annotation BEFORE super().__init_subclass__ so Pydantic picks it up
        cls.__annotations__["KIND"] = t.Literal[cls.__qualname__]
        if "KIND" not in cls.__dict__:
            cls.KIND = cls.__qualname__

        # for name, attr in list(cls.__dict__.items()):
        #     if not isinstance(attr, FieldInfo):
        #         continue

        #     extra = attr.json_schema_extra or {}
        #     if extra.get("is_param"):
            
        #         orig_type = cls.__annotations__.get(name, t.Any)
        #         cls.__annotations__[name] = Param[orig_type]
        #     elif extra.get("is_runtime"):
        #         orig_type = cls.__annotations__.get(name, t.Any)
        #         cls.__annotations__[name] = Runtime[orig_type]

        # Call super().__init_subclass__ AFTER all annotation modifications
        # This allows Pydantic to build the model with the correct schema
        super().__init_subclass__(**kwargs)

        # Update the model_fields to have the correct default
        if "KIND" in cls.model_fields:
            cls.model_fields["KIND"].default = cls.__qualname__

        # Auto-register the module in the global registry
        mod_registry.register()(cls)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        private_annotations = get_all_private_attr_annotations(self.__class__)

        # 2) Private attributes (ignore the registry itself)
        for name in self.__private_attributes__.keys():
            if name == "_registry":
                continue

            value = getattr(self, name)
            # annotations = get_all_private_attr_annotations(self.__class__)
            # SelfInit → compute from self, then re-read
            if isinstance(value, ObjInit) or isinstance(value, FuncInit):

                if name not in private_annotations:
                    annotation = None
                elif hasattr(private_annotations[name], "__pydantic_generic_metadata__"):
                    annotation = private_annotations[name].__pydantic_generic_metadata__['args'][0]
                else:
                    annotation = None

                computed = value(self, annotation) if isinstance(value, ObjInit) else value(annotation)
                setattr(self, name, computed)
                value = computed
                if isinstance(value, Param):
                    self._registry[name] = StateType.PARAM
                elif isinstance(value, Runtime):
                    self._registry[name] = StateType.RUNTIME
            
            # elif isinstance(value, FuncInit):
            #     print('FuncInit ', name)
            #     if name not in self.__class__.__annotations__:
            #         annotation = None
            #     elif hasattr(self.__class__.__annotations__[name], "__pydantic_generic_metadata__"):
            #         annotation = private_annotations[name].__pydantic_generic_metadata__['args'][0]
            #     else:
            #         annotation = None
            #     print('Annotation: ', annotation)
            #     computed = value(annotation)
            #     print('setting', computed)
            #     setattr(self, name, computed)
            #     print('set', getattr(self, name))
            #     value = computed
            #     if isinstance(value, Param):
            #         self._registry[name] = StateType.PARAM
            #     elif isinstance(value, Runtime):
            #         self._registry[name] = StateType.RUNTIME
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

        if id(self) in _seen:
            return

        _seen.add(id(self))
        # local params
        for name, state_type in self._registry.items():
            if state_type is not StateType.PARAM:
                continue

            param = getattr(self, name)
            if not isinstance(param, Param):
                continue

            if with_annotations:
                ann = self.__annotations__.get(name, t.Any)
                yield (param, ann)
            else:
                yield param

        # recurse into child modules
        if recurse:
            for name, mod in self.named_modules():
                yield from mod.parameters(
                    recurse=True, _seen=_seen, with_annotations=with_annotations
                )

    def modules(
        self,
        *,
        recurse: bool = True,
        f: t.Callable[['Module'], bool] | None = None,
        _seen: t.Optional[set[int]] = None,
        _skip_self: bool = False
    ):
        """Yield all Module objects."""
        if _seen is None:
            _seen = set()

        if not _skip_self and (f is None or f(self)):
            yield self
            _seen.add(id(self))
        for name, state_type in self._registry.items():
            if state_type is not StateType.MODULE:
                continue
            child = getattr(self, name)
            if id(child) in _seen:
                continue
            _seen.add(id(child))
            if f is None or f(child):
                print('Yielding child module: ', name, child)
                yield child
            if recurse:
                yield from child.modules(recurse=recurse, f=f, _seen=_seen, _skip_self=True)

    def named_modules(
        self,
        *,
        recurse: bool = True,
        prefix: str = "",
        f: t.Callable[['Module'], bool] | None = None,
        _seen: t.Optional[set[int]] = None,
        _skip_self: bool = False
    ):
        """Yield all module names and their Module objects."""
        if _seen is None:
            _seen = set()

        if not _skip_self and (f is None or f(self)):
            yield prefix.rstrip("."), self
            _seen.add(id(self))
        for name, state_type in self._registry.items():
            if state_type is not StateType.MODULE:
                continue
            child = getattr(self, name)
            child_prefix = f"{prefix}{name}."
            if id(child) in _seen:
                continue
            _seen.add(id(child))
            yield child_prefix.rstrip("."), child
            if recurse:
                yield from child.named_modules(
                    recurse=recurse, prefix=child_prefix, f=f, _seen=_seen, _skip_self=True
                )

    def named_parameters(
        self,
        *,
        recurse: bool = True,
        _seen: t.Optional[set] = None,
        prefix: str = "",
    ) -> t.Generator[tuple[str, Param], None, None]:
        """Yield all parameter names and their Param objects."""
        if _seen is None:
            _seen = set()

        for name, state_type in self._registry.items():
            if state_type is not StateType.PARAM:
                continue
            if name in _seen:
                continue
            _seen.add(name)
            param = getattr(self, name)
            if isinstance(param, Param):
                yield f"{prefix}{name}", param

        if recurse:
            for name, module in self.named_modules(recurse=True):
                if name == "":
                    child_prefix = ""
                else:
                    child_prefix = f"{name}."
                for param_name, param in module.named_parameters(
                    recurse=False, _seen=_seen, prefix=child_prefix
                ):
                    yield param_name, param

    def named_states(
        self,
        *,
        recurse: bool = True,
        _seen: t.Optional[set] = None,
        prefix: str = "",
    ):
        """Yield all state names and their Runtime objects."""
        if _seen is None:
            _seen = set()
        if id(self) in _seen:
            return
        _seen.add(id(self))
        # local states
        for name, state_type in self._registry.items():
            if state_type is not StateType.RUNTIME:
                continue
            state = getattr(self, name)
            if isinstance(state, Runtime):
                yield f"{prefix}{name}", state

        if recurse:
            for name, mod in self.named_modules(recurse=True):
                if name == "":
                    child_prefix = ""
                else:
                    child_prefix = f"{prefix}{name}."
                    yield from mod.named_states(recurse=True, _seen=_seen, prefix=child_prefix)

    def children(self):
        """Immediate child modules (non-recursive)."""
        return list(
            self.modules(recurse=False, _skip_self=True)
        )

    def named_children(self):
        """Immediate child modules (name, module) pairs."""
        return {
            name: child
            for name, child in self.named_modules(recurse=False, _skip_self=True)
        }

    def apply(
        self,
        fn: t.Callable[[t.Any], None],
        *,
        recurse: bool = True,
        include: t.Callable[[t.Any], bool] | t.Type | None = None,
        _seen: t.Optional[set[int]] = None,
    ):
        """
        Recursively apply *fn* to self and all registered objects.
        """
        if _seen is None:
            _seen = set()
        # targets: list[t.Any] = [self]
        targets: list[t.Any] = []
        for name, module in self.named_modules(recurse=recurse):
            targets.append(module)

        for obj in targets:
            if include is None:
                fn(obj)
            elif isinstance(include, type) and isinstance(obj, include):
                fn(obj)
            elif not isinstance(include, type) and include(obj):
                fn(obj)

        # for name, mod in self.modules(recurse=recurse, ):
        #     if state_type is not StateType.MODULE:
        #         continue
        #     child = getattr(self, name)
        #     if isinstance(child, Module):
        #         child.apply(fn, include=include, recurse=recurse, _seen=_seen)

    def train(self, mode: bool = True):
        """Recursively set Param.training for all parameters."""
        self._training.set(mode)
        for name, mod in self.named_modules(recurse=True, _skip_self=True):
            # if isinstance(mod, Module):
            mod.train(mode)
        return self

    def eval(self):
        """Alias for ``train(False)``."""
        return self.train(False)

    # TODO: figure out how to deal with Shareables
    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        # Avoid registering internal / pydantic internals
        if name.startswith("_"):
            return value

        if isinstance(value, Param):
            self._registry[name] = StateType.PARAM
        elif isinstance(value, Runtime):
            self._registry[name] = StateType.RUNTIME
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
                out[name] = obj.dump()
            elif state_type is StateType.RUNTIME and runtime and isinstance(obj, Runtime):
                out[name] = obj.dump()

        # recurse into modules
        if recurse:
            for name, child in self.named_modules(
                recurse=False, _skip_self=True
            ):
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
            elif state_type is StateType.RUNTIME and runtime:
                keys.add(name)

        if recurse:
            for name, child in self.named_modules(
                recurse=False, _skip_self=True
            ):
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
                obj.load(sd[name])
                found.add(name)
            elif state_type is StateType.RUNTIME and runtime and isinstance(obj, Runtime):
                obj.load(sd[name])
                found.add(name)

        # recurse
        if recurse:
            for name, child in self.named_modules(
                recurse=False, _skip_self=True
            ):
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
        description="The state dict for the module."
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

        load_cls = mod_registry[data['spec']['KIND']]

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
    _adapted: Param[V | None] = PrivateParam(None)
    _train_submods: bool = pydantic.PrivateAttr(default=True)
    _fixed: bool = pydantic.PrivateAttr(default=False)

    def fix(self):
        """Collapse to spec‑blob so only *adapted_param* remains trainable."""
        self._fixed = True

    def unfix(self, *, ctx: dict | None = None):
        self._fixed = False

    def parameters(self, *, recurse=True, _seen=None, with_annotations: bool=False):  # noqa: D401
        if _seen is None:
            _seen = set()
        if id(self) in _seen:
            return
        _seen.add(id(self))

        # always expose the *spec* parameter itself unless frozen
        if not self._fixed:
            if with_annotations:
                yield (self._adapted, V)
            else:
                yield self._adapted

        # inner numeric params – optional
        if recurse and self._train_submods and not self._fixed and self._adapted.data is not None:
            yield from self._adapted.data.parameters(recurse=True, _seen=_seen, with_annotations=with_annotations)

    def render(self) -> str:  # for LLM debugging
        adapted_name = self._adapted.data.__class__.__name__ if self._adapted.data else 'None'
        return f"AdaptModule(adapted={adapted_name}, fixed={self._fixed})"

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
    def adapted(self) -> V | None:
        """Get the adapted module
        """
        return self._adapted.data

    @adapted.setter
    def adapted(self, val: V):
        if self._fixed:
            raise RuntimeError("Cannot update adapted on a frozen AdaptModule")

        self._adapted.set(val)

    def state_dict(self, *, recurse=True, train=True, runtime=True) -> dict:
        """Get the state dict, including adapted sub-module."""
        out = super().state_dict(recurse=recurse, train=train, runtime=runtime)
        if self._adapted.data is not None:
            adapted_sd = self._adapted.data.state_dict(recurse=recurse, train=train, runtime=runtime)
            for k, v in adapted_sd.items():
                out[f"adapted.{k}"] = v
        return out

    def load_state_dict(self, sd, *, recurse = True, train = True, runtime = True, strict = True):
        # Separate adapted state dict from parent state dict
        adapted_sd = {k[len("adapted.") :]: v for k, v in sd.items() if k.startswith("adapted.")}
        parent_sd = {k: v for k, v in sd.items() if not k.startswith("adapted.")}

        # Load parent state dict
        
        super().load_state_dict(parent_sd, recurse=recurse, train=train, runtime=runtime, strict=strict)

        # Load adapted sub-module state dict
        if self._adapted.data is not None and adapted_sd:
            self._adapted.data.load_state_dict(adapted_sd, recurse=recurse, train=train, runtime=runtime, strict=strict)

    # @pydantic.field_validator('adapted', mode='before')
    # def validate_adapted(self, val: J | None | Param[J | None]) -> t.Set[str]:
    #     """Validate the allowed set."""

    #     if isinstance(val, Param):
    #         return val
        
    #     return Param[J](data=val)
    


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


# def RuntimeField(
#     *, default=..., default_factory=..., **kwargs
# ):
#     """Create a Field for Runtime with default value.
#     Frozen is always True for this. It can be treated as a way to initialize a variable that will be used at runtime.
#     """
#     f = Field(default, default_factory=default_factory,
#     frozen=True, **kwargs)
#     # mark this field as a "runtime"
#     meta = getattr(f, "json_schema_extra", None) or {}
#     meta["is_runtime"] = True
#     f.json_schema_extra = meta
#     return f
