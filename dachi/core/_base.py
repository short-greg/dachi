# 1st party
from abc import abstractmethod, ABC

from functools import lru_cache
from dataclasses import InitVar
from uuid import uuid4
from typing import Iterable, Union, Mapping, Generic, Callable, Any, Dict, Optional, Union, List
import typing as t
from functools import lru_cache
from typing import Generic, Iterable, Union
import inspect
import json
import copy
from dataclasses import dataclass



try:  # 3.12+
    from typing import dataclass_transform
except ImportError:  # 3.8–3.11
    from typing_extensions import dataclass_transform

# 3rd Party
from pydantic import BaseModel, Field, ConfigDict, create_model, field_validator, TypeAdapter

# Local
from dachi.utils import resolve_name


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

from typing import Generic, Union


def to_kind(cls): 
    """Convert a class to its kind."""
    
    return cls.__qualname__


class ShareableItem(t.Generic[J]):
    """Serializable leaf object with a ``data`` field."""

    def __init__(self, data: J | None=None):
        super().__init__()
        expected_type = self._get_expected_type()
        print(expected_type)
        if (
            expected_type is not None 
            and not isinstance(data, expected_type)
        ):
            raise TypeError(f"Expected data of type {expected_type}, got {type(data)}")
        self._data = data
        self._callbacks: list[Callable[[T]]] = []

    def get(self) -> J | None:

        return self._data
    
    def set(self, data: J | None):
        expected_type = self._get_expected_type()
        if (
            expected_type is not None and not isinstance(data, expected_type)
        ):
            raise TypeError(f"Expected data of type {expected_type}, got {type(data)}")
        self._data = data
        self.update_data_hook(data)

    def empty(self) -> bool:

        return self.data is None

    @property
    def data(self) -> J | None:
        """Get the data value."""
        return self._data

    @data.setter
    def data(self, value: J):
        """Set the data value and trigger update hook."""
        self.set(value)
        return value

    def update_data_hook(self, val: T) -> T:
        # override for any hooks / logic here for data
        # e.g. log, trigger dirty flag, coerce type
        for callback in self._callbacks:
            callback(val)

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
        """
        Serialise *this* ShareableItem → its spec counterpart.

        If *to_dict* is True, returns a dict; otherwise returns a BaseModel.
        """
        if isinstance(self._data, BaseModel):
            return self._data.model_json_schema()
        else:
            return {"type": type(self._data).__name__}
        
    def has_callback(self, callback: Callable[[T], None]) -> bool:
        return callback in self._callbacks

    def register_callback(self, callback: Callable[[T], None]) -> None:
        """Register a callback to be called when the data is updated."""
        self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[T], None]) -> bool:
        """Unregister a previously registered callback. 
        If callback does not exist will return False"""
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def _get_expected_type(self):
        """Resolve expected type from the immediate generic base (Param, State, Shared)."""
        cls = self.__class__
        for base in getattr(cls, "__orig_bases__", []):
            origin = getattr(base, "__origin__", None)
            if origin in {Param, Attr, Shared}:
                return t.get_args(base)[0]
        return None
    
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
    
    def __init__(self, data: J, fixed: bool=False):
        """
        Initialize a trainable parameter.

        Args:
            data (J): The initial value of the parameter.
            fixed (bool): Whether the parameter is fixed (unmodifiable).
        """

        super().__init__(data)
        self._fixed = fixed

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


class Attr(ShareableItem[J]):
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


class BaseSpec(BaseModel):
    """Base class for Specs
    Specs are automatically subclassed by BaseModule 
    to create a Spec for that Module. It can
    manually be subclassed if needed.
    """

    kind : str
    id   : str = Field(
        default_factory=lambda: str(uuid4())
    )
    style: t.Literal['structured'] = 'structured'

    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)

    @classmethod
    def class_kind(cls) -> str:
        """
        Return the constant literal value of the `kind` field that was
        set by create_model().  Raises if it hasn’t been frozen.
        """
        default = cls.model_fields['kind'].default
        if default is None:
            raise RuntimeError(f"{cls.__name__} has no fixed `kind` default")
        return default

    @classmethod
    def load_cls(cls):
        kind = cls.class_kind()
        if kind not in registry.list_entries():
            raise ValueError(f"Class kind '{kind}' not registered in registry.")
        return registry[kind].obj


def get_class_annotations(cls: type) -> dict[str, type]:
    """Safely get annotations with fallback to __annotations__"""
    try:
        hints = t.get_type_hints(cls)
    except Exception as e:
        # Log or handle if needed
        hints = {}

    raw = getattr(cls, '__annotations__', {})
    for k, v in raw.items():
        if k not in hints:
            hints[k] = v  # fallback type, maybe str or ForwardRef
    return hints


@dataclass_transform(kw_only_default=True, field_specifiers=(Field,))
class BaseModule:
    """Dataclass‑like runtime object without exec‑generated ``__init__``.
    
    A Pydantic BaseModel will be created for BaseModules
    automatically so all fields specified in the
    class header must be serializable by Pydantic to
    create the Spec.

    Use InitVar to indicate that a variable should be initialized in post_init. Other fields will automatically be included as member variables.

    """

    # populated by __init_subclass__
    __spec__: t.ClassVar[type[BaseSpec]]
    __spec_hooks__: t.ClassVar[t.List[str]] = []
    __item_fields__: t.ClassVar[list[tuple[str, t.Any, t.Any, bool]]]
    __is_initvar__: t.ClassVar[dict[str, bool]]
    training: bool = True  # True if any Module is in training mode; False when not

    def __post_init__(self):
        pass
    
    @classmethod
    def _spec_model_name(cls: type) -> str:
        """
        Return a deterministic, collision-free name for the spec model.

        Example:
            pkg_a.models.Leaf            ->  pkg_a_models_LeafSpec
            pkg_b.sub.Leaf.Inner         ->  pkg_b_sub_Leaf_InnerSpec
        """
        path = f"{cls.__module__}.{cls.__qualname__}" # .replace('.', '_')
        return f"{path}Spec"
    
    @classmethod
    def __build_schema_hook__(
        cls, 
        name: str, 
        type_: t.Any, 
        default: t.Any
    ) -> t.Any:
        """
        Hook for custom schema building logic.
        This is called for each field in the class.
        """
        raise ValueError(
            f"Unknown build schema hook name: {name}. "
            "This should be implemented in the subclass."
        )
    
    @classmethod
    def __build_schema__(cls) -> None:
        """
        Collect fields from *all* ancestors, then merge/override with annotations
        found on *cls* itself.  When an InitVar is later replaced by a method or
        property of the same name, we treat that as ‘no default supplied’ rather
        than letting the callable leak into runtime initialisation.
        """

        parent_fields: dict[str, tuple[t.Any, t.Any, bool]] = {}
        for base in cls.__mro__[1:]:
            if hasattr(base, "__item_fields__"):
                for n, typ, dflt, is_init in base.__item_fields__:
                    parent_fields.setdefault(n, (typ, dflt, is_init))


        ann = t.get_type_hints(cls, include_extras=True)
        for name, typ in ann.items():
            if t.get_origin(typ) is t.ClassVar:
                continue

            dflt  = getattr(cls, name, inspect._empty)
            is_iv = isinstance(typ, InitVar)

            if is_iv and (
                inspect.isfunction(dflt) or isinstance(dflt, property)
            ):
                # A callable of the same name replaced the field; treat as if the
                # user supplied *no* default.  We fall back to None so that
                # __init__ accepts a missing keyword and __post_init__ can decide
                # what to do.
                dflt = None

            if is_iv:
                typ = (
                    t.get_args(typ)[0]
                    if t.get_origin(typ) is InitVar else t.Any
                )

            parent_fields[name] = (typ, dflt, is_iv)

        # 3⃣  write back to canonical structures -------------------------------
        cls.__item_fields__ = [
            (n, *parent_fields[n]) for n in parent_fields
        ]
        cls.__is_initvar__ = {n: iv for n, (_, _, iv) in parent_fields.items()}

        # 4⃣  build / rebuild the pydantic spec --------------------------------
        spec_fields: dict[str, tuple[t.Any, t.Any]] = {}
        for n, typ, dflt, _ in cls.__item_fields__:
            origin = (
                cls.__build_schema_hook__(n, typ, dflt)
                if n in cls.__spec_hooks__
                else (
                    typ.schema()
                    if isinstance(typ, type) and issubclass(typ, BaseModule)
                    else typ
                )
            )
            spec_fields[n] = (origin, ... if dflt is inspect._empty else dflt)

        cls.__spec__ = create_model(
            f"{cls._spec_model_name()}",
            __base__       = BaseSpec,
            kind           = (t.Literal[cls.__qualname__], cls.__qualname__),
            # model_config   = ConfigDict(arbitrary_types_allowed=True),
            **spec_fields,
        )

    def __init_subclass__(cls, **__kwd):
        super(cls).__init_subclass__(**__kwd)
        if cls is BaseModule:
            return
        
        if '__spec__' not in cls.__dict__:
            cls.__build_schema__()

    def __init__(self, **__kwd: t.Any):
        self._children = []
        self._init_vars = {}
        self._parameters: dict[str, Param] = {}
        self._states: dict[str, Attr] = {}
        self._modules: dict[str, BaseModule] = {}

        for name, _typ, default, is_init in self.__class__.__item_fields__:
            if name in __kwd:
                val = __kwd.pop(name)
            elif default is not inspect._empty:
                val = default
            else:
                raise TypeError(f"Missing required keyword argument: {name!r}")

            if is_init:
                self._init_vars[name] = val
            else:
                setattr(self, name, val)

        if __kwd:
            raise TypeError(
                f"Unexpected keyword arguments: {', '.join(__kwd)}"
            )

        # child registration
        for v in vars(self).values():
            if isinstance(v, BaseModule):
                self._children.append(v)

        if hasattr(self, "__post_init__"):
            # Check if __post_init__ accepts all InitVars before calling

            # TODO: Consider whether
            # to move this to __init_subclass__
            sig = inspect.signature(self.__post_init__)
            accepted_params = set(sig.parameters.keys())
            unexpected = set(self._init_vars.keys()) - accepted_params
            if unexpected:
                raise RuntimeError(
                    f"__post_init__ does not accept InitVars passed in: {unexpected}. "
                    f"Accepted parameters: {accepted_params}. InitVars: {self._init_vars}"
                )
            self.__post_init__(**self._init_vars)
        elif len(self._init_vars) > 0:
            raise RuntimeError(
                'InitVars have been defined but there is no __post_init__ defined.'
            )

    # schema & spec helpers
    @classmethod
    def schema(cls) -> type[BaseSpec]:
        return cls.__spec__

    # ---- sub-module traversal ----------------------------------------
    def modules(
        self, *, 
        recurse: bool = True, 
        f: t.Callable[['BaseModule'], bool] | None = None):
        """Yield **self** first, then all sub-items depth-first."""
        if f is None or f(self):
            yield self
            if recurse:
                for child in self._modules.values():
                    yield from child.modules(recurse=True, f=f)

    def named_modules(
        self, *, 
        recurse: bool = True, 
        prefix: str = "", 
        f: t.Callable[['BaseModule'], bool] | None = None
    ):
        """Yield ``(dotted_name, module)`` pairs."""
        if f is None or f(self):
            yield prefix.rstrip("."), self
            if recurse:
                for name, child in self._modules.items():
                    child_prefix = f"{prefix}{name}."
                    yield from child.named_modules(recurse=True, prefix=child_prefix)

    def named_parameters(
        self, *, recurse: bool = True, prefix: str = ""
    ) -> t.Generator[tuple[str, Param], None, None]:
        """
        Yield all parameter names and their corresponding Param objects.
        """
        for name, p in self._parameters.items():
            #if train_only is None or p.training is train_only:
            # if train_only and isinstance(p, Param) or not train_only:
                # yield only if training is True or train_only is None
                # (i.e. we want all parameters)
            yield f"{prefix}{name}", p
        if recurse:
            for cname, child in self._modules.items():
                child_prefix = f"{prefix}{cname}."
                yield from child.named_parameters(recurse=True, prefix=child_prefix)

    def children(self):
        """Immediate child modules (non-recursive)."""
        return self._modules.values()

    def named_states(self, *, recurse: bool = True, prefix: str = ""):
        """
        Yield all states names and their corresponding Attr objects.
        """
        for name, s in self._states.items():
            yield f"{prefix}{name}", s
        if recurse:
            for cname, child in self._modules.items():
                child_prefix = f"{prefix}{cname}."
                yield from child.named_states(recurse=True, prefix=child_prefix)

    def apply(self, fn, *, include: t.Callable[[t.Any], bool] | t.Type | None = None):
        """
        Recursively apply *fn* to every registered object.

        If *filter_type* is given, only objects satisfying
        ``isinstance(obj, filter_type)`` are passed to *fn*.
        """
        targets = [self, *self._parameters.values(), *self._states.values()]
        for obj in targets:
            print(obj, include)
            if include is None:
                print('Including None')
                fn(obj)
            elif isinstance(include, t.Type) and isinstance(obj, include):
                print('Including type')
                fn(obj)
            elif not isinstance(include, t.Type) and include(obj):
                print('Including f')
                fn(obj)
        for child in self._modules.values():
            child.apply(fn, include=include)

    def eval_args(self):
        """Alias for ``train(False)``."""
        return self.train(False)

    def train(self, mode: bool = True):
        """Recursively set ``Param.training`` for all parameters."""
        self.training = mode
        for child in self._modules.values():
            child.train(mode)
        return self

    def named_children(self):
        """
        Yield all child module names and their corresponding modules.
        """
        return self._modules.items()

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
        raise ValueError(
            f"Unknown from_spec_hook name: {name}. "
            "This should be implemented in the subclass."
        )

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
        raise ValueError(
            f"Unknown from_spec_hook name: {name}. "
            "This should be implemented in the subclass."
        )

    def spec(
        self, *, 
        to_dict: bool = False
    ):
        """
        Serialise *this* runtime object → its spec counterpart.

        Nested `BaseModule` instances are recursively converted.
        `ModuleList` containers are converted element-wise.
        """
        data: dict[str, t.Any] = {}

        for name, is_init in self.__class__.__is_initvar__.items():
            if is_init:
                val = self._init_vars[name]
                if name in self.__spec_hooks__:
                    val = self.spec_hook(
                        name=name,
                        val=val,
                        to_dict=to_dict
                    )
                data[name] = val
                continue

            val = getattr(self, name)
            if name in self.__spec_hooks__:
                # run custom spec hook if defined
                data[name] = self.spec_hook(
                    name=name,
                    val=val,
                    to_dict=to_dict
                )
            elif isinstance(val, BaseModule):
                data[name] = val.spec(to_dict=False)
            
            else:
                data[name] = val

        spec_obj = self.__class__.__spec__(
            kind=self.__class__.__qualname__,
            id=str(id(self)),
            **data
        )
        return spec_obj.model_dump() if to_dict else spec_obj

    @classmethod
    def from_spec(
        cls,
        spec:  BaseSpec | dict,
        ctx:   "dict | None" = None,
    ):
        """
        Rebuild a runtime `BaseModule` from *spec*.

        • `ctx` caches already-created objects so identical specs
        resolve to the same instance.
        • Works for nested BaseModules       (key = spec['id'])
        • Works for Param / State / Shared   (key = spec['ref_name'])
        """
        ctx = ctx or {}

        # ---- 1) normalise input -----------------------------------------
        
        if isinstance(spec, dict) and "kind" in spec:
            spec_obj: BaseSpec = cls.__spec__.model_validate(spec)
        else:                                       # already a BaseSpec
            spec_obj = spec

        if isinstance(spec_obj, BaseSpec):
            key = spec_obj.id                       # BaseModule path
        # elif isinstance(spec_obj, dict) and "ref_name" in spec_obj:
        #     key = spec_obj["ref_name"]              # Shared / Param / State path
        else:
            key = None                              # primitives → no dedup

        if key and (hit := ctx.get(key)) is not None:
            return hit                              # reuse existing object

        kwargs: dict[str, t.Any] = {}

        for name, is_init in cls.__is_initvar__.items():
            val = getattr(spec_obj, name)

            cls_val = cls.__dict__.get(name)
            if (
                (isinstance(val, dict) 
                 and "kind" in val 
                 and cls_val is not None 
                 and issubclass(
                cls_val, BaseModule)) 
                or isinstance(val, BaseSpec)
                or name in cls.__spec_hooks__
            ):
            #    pass
            # (a) Nested BaseModule spec  -----------------------------
            # if isinstance(val, (BaseSpec, dict)):

                if isinstance(val, BaseSpec) and val.id in ctx:
                    val = ctx.get(val.id)  # reuse existing module

                elif name in cls.__spec_hooks__:
                    # run custom spec hook if defined
                    # id = val.id
                    val = cls.from_spec_hook(
                        name=name,
                        val=val,
                        ctx=ctx
                    )
                    # ctx[id] = val
                elif isinstance(val, BaseSpec):
                    id = val.id
                    sub_cls = registry[val.kind].obj
                    val = sub_cls.from_spec(val, ctx)
                    ctx[id] = val
                else:
                    # allow dicts with 'kind' to be parsed as BaseSpec
                    id = val['id']
                    sub_cls = registry[val["kind"]].obj
                    val = sub_cls.from_spec(val, ctx)
                    ctx[id] = val
                
            kwargs[name] = val

        # ---- 4) construct this module ----------------------------------
        obj = cls(**kwargs)

        # ---- 5) cache for future duplicates ----------------------------
        if key:
            ctx[key] = obj
            # ctx.put(key, obj)

        return obj

    def __setattr__(self, name, value):
        if isinstance(value, Param):
            self.register_parameter(name, value)
        elif isinstance(value, Attr):
            self.register_state(name, value)
        elif isinstance(value, BaseModule):
            self.register_module(name, value)
        else:
            super().__setattr__(name, value)

    def register_parameter(self, name: str, param: Param):
        """Register a parameter with the given name."""
        self._parameters[name] = param
        super().__setattr__(name, param)

    def register_state(self, name: str, state: Attr):
        """Register a state with the given name."""
        self._states[name] = state
        super().__setattr__(name, state)

    def register_module(self, name: str, module: 'BaseModule'):
        """Register a submodule with the given name."""
        self._modules[name] = module
        super().__setattr__(name, module)

    def parameters(self, *, recurse: bool = True, _seen: t.Optional[set[int]] = None) -> t.Iterator[Param]:
        """
        Yield all parameter names and their corresponding Param objects.
        """
        if _seen is None:
            _seen = set()

        for param in self._parameters.values():
            if id(param) not in _seen:
                # if train_only is None or param.training is train_only:
                _seen.add(id(param))
                yield param

        if recurse:
            for child in self._modules.values():
                yield from child.parameters(recurse=True, _seen=_seen)

    def state_dict(
        self,
        *,
        recurse: bool = True,
        train: bool = True,
        runtime: bool = True,
    ) -> dict[str, t.Any]:
        """
        Returns a dictionary representation of the module's state.

        Args:
            recurse: Whether to recurse into child modules.
            train: Whether to include training parameters (Param).
            runtime: Whether to include runtime states (Attr).
        """

        out: dict[str, t.Any] = {}

        if train:
            for name, param in self._parameters.items():
                out[name] = param.data

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
                elif isinstance(v, Attr) and runtime:
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
        """
        Load the state dictionary into the module.

        Args:
            sd: The state dictionary to load.
            recurse: Whether to recurse into child modules.
            train: Whether to include training parameters (Param).
            runtime: Whether to include runtime states (Attr).
            strict: Whether to enforce strict loading (i.e., all keys must match).
        """
        if not isinstance(sd, dict):
            raise TypeError(
                f"StateDict must be of type dict not {type(sd)}"
            )
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
                raise KeyError(
                    f"Unexpected keys in load_state_dict: {sorted(extra_keys)}"
                )



class RestrictedSchemaMixin:
    """
    Provide `_restricted_schema(mapping)` where **mapping** is
    {placeholder_cls: iterable_of_allowed_module_classes}.
    Patches the JSON-Schema so every "$ref" to each placeholder's *spec*
    is replaced by a `oneOf` union of the allowed spec classes.

    Purely cosmetic – runtime validation is unchanged.
    """

    @classmethod
    def _restricted_schema(
        cls,
        mapping: t.Mapping[
            type["BaseModule"],              # placeholder  (e.g. Task)
            Iterable[type["BaseModule"]]     # allowed mods (e.g. Action1…)
        ],
    ) -> dict:

        # normalise & freeze for cache key
        norm = tuple((ph, tuple(allowed)) for ph, allowed in mapping.items())
        return cls.__rs_cache(norm)

    @classmethod
    @lru_cache
    def __rs_cache(
        cls,
        norm: tuple[tuple[type["BaseModule"], tuple[type["BaseModule"], ...]], ...],
    ) -> dict:

        # 0) Build patch-tables for every placeholder
        union_schemas   = {}    # placeholder_spec_name → dict(oneOf=…)
        placeholder_refs = {}   # placeholder_spec_name → full "$ref" str

        for placeholder_cls, allowed in norm:
            allowed_specs = [m.schema() for m in allowed]
            union         = Union[tuple(allowed_specs)]
            union_schema  = TypeAdapter(union).json_schema()

            # union_schema *is* the JSON of oneOf already
            union_schemas[placeholder_cls.schema().__name__] = union_schema
            placeholder_refs[placeholder_cls.schema().__name__] = (
                f"#/$defs/{placeholder_cls.schema().__name__}"
            )

        # 1) For convenience, make a *root* union of all first-level allowed specs
        #    (not strictly required but matches earlier behaviour)
        top_specs = [s for _, allowed in norm for s in allowed]
        root_schema = TypeAdapter(Union[tuple(m.schema() for m in top_specs)]
                                  ).json_schema()

        # 2) Walk & patch
        patched = copy.deepcopy(root_schema)

        def _walk(obj):
            if isinstance(obj, dict):
                ref = obj.get("$ref")
                if ref:
                    # check each placeholder
                    for spec_name, target_ref in placeholder_refs.items():
                        if ref == target_ref:
                            obj.clear()
                            obj.update(union_schemas[spec_name])
                            break
                else:
                    for v in obj.values():
                        _walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    _walk(v)

        _walk(patched)
        return patched


V = t.TypeVar("V", bound=BaseModule)


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


@dataclass
class Checkpoint(Generic[V]):
    """Checkpoint for BaseModle objects, containing spec and state_dict."""
    spec: BaseSpec = Field(
        description="Specification of the object, including its kind and id."
    )
    state_dict: Dict[str, Any] = Field(
        description="State dictionary containing parameters and states."
    )

    def save(self, path: str):
        """Save the checkpoint to a file."""
        with open(path, 'w') as f:
            spec_data = self.spec.model_dump()
            data = {
                "spec": spec_data,
                "state_dict": self.state_dict

            }
            f.write(json.dumps(data, indent=2))
            # f.write(self.model_dump_json(indent=2))
    
    @classmethod
    def load(cls, path: str) -> "Checkpoint":
        """Load a checkpoint from a file."""
        with open(path, 'r') as f:
            data = f.read()
        data = json.loads(data)

        load_cls = registry[data['spec']['kind']]

        spec = load_cls.obj.__spec__.model_validate(data['spec'])
        state_dict = data['state_dict']
        return cls(spec=spec, state_dict=state_dict)

    @classmethod  
    def load_module(cls, path: str, ctx: Optional[dict] = None) -> V:
        """Reconstruct the BaseModule from the checkpoint."""
        if ctx is None:
            ctx = {}
        obj = cls.load(path)
        module_cls = obj.spec.load_cls()
        module = module_cls.from_spec(obj.spec, ctx=ctx)
        module.load_state_dict(obj.state_dict)
        return module
    
    @classmethod
    def save_module(self, module: BaseModule, path: str):
        """Save the BaseModule as a checkpoint."""
        spec = module.spec(to_dict=False)
        state_dict = module.state_dict(
            recurse=True, train=True, runtime=True
        )
        checkpoint = Checkpoint(
            spec=spec,
            state_dict=state_dict
        )
        checkpoint.save(path)


class Registry:
    """Registry for BaseModule classes and functions.
    Allows registration, filtering, and retrieval of objects
    by name, type, tags, and package.
    """
    def __init__(self):
        self._entries: Dict[str, RegistryEntry] = {}

    def register(self,
                 name: Optional[str] = None,
                 tags: Optional[Dict[str, Any]] = None,
                 description: Optional[str] = None) -> Callable[[Union[type, Callable]], Union[type, Callable]]:
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
        """
        Filter the registry entries based on the given criteria.
        Args:
            obj_type: The type of the object to filter by.
            tags: A dictionary of tags to filter by.
            package: The package to filter by.

        Returns:
            A dictionary of matching registry entries.
        """
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

registry = Registry()
V = t.TypeVar("V", bound=BaseModule)


class AdaptModule(
    BaseModule, 
    RestrictedSchemaMixin, 
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
    # allowed:   InitVar[t.FrozenSet[str] | None]  = None 
    # train_submods: bool = True   # expose inner numeric params?
    # adapted: V
    # fixed: bool = False

    def __post_init__(self):
        """
        Initialize the adapted module.
        """
        super().__post_init__()
        self._allowed = list()
        self._adapted_param = Param(data=None)
        self._adapted = None
        self._adapted_param.register_callback(self.update_adapted)
        self.train_submods = True
        self.fixed = False

    @property
    def allowed(self) -> t.List:
        """
        Get the allowed sub-module classes.
        """
        return [*self._allowed]

    @allowed.setter
    def allowed(self, allowed: t.List):
        """
        Set the allowed sub-module classes.
        """
        if allowed is None:
            self._allowed = None
            return
        allowed_set = set()
        
        for item in sorted(set(allowed), key=str):
            if inspect.isclass(item) and issubclass(item, BaseModule):
                allowed_set.add(to_kind(item))                      # fully-qualified
            elif isinstance(item, str):
                obj = resolve_name(item, namespace={**globals(), **locals()}, search_sys_modules=True)
                if not (inspect.isclass(obj) and issubclass(obj, BaseModule)):
                    raise TypeError(
                        f"'{item}' is not a BaseModule subclass"
                    )
                allowed_set.add(to_kind(obj))
            else:
                raise TypeError(f"Invalid entry {item!r}")
        self._allowed = list(allowed_set)

    @property
    def adapted(self) -> BaseModule:
        return self._adapted

    @adapted.setter
    def adapted(self, val: V):
        self._adapted = val
        # 1) Create a Param that holds the *spec* of the sub‑module
        if val is not None:
            self._adapted_param.unregister_callback(self.update_adapted)
            self._adapted_param.set(data=val.spec())
            self._adapted_param.register_callback(self.update_adapted)
        else:
            self._adapted_param.unregister_callback(self.update_adapted)
            self._adapted_param.set(data=None)
            self._adapted_param.register_callback(self.update_adapted)

    @property
    def adapted_param(self) -> Param:
        return self._adapted_param

    @classmethod
    def schema(
        cls,
        mapping: Mapping[type[BaseModule], Iterable[type[BaseModule]]] | None = None,
    ) -> dict:
        if mapping is None:
            return super().schema()
        # canonicalise order so schema output is deterministic regardless of
        # caller list ordering – fixes equality test failures.
        canonical: dict[type[BaseModule], tuple[type[BaseModule], ...]] = {}
        for ph, allowed in mapping.items():
            # remove dups then sort by class name for a stable order
            unique_sorted = tuple(sorted(set(allowed), key=lambda c: c.__name__))
            canonical[ph] = unique_sorted
        return cls._restricted_schema(canonical)
        # self, mapping: t.Optional[dict[type[BaseModule], t.Iterable[type[BaseModule]]]] = None):
        # if mapping is None:
        #     return super().schema()
        # return self._restricted_schema(mapping)
        # if mapping is None:
        #     return super().schema()
        # return cls._restricted_schema(mapping)

    def update_adapted(self, new_spec: BaseSpec):
        """Callback fired when *adapted_param* changes."""

        if self.fixed:
            raise RuntimeError("Cannot update adapted on a frozen AdaptModule")

        # if self.allowed and new_spec.kind not in self.allowed:

        #     raise ValueError(
        #         f"Spec kind '{new_spec.kind}' not allowed. Allowed: {sorted(self.allowed)}"
        #     )

        # old = self._adapted
        
        sub_cls = registry[new_spec.kind].obj
        self._adapted = sub_cls.from_spec(new_spec, ctx={})
        # self.on_swap(old, self._adapted)

    # def on_swap(self, old: BaseModule, new: BaseModule):
    #     """Override or monkey‑patch to react after *adapted* is rebuilt."""
    #     pass

    def fix(self):
        """Collapse to spec‑blob so only *adapted_param* remains trainable."""
        self.fixed = True

    def unfix(self, *, ctx: dict | None = None):
        if not self.fixed:
            return
        ctx = ctx or {}
        spec = self.adapted_param.data  # already a BaseSpec
        sub_cls = registry[spec.kind].obj
        self._adapted = sub_cls.from_spec(spec, ctx)
        self.fixed = False

    def forward(self, *a, **k):          # type: ignore[override]
        return self.adapted(*a, **k)

    def parameters(self, *, recurse=True, _seen=None):  # noqa: D401
        if _seen is None:
            _seen = set()
        if id(self) in _seen:
            return
        _seen.add(id(self))

        # always expose the *spec* parameter itself unless frozen
        if not self.fixed:
            yield self.adapted_param

        # inner numeric params – optional
        if recurse and self.train_submods and not self.fixed:
            yield from self.adapted.parameters(recurse=True, _seen=_seen)

    def state_dict(
        self, *, 
        recurse: bool = True, 
        train: bool = True, 
        runtime: bool = True):
        sd = {}
        # sd = super().state_dict()
        # spec Param
        # nested params / attrs
        if recurse:
            for k, v in self._adapted.state_dict(recurse=True, train=train, runtime=runtime).items():
                sd[f"_adapted.{k}"] = v
        sd["_adapted_param"] = self.adapted_param.dump()
        print(list(sd.keys()))
        return sd

    def load_state_dict(self, sd: dict[str, t.Any], *, recurse: bool = True, train: bool = True, runtime: bool = True, strict: bool = True):
        # 1) restore spec first (this rebuilds `adapted` via callback)
        # super().load_state_dict(
        #     sd, recurse=recurse, train=train,
        #     runtime=runtime, strict=strict
        # )
        if "_adapted_param" in sd:
            # pass
            cur_cls = registry[sd['_adapted_param']['kind']].obj
            spec = cur_cls.schema().model_validate(sd['_adapted_param'])
            print(spec)
            self._adapted_param.data = spec
            # self.adapted_param.load(sd["adapted_param"])
        # 2) pass nested keys to adapted module
        nested = {k[len("_adapted."):]: v for k, v in sd.items() if k.startswith("_adapted.")}
        if nested:
            self.adapted.load_state_dict(nested, recurse=True, train=train, runtime=runtime, strict=strict)
        # strict checking
        if strict:
            expected = self.state_keys(recurse=True, train=train, runtime=runtime)
            missing = expected - sd.keys()
            extra = sd.keys() - expected
            if missing:
                raise KeyError(f"Missing keys in load_state_dict: {sorted(missing)}")
            if extra:
                raise KeyError(f"Unexpected keys in load_state_dict: {sorted(extra)}")


    # def state_dict(self, *, recurse=True, train=True, runtime=True):
    #     out: dict[str, t.Any] = {}
    #     # spec is always saved so we can rebuild; treated as "train" state
    #     if train:
    #         out["adapted"] = self.adapted_param.data.model_dump()
    #     if recurse:
    #         inner = self.adapted.state_dict(recurse=True, train=train, runtime=runtime)
    #         out.update({f"adapted_vals.{k}": v for k, v in inner.items()})
    #     return out

    def render(self) -> str:  # for LLM debugging
        return f"AdaptModule(adapted={self.adapted.__class__.__name__}, fixed={self.fixed})"


class ParamSet(object):
    """ParamSet is a 
    """

    def __init__(self, params: t.List[Param]=None):
        """

        Args:
            params (t.List[Param], optional): . Defaults to None.
        """
        self.params = params or []

    @classmethod
    def build(cls, module: BaseModule) -> "ParamSet":
        """Build a ParamSet from a BaseModule, collecting all parameters."""
        params = list(
            module.parameters(recurse=True)
        )
        return cls(params=params)

    def update(self, param_set: Dict):
        """Load the parameters into a BaseModule."""
        for i, param in enumerate(self.params):
            key = f"param_{i}"
            if key in param_set:
                param.data = param_set[key]
    
    def schema(self) -> dict:
        """
        Return the JSON schema for all parameters in the set.
        """
        return {f"param_{i}": param.schema() for i, param in enumerate(self.params)}

    def to_dict(self) -> dict:
        """
        Dump all parameters' data to a dictionary.
        """
        return {
            f"param_{i}": param.dump() 
            for i, param in enumerate(self.params)
        }
