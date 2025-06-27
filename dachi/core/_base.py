from __future__ import annotations
from abc import abstractmethod, ABC

from typing import Generic, Callable, Any, Dict, Optional, Union, List
import inspect
from dataclasses import InitVar
import inspect
import typing as t
from uuid import uuid4
import json
from dataclasses import dataclass

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


try:  # 3.12+
    from typing import dataclass_transform
except ImportError:  # 3.8–3.11
    from typing_extensions import dataclass_transform

from pydantic import BaseModel, Field, ConfigDict, create_model, field_validator

T = t.TypeVar("T")
J = t.TypeVar("J", bound=t.Union[BaseModel, dict, str, int, float, bool])

from typing import Generic, Union
# -----------------------------------------------------------
# Shareable leaf hierarchy
# -----------------------------------------------------------

def to_kind(cls): 
    """Convert a class to its kind."""
    
    return cls.__qualname__


class ShareableItem(t.Generic[J]):
    """Serializable leaf object with a ``data`` field."""

    def __init__(self, data: J):
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

    def get(self) -> J:

        return self._data
    
    def set(self, value: J):
        expected_type = self._get_expected_type()
        print(expected_type)
        if (
            expected_type is not None and not isinstance(value, expected_type)
        ):
            raise TypeError(f"Expected data of type {expected_type}, got {type(value)}")
        self._data = value
        self.update_data_hook(value)

    @property
    def data(self) -> J:
        """Get the data value."""
        return self.get()

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

    def register_callback(self, callback: Callable[[T], None]) -> None:
        """Register a callback to be called when the data is updated."""
        self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[T], None]) -> None:
        """Unregister a previously registered callback."""
        self._callbacks.remove(callback)

    def _get_expected_type(self):
        """Resolve expected type from the immediate generic base (Param, State, Shared)."""
        cls = self.__class__
        for base in getattr(cls, "__orig_bases__", []):
            origin = getattr(base, "__origin__", None)
            if origin in {Param, Attr, Shared}:
                return t.get_args(base)[0]
        return None


class Param(ShareableItem[J]):
    """Trainable parameter; ``training`` may be toggled to freeze it."""
    
    def __init__(self, data: J, fixed: bool=False):

        super().__init__(data)
        self._fixed = fixed

    def set(self, value):
        if self._fixed:
            raise RuntimeError(
                'Cannot set parameter that is fixed.'
            )
        value = super().set(value)

    def is_fixed(self) -> bool:
        return self._fixed
    
    def fix(self):

        self._fixed = True

    def unfix(self):

        self._fixed = False


    # model_config = ConfigDict(
    #     json_schema_extra=lambda s, h: {"properties": {"data": s["properties"]["data"]}}
    # )
    # def __init__(self, data):
    #     super().__init__(data)
    #     self._callbacks: list[Callable[[T]]] = []


class Attr(ShareableItem[J]):
    """Mutable runtime state (e.g. counters, RNG seeds, rolling averages)."""
    pass


class Shared(ShareableItem[J]):
    """Pointer‑like wrapper whose value should *not* enter ``state_dict``."""
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
    kind : str
    id   : str = Field(
        default_factory=lambda: str(uuid4())
    )
    style: t.Literal['structured'] = 'structured'

    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=False)

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


# BaseModule – runtime process node
@dataclass_transform(kw_only_default=True, field_specifiers=(Field,))
class BaseModule:
    """Dataclass‑like runtime object without exec‑generated ``__init__``."""

    # populated by __init_subclass__
    __spec__: t.ClassVar[type[BaseSpec]]

    __spec_hooks__: t.ClassVar[t.List[str]] = []
    __item_fields__: t.ClassVar[list[tuple[str, t.Any, t.Any, bool]]]
    __is_initvar__: t.ClassVar[dict[str, bool]]
    training: bool = True  # True if any Module is in training mode; False when not

    # ---------------------------------------------------
    # class construction hook
    # ---------------------------------------------------
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
        ann = t.get_type_hints(cls, include_extras=True)
        fields: list[tuple[str, t.Any, t.Any, bool]] = []

        field_set = set()
        for name, type_ in ann.items():
            # Skip ClassVar or private
            if t.get_origin(type_) is t.ClassVar:
                continue

            # default = cls.__dict__.get(name, inspect._empty)
            default = getattr(cls, name, inspect._empty)
            is_init = isinstance(type_, InitVar)

            if is_init:
                type_ = t.get_args(type_)[0] if t.get_origin(type_) is InitVar else t.Any

            fields.append((name, type_, default, is_init))
            field_set.add(name)

        if len(set(cls.__spec_hooks__).difference(field_set)) != 0:
            raise ValueError(
                f"SpecHooks contains fields not specified "
                f"{cls.__spec_hooks__}"
            )

        cls.__item_fields__ = fields
        cls.__is_initvar__  = {
            n: is_init for n, *_, is_init in fields
        }

        # Now build the Pydantic model
        spec_fields: dict[str, tuple[t.Any, t.Any]] = {}
        for name, typ, _raw_default, _ in fields:
            # 1) fetch whatever MRO gives you
            for base in cls.__mro__[1:]:  # skip cls itself
                member = base.__dict__.get(name)
                if member is None:
                    continue
                # only object‐level routines or properties cause trouble
                if callable(member) or isinstance(member, property):
                    raise RuntimeError(
                        f"Field name {name!r} conflicts with inherited "
                        f"{'method' if callable(member) else 'property'} "
                        f"'{name}' on {base.__name__}; please rename the field."
                    )
            candidate = getattr(cls, name, inspect._empty)

            # 2) only accept it if the class itself defined it,
            #    and it really matches the candidate
            own = cls.__dict__.get(name, inspect._empty)
            if own is candidate:
                default = own
            else:
                default = inspect._empty

            # 3) dispatch to hook or normal mapping
            if name in cls.__spec_hooks__:
                origin = cls.__build_schema_hook__(name, typ, default)
            else:
                origin = typ
                if isinstance(origin, type) and issubclass(origin, BaseModule):
                    origin = origin.schema()

            spec_fields[name] = (
                origin,
                ... if default is inspect._empty else default
            )
        # for name, type_, default, _ in fields:
        #     if name in cls.__spec_hooks__:
        #         origin = cls.__build_schema_hook__(name, type_, default)
        #     else:
        #         origin = type_
        #         if (
        #             isinstance(origin, type) 
        #             and issubclass(origin, BaseModule)
        #         ):
        #             origin = origin.schema()

        #     spec_fields[name] = (
        #         origin,
        #         ... if default is inspect._empty else default
        #     )

        cls.__spec__ = create_model(
            f"{cls._spec_model_name()}",
            __base__       = BaseSpec,
            kind = (t.Literal[cls.__qualname__], cls.__qualname__),
            model_config   = ConfigDict(arbitrary_types_allowed=True),
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
        self._states: dict[str, Attr] = {}
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
        self, *, recurse: bool = True, prefix: str = ""
    ) -> t.Generator[tuple[str, Param]]:
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

    def eval_args(self):
        """Alias for ``train(False)``."""
        return self.train(False)

    def train(self, mode: bool = True):
        """Recursively set ``Param.training`` for all parameters."""
        self.training = mode
        for child in self._modules.values():
            child.train(mode)
        return self

    # @property
    # def training(self) -> bool:
    #     # True if ANY param is in training mode; False when all frozen
    #     return any(p.training for p in self._parameters.values())

    def named_children(self):
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

        # ---- 2) deduplication key ---------------------------------------
        if isinstance(spec_obj, BaseSpec):
            key = spec_obj.id                       # BaseModule path
        # elif isinstance(spec_obj, dict) and "ref_name" in spec_obj:
        #     key = spec_obj["ref_name"]              # Shared / Param / State path
        else:
            key = None                              # primitives → no dedup

        if key and (hit := ctx.get(key)) is not None:
            return hit                              # reuse existing object

        # ---- 3) build kwargs for this module ----------------------------
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
                print(name)

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
                    print('Modules: ', val)
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
        self._parameters[name] = param
        super().__setattr__(name, param)

    def register_state(self, name: str, state: Attr):
        self._states[name] = state
        super().__setattr__(name, state)

    def register_module(self, name: str, module: BaseModule):
        self._modules[name] = module
        super().__setattr__(name, module)

    def parameters(self, *, recurse: bool = True, _seen: t.Optional[set[int]] = None) -> t.Iterator[Param]:
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
    
    def __call__(self,
                 name: Optional[str] = None,
                 tags: Optional[Dict[str, Any]] = None,
                 description: Optional[str] = None) -> Callable[[Union[type, Callable]], Union[type, Callable]]:
        return self.register(
            name=name,
            tags=tags,
            description=description
        )


registry = Registry()


class AdaptModule(BaseModule, Generic[V]):
    
    adapted: V
    fixed: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.adapted_param = Param(
            data=self.adapted.spec()
        )
        self.adapted_param.register_callback(
            self.update_adapted
        )
        self.training = True

    def fix(self):
        """Collapse to spec-blob so LLM / optimiser sees a single Param-like leaf."""
        if not self.fixed:
            # self.adapted = self.adapted.spec(to_dict=True)
            self.fixed = True

    def unfix(self, *, ctx: t.Dict | None = None):
        """Rebuild the real module from the stored spec."""
        if self.fixed:
            ctx = ctx or dict()
            sub_cls = registry[self.adapted["kind"]].obj
            self.adapted = sub_cls.from_spec(self.adapted, ctx)
            self.fixed = False

    def update_adapted(self, adapted: BaseSpec):

        if self.fixed:
            raise RuntimeError(
                "Cannot update adapted on a frozen ParamModule"
            )
        self.adapted = (
            self.adapted.from_spec(adapted, ctx=None)
        )

    # traversal overrides 
    def parameters(self, *, recurse=True, _seen=None) -> t.Iterator[Param]:
        if _seen is None:
            _seen = set()
        if not self.fixed:
            # behave like a single scalar param: expose the spec blob
            # fake = Param(data=self.adapted.schema(), training=True)
            if id(self) not in (_seen or set()):
               yield self.adapted_param

        yield from self.adapted.parameters(
            recurse=recurse, _seen=_seen
        )

    # TODO: I think this is not correct
    # Even if it is not frozen it should update
    # the state dict of adapted
    # so state_dict should consist of the spec +
    # the state_dict of adapted
    def state_dict(self, *, recurse=True, train=True, runtime=True):
        out = {}
        if not self.fixed:
            # spec is the "value"; no runtime state
            out.update({"adapted": self.adapted.spec(to_dict=True)})
        if recurse:
            inner = self.adapted.state_dict(
                recurse=True, train=train, runtime=runtime
            )
            out.update({f"adapted_vals.{k}": v for k, v in inner.items()})
        return out

    def load_state_dict(self, sd, *, recurse=True, train=True, runtime=True, strict=True):
        if self.fixed:
            if "adapted" in sd:
                adapted = sd["adapted"]
                self.adapted = self.adapted.__class__.from_spec(
                    adapted, ctx=None
                )
            elif strict:
                raise KeyError("Missing key 'adapted' for frozen ParamProcess")
            # return

        # pass through to child
        prefix = "adapted_vals."
        inner_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        self.adapted.load_state_dict(
            inner_sd, recurse=recurse, train=train, runtime=runtime, strict=strict
        )


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
