# 1st party
from __future__ import annotations
from abc import abstractmethod, ABC
import inspect
from typing import Union, Generic, Callable, Any, Dict, Optional, Union, List
import typing as t
import pydantic
from pydantic import Field
from typing import Generic, Union
import inspect
import json
from enum import Enum, auto
from dachi.utils import get_all_private_attr_annotations

from enum import Enum
from ._shareable import Runtime, Param, Shared, ParamSet, Trainable, ShareableItem
from ._base import StorableState, to_kind
from ._registry import Registry

T = t.TypeVar("T")


class StateType(Enum):

    MODULE: str = auto()
    RUNTIME: str = auto()
    PARAM: str = auto()


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


T = t.TypeVar("T")

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


class Module(pydantic.BaseModel, StorableState, Trainable):
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
        
        qual_name = to_kind(cls)
        cls.__annotations__["KIND"] = t.Literal[qual_name]

        if "KIND" not in cls.__dict__:
            
            cls.KIND = qual_name

        super().__init_subclass__(**kwargs)

        if "KIND" in cls.model_fields:
            cls.model_fields["KIND"].default = qual_name

    def model_post_init(self, __context):
        super().model_post_init(__context)
        private_annotations = get_all_private_attr_annotations(self.__class__)

        # 2) Private attributes (ignore the registry itself)
        for name in self.__private_attributes__.keys():
            if name == "_registry":
                continue
            
            try:
                value = getattr(self, name)
            except AttributeError:
                # The value has not been set up yet
                continue
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

    def to_spec(self) -> dict:
        """Convert the Module to a specification dictionary."""
        return self.model_dump()
    
    @classmethod
    def from_spec(cls, spec: dict) -> Module:
        """Reconstruct a Module from its specification dictionary."""
        return cls.model_validate(spec)
    
    @classmethod
    def to_schema(cls) -> dict:
        """Convert the Module class to a schema dictionary."""
        return cls.model_json_schema()

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



mod_registry = Registry[Module]()


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

    def to_param_set(self) -> ParamSet:
        """Build a ParamSet from a BaseModule, collecting all parameters."""
        params, annotations = list(
            self.parameters(recurse=True, with_annotations=True)
        )
        return ParamSet[annotations](params=params)



MODULE = t.TypeVar("MODULE", bound=Module)


class AdaptModule(
    Module, 
    Generic[MODULE]
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
    _adapted: Param[MODULE | None] = PrivateParam(None)
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
                yield (self._adapted, MODULE)
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
        adapted: MODULE | None = None,
        *,
        train_submods: bool = True,
        fixed: bool = False,
        **kwargs
    ) -> "AdaptModule[MODULE]":
        """Build an AdaptModule wrapping *adapted*."""
        adapt_mod = cls(
            **kwargs
        )
        adapt_mod._adapted.set(adapted)
        adapt_mod._train_submods = train_submods
        adapt_mod._fixed = fixed
        return adapt_mod

    @property
    def adapted(self) -> MODULE | None:
        """Get the adapted module
        """
        return self._adapted.data

    @adapted.setter
    def adapted(self, val: MODULE):
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
