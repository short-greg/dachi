# 1st party
import typing
from abc import ABC, abstractmethod
from typing import Dict, Literal
from dataclasses import dataclass
import json
from uuid import uuid4
import inspect, json
from typing import Any, Dict, get_type_hints, Literal
from enum import Enum
import sys
import importlib
# 3rd party
from pydantic import (
    BaseModel, create_model, ConfigDict, PrivateAttr, Field
)
# from pydantic.generics import GenericModel
from pydantic_core import core_schema
from pydantic.fields       import FieldInfo
import pydantic

# local
from ._render import render
import typing as t

# local
from . import Renderable


import inspect, typing, sys
from typing import Any, Dict, Union, get_type_hints
from uuid   import uuid4

# from pydantic             import BaseModel, ConfigDict, Field, 
# from pydantic_extra_types  import url          # just to show ext. types still work


S = typing.TypeVar('S', bound=pydantic.BaseModel)


# --- wrappers ---------------------------------------------------------
T = t.TypeVar("T")
PRIMITIVE = str | int | float | bool
V = t.TypeVar(
    "V", 
    bound=t.Union[PRIMITIVE, pydantic.BaseModel, typing.Enum, 'BaseProcess']
)

# class _Wrapper(BaseModel, t.Generic[T]): value: T
# class Attr(_Wrapper[T]): ...
# class Param(_Wrapper[T]): ...


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
    def parameters(self) -> typing.Iterator['Param']:
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


# singleton.py
"""
Decorator that turns a class into a singleton accessed via `Cls.obj`.

Compatible with normal classes *and* Pydantic BaseModel (v1 & v2).

Key points
----------
* `Cls()` raises TypeError with a helpful message.
* `Cls.obj` lazily constructs & returns the single instance.
* Each subclass automatically becomes its own singleton.
* Original validation / __init__ code *still runs* (important for Pydantic).
* Clear error chaining if the first construction fails.
"""

def singleton(cls):
    # already done?
    if getattr(cls, "__is_singleton__", False):
        return cls

    orig_meta = type(cls)                     # e.g. ModelMetaclass for Pydantic

    # ------------------------------------------------------------------ #
    # Custom metaclass extending the original one
    # ------------------------------------------------------------------ #
    class _SingletonMeta(orig_meta):
        # block direct instantiation
        def __call__(self, *a, **kw):
            raise TypeError(
                f"{self.__name__} is a singleton. "
                f"Use {self.__name__}.obj instead of instantiating it."
            )

        # runs for every subclass
        def __init__(self, name, bases, ns, **kw):
            super().__init__(name, bases, ns, **kw)
            self._instance = None
            self.__is_singleton__ = True      # avoid re-decoration

        def _get_instance(self, *a, **kw):
            if self._instance is None:
                try:
                    # CALL THE ORIGINAL FACTORY so Pydantic validation happens
                    self._instance = super(_SingletonMeta, self).__call__(*a, **kw)
                except Exception as e:
                    # annotate & re-raise with original traceback
                    raise type(e)(
                        f"Error while creating the singleton instance of "
                        f"{self.__name__}: {e}"
                    ).with_traceback(e.__traceback__) from e
            return self._instance

        # make *every* subclass a singleton automatically
        def __init_subclass__(subcls, **kw):
            super().__init_subclass__(**kw)
            singleton(subcls)                 # no-op if already wrapped

    # ------------------------------------------------------------------ #
    # Build a *subclass* of the original class under the new metaclass.
    # We inherit everything – no attribute copying (crucial for Pydantic).
    # ------------------------------------------------------------------ #
    attrs = {
        "__module__" : cls.__module__,
        "__qualname__": cls.__qualname__,
        "__doc__"   : cls.__doc__,
        "__is_singleton__": True,
    }
    Wrapped = _SingletonMeta(cls.__name__, (cls,), attrs)

    # descriptor so `Wrapped.obj` is an attribute
    class _ObjDescriptor:
        def __get__(self, _, owner):
            return owner._get_instance()

    Wrapped.obj = _ObjDescriptor()
    return Wrapped


class BaseSpec(BaseModel):
    """
    A *specification* object decoupled from the runtime constructor.
    • `from_runtime`  — runtime  → spec
    • `to_runtime`    — spec     → runtime
    • `dependencies`  — for flat specs, tells the build system which
                        other specs must be built first (by `id` or key)
    """

    kind:  str
    id:    str = Field(default_factory=lambda: str(uuid4()))
    style: Literal["flat", "structured"] = "structured"

    # ------------------------------------------------------------------
    #  Config
    # ------------------------------------------------------------------
    model_config = ConfigDict(extra="forbid")     # no stray keys

    # ------------------------------------------------------------------
    #  Dependency helper
    # ------------------------------------------------------------------
    @classmethod
    def dependencies(cls) -> list[str]:
        """
        Return a list of *ids* (or any unique keys) this spec depends on.
        Only relevant when `style == "flat"`.
        Default implementation: scan fields; if value is a Ref add its
        target_id; if value is list/tuple of Ref, add those too.
        Override if you need something richer.
        """
        deps: list[str] = []
        hints = get_type_hints(cls)
        for name, typ in hints.items():
            origin = typing.get_origin(typ) or typ
            if isinstance(origin, type) and issubclass(origin, BaseItem):
                deps.append(name)
        return deps

    # ------------------------------------------------------------------
    #  Runtime ⇄ Spec conversion
    # ------------------------------------------------------------------
    @classmethod
    def from_runtime(
        cls,
        runtime: 'BaseItem',
        *,
        ctx:   'BuildContext' | None = None,
        style: str | None = None,
    ) -> "BaseSpec":
        """
        Default 1-to-1 field extraction.
        Custom specs can override to rename / convert fields.
        """
        name = str(id(runtime))
        spec = ctx.load_spec(name)
        if spec is not None:
            return spec
        kwargs = runtime.spawn_kwargs()
        init_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, BaseItem):
                if style == 'flat' and ctx is None:
                    raise RuntimeError(
                        'BuildContext must not be none if using flat'
                    )
                elif ctx is not None:
                    v = ctx.register_item(v)
                else:
                    v = v.to_spec()
            init_kwargs[k] = v

        return cls(
            **init_kwargs,
            kind  = runtime.__class__.__qualname__,
            id    = name,
            style = style or cls.style,
        )

    # NOTE: ctx is passed so a custom spec could resolve other specs first
    def to_runtime(
        self,
        *,
        ctx: 'BuildContext' | None = None,
    ) -> 'BaseItem':
        """
        Default: parallel-field mapping back into runtime constructor.
        Override to rename / add derived args.
        """
        kwargs = {}
        # load the runtime_class from the "kind" registry
        # Find the runtime class by matching the "kind" field with class names in the module
        kind = self.kind
        runtime_cls = None

        # look up the kind
        module = sys.modules.get(self.__module__)
        if module:
            for name, obj in vars(module).items():
                if isinstance(obj, type) and obj.__qualname__ == kind:
                    runtime_cls = obj
                    break
        if runtime_cls is None:
            raise RuntimeError(f"Could not find class matching kind '{kind}' in module '{self.__module__}'")
        
        runtime_cls = ...
        for k, v in self.model_dump(exclude=['kind', 'id', 'style']).items():

            if isinstance(v, BaseSpec):
                if self.style == 'flat':
                    v = ctx.load_item(v.id)
                    if v is None:
                        raise RuntimeError(
                            'Cannot load dependencies before '
                        )
                else:
                    v = v.to_runtime(ctx=ctx)
            kwargs[k] = v

        return runtime_cls(**kwargs)

    # @classmethod
    # def _find_class_by_kind(cls, kind: str) -> type['BaseProcess'] | None:
    #     """
    #     Resolve `kind` to an actual class *without* a central registry.

    #     Strategy:
    #     1. Check the module where the current concrete subclass lives.
    #     2. Fallback: scan already-imported modules for a matching attribute.
    #        (Keeps things simple while you iterate.)
    #     """
    #     # 1 — module-local lookup
    #     mod = sys.modules.get(cls.__module__)
    #     if mod and hasattr(mod, kind):
    #         target = getattr(mod, kind)
    #         if isinstance(target, type) and issubclass(target, BaseProcess):
    #             return target

    #     # 2 — best-effort global scan of loaded modules
    #     for m in sys.modules.values():
    #         if m and hasattr(m, kind):
    #             target = getattr(m, kind)
    #             if isinstance(target, type) and issubclass(target, BaseProcess):
    #                 return target
    #     return None

class BaseItem(ABC, Renderable, Trainable):

    @abstractmethod
    @classmethod
    def to_schema(cls) -> BaseSpec:
        pass

    @abstractmethod
    def from_spec(
        cls, 
        spec: BaseSpec, 
    ) -> typing.Self:
        pass

    @abstractmethod
    def to_spec(self) -> BaseSpec:
        pass

    @abstractmethod
    def from_flat_spec(
        cls, 
        spec: BaseSpec, 
        context: 'BuildContext'=None
    ) -> typing.Self:
        pass

    @abstractmethod
    def to_flat_spec(
        self, 
        context: 'BuildContext'=None
    ) -> 'BuildContext':
        pass

    @abstractmethod
    def state_dict(self, train_only: bool=False) -> typing.Dict:
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: typing.Dict) -> typing.Self:
        pass

    @abstractmethod
    def parameters(self) -> typing.Iterator['Param']:
        pass


class BuildContext:
    """
    Keeps bidirectional maps and detects cycles.
    `.specs`   list[dict]  (flat payloads, index == id)
    `.obj2id`  {object: id}
    `.stack`   recursion stack for cycle detection
    """
    def __init__(self) -> None:
        self.i = 0
        self.specs: dict[str, BaseSpec] = {}
        self.items: dict[str, BaseItem] = {}          # dfs stack for cycle check
        self.refs: dict[str, int] = {}

    def register_spec(self, spec: BaseSpec) -> 'Ref':

        if spec.id in self.specs:
            return self.specs[spec.id]
        
        self.specs[spec.id] = spec
        self.refs[spec.id] = Ref(
            id=len(self.refs),
            target_id=spec.id
        )
        return self.refs[spec.id]
    
    def load_spec(self, id: str) -> typing.Union['Ref', None]:
        return self.refs.get(id)

    def register_item(self, id: str, item: BaseItem) -> 'Ref':

        if id in self.items:
            return self.items[id]
        
        self.items[id] = item
        self.refs[id] = Ref(
            id=len(self.refs),
            target_id=id
        )
        return self.refs[id]
    
    def load_item(self, id: str) -> typing.Union['Ref', None]:
        return self.items.get(id)

    def resolve_spec(self, ref: 'Ref') -> dict:
        return self.specs[ref.target_id]

    def resolve_item(self, ref: 'Ref') -> dict:
        return self.items[ref.target_id]


class Ref(
    BaseModel
):
    """Attr is used to specify state within a system that will be serialized
    that is not a part of the public interface.
    """
    id: str
    target_id: str


class Attr(
    BaseItem, typing.Generic[V]
):
    """Attr is used to specify state within a system that will be serialized
    that is not a part of the public interface.
    """
    def __init__(
        self, name: str, 
        data: pydantic.BaseModel | PRIMITIVE
    ):
        """

        Args:
            name (str): The param name
            data (Trainable): the data in the param
            training (bool, optional): whether training or not. Defaults to False.
        """
        self.name = name
        self.data = data

    def to_schema(cls) -> BaseSpec:
        pass
    
    def to_spec(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def from_flat_spec(
        cls, 
        spec: BaseSpec, 
        context: 'BuildContext'=None
    ) -> typing.Self:
        pass

    def to_flat_spec(
        self, 
        context: 'BuildContext'=None
    ) -> 'BuildContext':
        pass

    def from_spec(
        cls, 
        spec: BaseSpec
    ) -> typing.Self:
        pass

    def to_spec(
        self
    ) -> 'BuildContext':
        pass

    def render(self) -> str:
        """Convert the Parameter to a string
        IF the text for the paramter has not been 
        updated 

        Returns:
            str: 
        """
        if self.data is not None:
            return render(self.data)
        return self.text


class Param(
    Attr, typing.Generic[V]
):
    """Param is used to specify trainable parameters that exist within
    the system
    """
    def __init__(
        self, 
        name: str, 
        data: V, 
        training: bool=False
    ):
        """

        Args:
            name (str): The param name
            data (Trainable): the data in the param
            training (bool, optional): whether training or not. Defaults to False.
        """
        super().__init__(
            name, data
        )
        self.training = training

    def to_schema(cls) -> BaseSpec:
        # the schema fo rthe param needs to be created
        # if the value is a a "baseitem" then the schema will 
        # be the baseitem
        # if it is a pydantic base model the schema will just
        # be the standard schema for the model
        # if it is a primitive or an enum then 
        # it will need to be created
        pass

    def to_spec(self):
        pass

    def from_spec(cls, spec):
        pass

    def render(self) -> str:
        """Convert the Parameter to a string
        IF the text for the paramter has not been 
        updated 

        Returns:
            str: 
        """
        if self.data is not None:
            return render(self.data)
        return self.text


class BaseProcess(BaseItem):
    __spec__: type[BaseSpec] | None = None
    spec_style: Literal["flat", "structured"] = "structured"

    # ---------- construction  --------------------------------------------- #
    def __new__(cls, *args, **kwargs):
        if cls is BaseProcess:
            raise TypeError("BaseProcess is abstract")
        obj = super().__new__(cls)
        # spec_model = cls.to_schema()
        bound = inspect.signature(cls.__init__).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        obj.__init_args__ = bound.arguments
        return obj

    def __init__(self) -> None:
        self._attrs: Dict[str, Attr] = {}

    # ---------- spec helpers  --------------------------------------------- #
    @classmethod
    def to_schema(cls) -> type[BaseSpec]:
        if cls.__spec__ is not None:
            return cls.__spec__

        cls._validate_signature()
        sig   = inspect.signature(cls.__init__)
        hints = get_type_hints(cls.__init__, include_extras=True)

        # --- core param fields ----------------------------------------
        fields: dict[str, tuple[Any, Any]] = {}
        for name, param in list(sig.parameters.items())[1:]:
            typ     = hints[name]
            default = (param.default if param.default is not param.empty else ...)
            origin  = typing.get_origin(typ) or typ
            if isinstance(origin, type) and issubclass(origin, BaseItem):
                typ = typing.Union[origin, origin.to_schema(), Ref]
            fields[name] = (typ, default)

        # --- injected meta-fields -------------------------------------
        fields["id"]   = (str, pydantic.Field(default_factory=lambda: "unset"))  # filled at to_spec()
        fields["kind"] = (typing.Literal[cls.__qualname__], cls.__qualname__)
        fields["style"] = (typing.Literal["flat", "structured"], cls.spec_style)

        model = create_model(                               # <- pydantic-v2
            f"{cls.__name__}Spec",
            __base__      = BaseSpec,
            model_config  = ConfigDict(arbitrary_types_allowed=True),
            **fields,
        )
        cls.__spec__ = model
        return model
    
    def __init_kwargs__(self) -> typing.Dict:
        return self.__init_kwargs__

    # plain (structured) spec  --------------------------------------------- #
    def to_spec(self, id: str) -> BaseSpec:

        return self.to_schema().from_runtime(
            self, id=id, style='structured'
        )

    @classmethod
    def from_spec(cls, spec: BaseSpec | dict, *, ctx: BuildContext | None = None):

        return spec.to_runtime(
            ctx=ctx
        )

    def to_flat_spec(self, ctx: BuildContext) -> BaseSpec:
        
        return self.to_schema().from_runtime(
            self, ctx=ctx, style='flat'
        )

    @classmethod
    def from_spec(cls, spec: BaseSpec, ctx: BuildContext):

        return spec.to_runtime(
            ctx=ctx
        )

    def dependencies(self) -> list[BaseSpec]:
        """
        Topologically-sorted list of *unique* specs required
        to build this process (child-before-parent).
        """
        # loop over all "arg" dependencies
        dependencies = []
        for k, v in self.__init_kwargs__.items():
            # Check if the type hint for this kwarg is a BaseItem subclass
            param_type = type(self.__init_kwargs__[k])
            if isinstance(param_type, type) and issubclass(param_type, BaseItem):
                dependencies.append(v)
            # if isinstance(v, BaseItem):
            #     dependencies.append(v)

    # ---------- trainable helpers (unchanged) ----------------------------- #
    def register_attr(self, name: str, val: Attr) -> None:
        self._attrs[name] = val
        setattr(self, name, val)

    def state_dict(self, train_only: bool = False) -> Dict[str, Any]:
        return {
            k: v.state_dict() for k, v in self._attrs.items() 
            if (train_only and isinstance(v, Param))
            or not train_only
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for k, v in state_dict.items():
            self._attrs[k].load_state_dict(v)

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls is not BaseProcess:
            cls._validate_signature()
            cls.to_schema()

# --------------------------------------------------------------------------- #
#  Helper: forbid adding brand-new fields after construction
# --------------------------------------------------------------------------- #
def _locked_setattr(self: "BaseStruct", name: str, value: Any):
    if (
        name not in self.__class__.model_fields  # not declared in schema
        and name not in self.__dict__            # not an existing runtime attr
    ):
        raise AttributeError(
            f"Cannot add new field '{name}' to frozen BaseStruct '{self.__class__.__name__}'"
        )
    object.__setattr__(self, name, value)


class BaseStruct(BaseModel, BaseItem):
    """
    Passive data holder.

    * Runtime instance is a mutable Pydantic model **except** you cannot add
      new attributes after __post_init__.
    * A separate Spec model (cls.__spec__) is generated automatically unless
      the user defines one manually.
    """

    # ------------------------------------------------------------------ #
    #  Config
    # ------------------------------------------------------------------ #
    model_config = ConfigDict(extra="allow")   # runtime can keep private attrs

    # ------------------------------------------------------------------ #
    #  Spec generation
    # ------------------------------------------------------------------ #
    __spec__: type[BaseSpec] | None = None

    @classmethod
    def _build_spec(cls) -> type[BaseSpec]:
        """Generate <StructName>Spec dynamically."""
        sig   = inspect.signature(cls.__init__)
        hints = get_type_hints(cls.__init__, include_extras=True)

        # --- main fields ------------------------------------------------
        fields: dict[str, tuple[Any, FieldInfo | Any]] = {}
        for name, p in list(sig.parameters.items())[1:]:   # skip self
            typ     = hints[name]
            default = p.default if p.default is not p.empty else ...
            origin  = typing.get_origin(typ) or typ
            if isinstance(origin, type) and issubclass(origin, BaseItem):
                typ = Union[origin, origin.to_schema(), Ref]    # mirror BaseProcess
            fields[name] = (typ, default)

        # --- meta fields ------------------------------------------------
        fields["id"]    = (str, Field(default_factory=lambda: "unset"))
        fields["kind"]  = (typing.Literal[cls.__qualname__], cls.__qualname__)
        fields["style"] = (typing.Literal["structured", "flat"], "structured")

        spec = create_model(
            f"{cls.__name__}Spec",
            __base__     = BaseSpec,
            model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid"),
            **fields,
        )
        return spec

    @classmethod
    def to_schema(cls) -> type[BaseSpec]:
        if cls.__spec__ is None:
            cls.__spec__ = cls._build_spec()
        return cls.__spec__

    def init_kwargs(self) -> typing.Dict:
        """
        Return the kwargs to spawn a new object for this class.
        This will return a dict of field names and their values.
        """
        return self.model_dump()

    # ------------------------------------------------------------------ #
    #  Sub-class hook: validate & lock attribute addition
    # ------------------------------------------------------------------ #
    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)

        if cls is BaseStruct:
            return

        # 1. signature hygiene
        # validate_signature_no_vararg(cls.__init__)

        # 2. build spec unless user supplied one
        if getattr(cls, "__spec__", None) is None:
            cls.__spec__ = cls._build_spec()

        # 3. forbid adding brand-new attributes after post-init
        cls.__setattr__ = _locked_setattr

    # ------------------------------------------------------------------ #
    #  Construction & post-init hook
    # ------------------------------------------------------------------ #
    def model_post_init(self, __ctx: Any):
        hook = getattr(self, "__post_init__", None)
        if callable(hook):
            hook()
        # lock further additions
        object.__setattr__(self, "_locked", True)

    # ------------------------------------------------------------------ #
    #  Spec serialisation helpers
    # ------------------------------------------------------------------ #
    def to_spec(self) -> BaseSpec:

        return self.to_schema().from_runtime(
            self
        )

        # data: dict[str, Any] = {
        #     "id":    str(id(self)) if self.__spec__().id == "unset" else self.__spec__().id,
        #     "kind":  self.__class__.__qualname__,
        #     "style": "structured",
        # }

        # for name in self.to_schema().model_fields:
        #     if name in data:
        #         continue
        #     val = getattr(self, name)
        #     if isinstance(val, BaseItem):                 # recurse
        #         val = val.to_spec()
        #     data[name] = val
        # return self.__spec__(**data)

    @classmethod
    def from_spec(cls, spec: BaseSpec, *, ctx: BuildContext | None = None):

        return spec.to_runtime(
            ctx=ctx
        )

    # ------------------------------------------------------------------ #
    #  Flat spec helpers (same algo as BaseProcess)
    # ------------------------------------------------------------------ #

    def to_flat_spec(self, ctx: BuildContext) -> BaseSpec:
        return self.to_schema().from_runtime(self, ctx=ctx, style='flat')

    # ------------------------------------------------------------------ #
    #  Params discovery + state helpers
    # ------------------------------------------------------------------ #
    def parameters(self):
        for name, val in self.__dict__.items():
            if isinstance(val, Param):
                yield val
            elif isinstance(val, BaseStruct):
                yield from val.parameters()

    def state_dict(self, train_only: bool=False):
        return {
            f: p.state_dict() for f, p in self._attrs.items() 
            if train_only and isinstance(p, Param) or
            not train_only
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        for name, val in state_dict.items():
            attr = getattr(self, name, None)
            if isinstance(attr, Attr):
                attr.load_state_dict(val)

# class StructLoadException(Exception):
#     """Exception StructLoad
#     """

#     def __init__(
#         self, message="Struct loading failed.", errors=None
#     ):
#         """Create a StructLoadException with a message

#         Args:
#             message (str, optional): The message. Defaults to "Struct loading failed.".
#             errors (optional): The errors. Defaults to None.
#         """
#         super().__init__(message)
#         self.errors = errors


"""

1. Tree
2. Tasks

tree: {
sequence
    selector
        x
        y
    parallel
        a
        b
        c
}
tasks: {

}

Convrt this into the flattened form

TreeSpec

1) create the spec
2) convert to a tree
"""

# class BaseRef:
#     pass


# class BuildContext(object):
#     """BuildContext is used to record any items that have been created
#     for building up a system
#     """
#     def __init__(self):
#         """
#         """
#         self._items: typing.Dict[str, BaseItem] = {}
#         self._specs: typing.Dict[str, BaseSpec] = {}

#     def set_item(self, id: str, item: BaseItem):
#         """Set an item using the spec id

#         Args:
#             id (str): The ID for the itme
#             item (BaseItem): The item
#         """
#         self._items[id] = item

#     def get_item(self, id: str) -> None | BaseItem:
#         """Get an item from the build context

#         Args:
#             id (str): The id of the item to get

#         Returns:
#             None | BaseItem: BaseItem if item exists otherwise None
#         """
#         return self._items.get(id)

#     def set_spec(self, spec: BaseSpec):
#         """Set spec

#         Args:
#             spec (BaseSpec): Set a 
#         """
#         self._specs[spec.id] = spec

#     def get_spec(self, id: str) -> None | BaseSpec:
#         """Get a spec from the BaseSpec

#         Args:
#             id (str): The id of the spec

#         Returns:
#             None | BaseSpec: The spec specified by the id
#         """
#         return self._specs.get(id)
    
#     def to_spec(self) -> BaseSpec:
#         pass


# @dataclass
# class TemplateField(Renderable):
#     """Use for rendering a field in a BaseModel
#     """
#     type_: str
#     description: str
#     default: typing.Any = None
#     is_required: bool = True

#     def to_dict(self) -> typing.Dict:
#         """Convert the template to a dict

#         Returns:
#             typing.Dict: the template
#         """
#         return {
#             'type': self.type_,
#             'description': self.description,
#             'default': self.default,
#             'is_required': self.is_required
#         }
    
#     def render(self) -> str:
#         """Convert the template to a string

#         Returns:
#             str: The string of the template.
#         """
#         return str(self.to_dict())


# class Storable(ABC):
#     """Object to serialize objects to make them easy to recover
#     """

#     @abstractmethod
#     def load_state_dict(self, state_dict: typing.Dict):
#         """Load the state dict for the object

#         Args:
#             state_dict (typing.Dict): The state dict
#         """
#         pass
        
#     @abstractmethod
#     def state_dict(self) -> typing.Dict:
#         """Retrieve the state dict for the object

#         Returns:
#             typing.Dict: The state dict
#         """
#         pass


# class Trainable(Storable):
#     """
#     Trainable  mixin for objects that can be trained and defines the interface.
    
#     Notes:
#         - This class is designed to be used as a mixin and should not be 
#           instantiated directly.
#         - Subclasses must implement all abstract methods to provide the 
#           required functionality.
#     """
#     # @abstractmethod
#     # def update_param_dict(self, param_dict: typing.Dict):
#     #     """Update the state dict for the object

#     #     Args:
#     #         state_dict (typing.Dict): The state dict
#     #     """
#     #     pass

#     # @abstractmethod
#     # def param_dict(self) -> typing.Dict:
#     #     """Update the state dict for the object

#     #     """
#     #     pass

#     @abstractmethod
#     def data_schema(self) -> typing.Dict:
#         """

#         Returns:
#             typing.Dict: 
#         """
#         pass


# class ParamSet(object):
#     """A set of parameters
#     This is used to define a set of parameters
#     and their structure
#     """

#     def __init__(self, params: typing.List[Param]):
#         """Instantiate a set of parameters
#         Args:
#             params (typing.List[Param]): The parameters to set
#         """
#         super().__init__()
#         self.params = params

#     def data_schema(self) -> typing.Dict:
#         """
#         Generates a JSON schema dictionary for the parameters.
#         The schema defines the structure of a JSON object with the title "ParamSet".
#         It includes the properties and required fields based on the parameters.
#         Returns:
#             typing.Dict: A dictionary representing the JSON schema.
#         """
#         schema = {
#             "title": "ParamSet",
#             "type": "object",
#             "properties": {},
#             "required": []
#         }
#         for param in self.params:
#             schema["properties"][param.name] = param.data_schema()
#             schema["required"].append(param.name)
#         return schema

#     def update_param_dict(self, data: typing.Dict) -> bool:
#         """Update the text for the parameter
#         If not in "training" mode will not update

#         Args:
#             text (str): The text to update with
        
#         Returns:
#             True if updated and Fals if not (not in training mode)
#         """
#         for param in self.params:
#             if param.name in data:
#                 param.update_param_dict(data[param.name])
    
#     def param_dict(self):
#         """Update the text for the parameter
#         If not in "training" mode will not update

#         Args:
#             text (str): The text to update with
        
#         Returns:
#             True if updated and Fals if not (not in training mode)
#         """
#         data = {}
#         for param in self.params:
#             if param.training:
#                 data[param.name] = param.param_dict()
#         return data
    
#     def param_structure(self):
#         """Update the text for the parameter
#         If not in "training" mode will not update
#         Args:
#             text (str): The text to update with
#         Returns:
#             True if updated and Fals if not (not in training mode)
#         """

#         data = {}
#         for param in self.params:
#             if param.training:
#                 data[param.name] = param.param_structure()
#         return data



# def dict_state_dict(d) -> typing.Dict:
#     """Convert the dictionary into a state dict.
#     All "Storable" values will be converted into a 
#     state dict

#     Args:
#         d : The dictionary to convert

#     Returns:
#         typing.Dict: The dictionary
#     """
    
#     return {
#         k: v.state_dict() if isinstance(v, Storable)
#         else v
#         for k, v in d.items()
#     }


# def list_state_dict(d) -> typing.List:
#     """Convert the list input into a "state dict"
#     The actual output of this will be a list, though.

#     Args:
#         d : Get the state dict for a list

#     Returns:
#         typing.List: The state "dict" for the list
#     """

#     return [
#         v.state_dict() if isinstance(v, Storable)
#         else v
#         for v in d
#     ]


# def load_dict_state_dict(d, state):
#     """

#     Args:
#         d : _description_
#         state : _description_
#     """
#     for k, v in d.items():
#         if k not in state:
#             continue
#         if isinstance(v, Storable):
#             v.load_state_dict(state[k])
#         else:
#             d[k] = state[k]


# def load_list_state_dict(d, state):
        
#     for i, a in enumerate(d):
#         if isinstance(a, Storable):
#             a.load_state_dict()
#         else:
#             d[i] = state[i]


    # def data_schema(self) -> typing.Dict:

    #     sub_schema = self.data.data_schema()
    #     schema = {
    #         "title": self.name,
    #         "type": "object",
    #         "properties": sub_schema,
    #         # "required": [self.name]
    #     }
    #     return schema

    # def update_param_dict(self, data: typing.Dict) -> bool:
    #     """Update the text for the parameter
    #     If not in "training" mode will not update

    #     Args:
    #         text (str): The text to update with
        
    #     Returns:
    #         True if updated and Fals if not (not in training mode)
    #     """
    #     if self.training:
    #         self.data.load_state_dict(data)
    #         return True
    #     return False

    # def param_dict(self):
    #     """Update the text for the parameter
    #     If not in "training" mode will not update

    #     Args:
    #         text (str): The text to update with
        
    #     Returns:
    #         True if updated and Fals if not (not in training mode)
    #     """
    #     if self.training:
    #         return self.data.state_dict()
    #     return {}
    
    # def param_structure(self):
    #     if self.training:
    #         return self.data.param_structure()
    #     return {}

# # --- smoke test -------------------------------------------------------
# bb = Blackboard(messages=["hello", "world"])

# assert bb.messages == ["hello", "world"]
# assert bb._cursor.value == 0
# assert bb._capacity.value == 256
# assert bb._state_names  == {"_cursor"}
# assert bb._param_names  == {"_capacity"}

# print("✓ BaseItem scaffold works as expected.")
