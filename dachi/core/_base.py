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
    BaseModel, create_model, ConfigDict, PrivateAttr
)
from pydantic.generics import GenericModel
from pydantic_core import core_schema
import pydantic

# local
from ._render import render
import typing as t

# local
from . import Renderable

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
    kind: str
    id: str = pydantic.Field(default_factory=uuid4)
    model_config = ConfigDict(extra="forbid")
    style: Literal["flat", "structured"] = "structured"



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

    @abstractmethod
    def state_dict(self, only_trainable: bool=False) -> typing.Dict:
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
        self.specs: list[dict] = []
        self.obj2id: dict[Any, int] = {}
        self.stack: list[int] = []          # dfs stack for cycle check

    # ---- helpers --------------------------------------------------------- #
    def register(self, obj, payload: dict) -> 'Ref':
        if obj in self.obj2id:              # already emitted
            return Ref(id=len(self.specs), target_id=self.obj2id[obj])

        if obj in self.stack:
            raise ValueError("Cyclic dependency detected")

        idx = len(self.specs)
        self.obj2id[obj] = idx
        self.specs.append(payload)
        return Ref(id=len(self.specs), target_id=idx)

    def resolve(self, ref: 'Ref') -> dict:
        return self.specs[ref.target_id]



class Ref(
    BaseModel, typing.Generic[V]
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
        spec_model = cls.to_schema()
        bound = inspect.signature(cls.__init__).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        obj.__spec_obj__ = spec_model(**bound.arguments)
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

    # plain (structured) spec  --------------------------------------------- #
    def to_spec(self) -> BaseSpec:
        data = {"style": self.spec_style,
                "kind":  self.__class__.__qualname__}

        # user fields
        for fld in self.__spec__.model_fields:
            if fld in {"style", "kind", "id"}:
                continue
            val = getattr(self, fld)
            data[fld] = val.to_spec() if isinstance(val, BaseItem) else val

        # id: caller supplied?  else use python object id
        supplied = getattr(self.__spec_obj__, "id", None)
        data["id"] = supplied if supplied and supplied != "unset" else str(id(self))

        return self.__spec__(**data)

    @classmethod
    def _find_class_by_kind(cls, kind: str) -> type['BaseProcess'] | None:
        """
        Resolve `kind` to an actual class *without* a central registry.

        Strategy:
        1. Check the module where the current concrete subclass lives.
        2. Fallback: scan already-imported modules for a matching attribute.
           (Keeps things simple while you iterate.)
        """
        # 1 — module-local lookup
        mod = sys.modules.get(cls.__module__)
        if mod and hasattr(mod, kind):
            target = getattr(mod, kind)
            if isinstance(target, type) and issubclass(target, BaseProcess):
                return target

        # 2 — best-effort global scan of loaded modules
        for m in sys.modules.values():
            if m and hasattr(m, kind):
                target = getattr(m, kind)
                if isinstance(target, type) and issubclass(target, BaseProcess):
                    return target
        return None

    @classmethod
    def from_spec(cls, spec: BaseSpec | dict, *, ctx: BuildContext | None = None):
        # --- handle raw dict first ----------------------------------
        if isinstance(spec, dict):
            target_kind = spec.get("kind")
            if target_kind and target_kind != cls.__qualname__:
                real_cls = cls._find_class_by_kind(target_kind) or cls
                return real_cls.from_spec(spec, ctx=ctx)   # recurse to correct class

            # strip extras then validate
            model_fields = cls.to_schema().model_fields
            spec = cls.to_schema()(
                **{k: v for k, v in spec.items() if k in model_fields}
            )

        # (rest of the original from_spec stays identical)
        # ------------------------------------------------------------------
        sig = inspect.signature(cls.__init__)
        kwargs = {}
        for name in list(sig.parameters)[1:]:
            val = getattr(spec, name)
            if isinstance(val, Ref):
                if ctx is None:
                    raise TypeError("Ref found but no BuildContext supplied")
                payload = ctx.resolve(val)
                tgt_cls = get_type_hints(cls.__init__)[name]
                val = tgt_cls.from_spec(payload, ctx=ctx)
            elif isinstance(val, BaseSpec):
                tgt_cls = get_type_hints(cls.__init__)[name]
                val = tgt_cls.from_spec(val)
            kwargs[name] = val
        return cls(**kwargs)

    # ---------- flat spec -------------------------------------------------- #
    def to_flat_spec(self, ctx: BuildContext | None = None) -> BuildContext:
        if ctx is None:
            ctx = BuildContext()

        payload: dict[str, Any] = {"_type": self.__class__.__qualname__}
        for name in self.__spec__.model_fields:
            val = getattr(self, name)
            if isinstance(val, BaseItem):
                ref = val.to_flat_spec(ctx)
                payload[name] = ref
            else:
                payload[name] = val

        ctx.register(self, payload)           # may raise on cycles
        return ctx

    @classmethod
    def from_flat_spec(cls, ctx: BuildContext, id: int):
        payload = ctx.specs[id]
        return cls.from_spec(payload, ctx=ctx)

    # ---------- dependency extraction ------------------------------------- #
    def dependencies(self) -> list[BaseSpec]:
        """
        Topologically-sorted list of *unique* specs required
        to build this process (child-before-parent).
        """
        ctx = self.to_flat_spec()
        return [cls.from_spec(p)              # type: ignore
                for p in ctx.specs]           # already in topo order

    # ---------- trainable helpers (unchanged) ----------------------------- #
    def register_attr(self, name: str, val: Attr) -> None:
        self._attrs[name] = val
        setattr(self, name, val)

    def state_dict(self, train_only: bool = False) -> Dict[str, Any]:
        return {k: v.value for k, v in self._attrs.items()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for k, v in state.items():
            self._attrs[k].value = v

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls is not BaseProcess:
            cls._validate_signature()
            cls.to_schema()


# --- BaseItem ---------------------------------------------------------
class BaseStruct(BaseModel, BaseItem):
    model_config = ConfigDict(extra="allow")           # accept attrs added later
    _attr_names: set[str] = PrivateAttr(default_factory=set)
    # _param_names: set[str] = PrivateAttr(default_factory=set)

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        orig_setattr = cls.__setattr__

        # TODO: Create Spec

        def magic_setattr(self, name, value):
            orig_setattr(self, name, value)            # real assignment
            if isinstance(value, Attr):
                self._attr_names.add(name)

        cls.__setattr__ = magic_setattr

    def __new__(cls, *args, **kwargs):
        if cls is BaseStruct:
            raise TypeError("BaseItem is an abstract base and cannot be instantiated directly.")
        return super().__new__(cls)

    def model_post_init(self, __ctx):
        hook = getattr(self, "__post_init__", None)
        if callable(hook):
            hook()

    def to_schema(cls) -> type[BaseSpec]:
        # TODO: Create a BaseSpec
        #  this is a pydantic.BaseModel
        #
        pass
        # return cls.model_json_schema()

    def to_spec(self) -> BaseSpec:
        pass    

    def from_spec(cls, spec: BaseSpec):
        pass

    def parameters(self) -> typing.Iterator[Param]:
        # TODO: loop over all "parameters"
        pass



    # def trainable_dict(self) -> Dict[str, Any]:
    #     return {k: v.value for k, v in self._params.items()}


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
