# 1st party
import typing
from typing import Literal
import uuid
import inspect
from typing import Any, get_type_hints, Literal
# 3rd party
from pydantic import (
    BaseModel, create_model, ConfigDict, Field
)
# from pydantic.generics import GenericModel
from pydantic_core import core_schema
from pydantic.fields       import FieldInfo

# local
from ._render import render
import typing as t

# local
from . import Renderable


import inspect, typing, sys
from typing import Any, Dict, Union, get_type_hints
from uuid   import uuid4

# ---------------------------------------------------------------------#
#  The universal data-holder spec
# ---------------------------------------------------------------------#
class BaseSpec(BaseModel):
    kind : str
    id   : str = Field(default_factory=lambda: str(uuid.uuid4()))
    style: Literal["flat", "structured"] = "structured"

    model_config = ConfigDict(extra="forbid")

# ---------------------------------------------------------------------#
#  BaseItem  – a dataclass-like runtime object
# ---------------------------------------------------------------------#

class BaseItem:

    __spec__: type[BaseSpec] = BaseSpec
    # -----------------  class creation hook ------------------------
    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls is BaseItem:                    # don't process the abstract root
            return

        # -------- gather field defs from annotations ---------------
        annots  = get_type_hints(cls, include_extras=True)
        fields  : list[tuple[str,Any,Any]] = []     # (name, type, default)

        for name, typ in annots.items():
            if name.startswith('_'):               # private attrs not in spec
                continue
            default = getattr(cls, name, inspect._empty)
            fields.append((name, typ, default))

        cls.__item_fields__ = fields               # cache on the class

        # ---------- synthesize __init__ if not provided ------------
        if '__init__' not in cls.__dict__:
            sig_params = [
                inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            body_lines = []

            for name, _, default in fields:
                if default is inspect._empty:
                    sig_params.append(
                        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    )
                else:
                    sig_params.append(
                        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                          default=default)
                    )
                body_lines.append(f'    self.{name} = {name}')

            body_lines.append('    if hasattr(self, "__post_init__"): self.__post_init__()')
            src = "def __init__(" + ', '.join(p.name + ("="+repr(p.default) if p.default is not inspect._empty else '')
                                               for p in sig_params) + '):\n'
            src += '\n'.join(body_lines) + '\n'
            ns : dict[str,Any] = {}
            exec(src, ns)
            cls.__init__ = ns['__init__']

    # -------------- forbid direct instantiation --------------------
    def __new__(cls, *a, **kw):
        if cls is BaseItem:
            raise TypeError("BaseItem is abstract")
        return super().__new__(cls)

    # ---------------------- schema helpers -------------------------
    @classmethod
    def to_schema(cls) -> type[BaseSpec]:
        if cls.__spec__ is not None:
            return cls.__spec__

        if cls is BaseItem:
            raise TypeError("BaseItem has no schema")

        fields: dict[str, tuple[Any, Any]] = {}
        for name, typ, default in cls.__item_fields__:
            origin = typing.get_origin(typ) or typ
            if isinstance(origin, type) and issubclass(origin, BaseItem):
                typ = origin.to_schema()
            default_val = ... if default is inspect._empty else default
            fields[name] = (typ, default_val)

        model = create_model(
            f'{cls.__name__}Spec',
            __base__     = BaseSpec,
            model_config = ConfigDict(arbitrary_types_allowed=True),
            **fields
        )
        # _SCHEMA_CACHE[cls] = model
        return model

    def to_spec(self, *, style='structured'):
        schema_cls = self.__class__.to_schema()
        data = {}
        for name, _, _ in self.__class__.__item_fields__:
            val = getattr(self, name)
            if isinstance(val, BaseItem):
                val = val.to_spec(style=style)
            data[name] = val
        return schema_cls(**data, kind=self.__class__.__qualname__, style=style)

    #  Pydantic hook so ItemList[Task] works
    @classmethod
    def __get_pydantic_core_schema__(cls, _st, handler):
        if cls is BaseItem:
            raise TypeError("BaseItem has no schema")
        return handler.generate_schema(cls.to_schema())
