import inspect, typing, uuid
from typing import Any, get_type_hints, dataclass_transform
from pydantic import BaseModel, Field, ConfigDict, create_model
import pydantic
from pydantic.fields import FieldInfo

from ._render import render
import typing as t

try:
    from typing import dataclass_transform
except ImportError:
    from typing_extensions import dataclass_transform


T = t.TypeVar("T")
PRIMITIVE = str | int | float | bool
V = t.TypeVar(
    "V", 
    bound=t.Union[PRIMITIVE, pydantic.BaseModel, typing.Enum, 'BaseItem']
)


class BuildContext:
    """
    Keeps bidirectional maps and detects cycles.
    `.specs`   list[dict]  (flat payloads, index == id)
    `.obj2id`  {object: id}
    `.stack`   recursion stack for cycle detection
    """
    def __init__(self) -> None:
        self.i = 0
        self.specs: dict[str, 'BaseSpec'] = {}
        self.items: dict[str, 'BaseItem'] = {}          # dfs stack for cycle check
        self.refs: dict[str, int] = {}

    def register_spec(self, spec: 'BaseSpec') -> 'Ref':

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

    def register_item(self, id: str, item: 'BaseItem') -> 'Ref':

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


class BaseSpec(BaseModel):
    kind : str
    id   : str = Field(default_factory=lambda: str(uuid.uuid4()))
    style: typing.Literal["structured"] = "structured"
    model_config = ConfigDict(extra="forbid")


class Ref(
    BaseModel
):
    """Attr is used to specify state within a system that will be serialized
    that is not a part of the public interface.
    """
    id: str
    target_id: str


@dataclass_transform(kw_only_default=True, field_specifiers=(FieldInfo,))
class BaseItem:
    __spec__: type[BaseSpec] | None = None

    def __init_subclass__(cls, **kw) -> None:
        super().__init_subclass__(**kw)
        if cls is BaseItem:
            return
        annots = get_type_hints(cls, include_extras=True)
        fields: list[tuple[str, Any, Any]] = []
        for name, typ in annots.items():
            if name.startswith('_'):
                continue
            default = getattr(cls, name, inspect._empty)
            fields.append((name, typ, default))
        cls.__item_fields__ = fields

        if '__init__' not in cls.__dict__:
            params, body = [], []
            for name, _typ, default in fields:
                params.append(
                    inspect.Parameter(
                        name, inspect.Parameter.KEYWORD_ONLY,
                        default=None if default is inspect._empty else default
                    )
                )
                body.append(f'    self.{name} = {name}')
            body.append('    if hasattr(self, "__post_init__"): self.__post_init__()')

            sig = inspect.Signature(
                parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + params
            )
            ns: dict[str, Any] = {}
            exec(f'def __init__{sig}:\n' + '\n'.join(body), ns)
            cls.__init__ = ns['__init__']                               # type: ignore[attr-defined]

    def __new__(cls, *a, **kw):
        if cls is BaseItem:
            raise TypeError("BaseItem is abstract")
        return super().__new__(cls)

    # ---------- spec helpers ----------------------------------
    @classmethod
    def to_schema(cls) -> type[BaseSpec]:
        if cls.__spec__:
            return cls.__spec__
        fields: dict[str, tuple[Any, Any]] = {}
        for name, typ, default in cls.__item_fields__:
            origin = typing.get_origin(typ) or typ
            if isinstance(origin, type) and issubclass(origin, BaseItem):
                typ = origin.to_schema()
            fields[name] = (typ, ... if default is inspect._empty else default)
        cls.__spec__ = create_model(
            f'{cls.__name__}Spec', __base__=BaseSpec,
            model_config=ConfigDict(arbitrary_types_allowed=True),
            **fields
        )
        return cls.__spec__

    def to_spec(
        self,
        context: 'BuildContext' = None
    ) -> BaseSpec:
        data = {}
        for name, *_ in self.__class__.__item_fields__:
            val = getattr(self, name)
            if isinstance(val, BaseItem):
                val = val.to_spec()
            data[name] = val
        return self.__class__.to_schema()(
            kind=self.__class__.__qualname__,
            **data
        )

    @classmethod
    def from_spec(
        cls, 
        spec: BaseSpec,
        context: 'BuildContext' = None
    ) -> typing.Self:
        pass

    def register_attr(self, name: str, val: 'Attr') -> None:
        self._attrs[name] = val
        setattr(self, name, val)

    def state_dict(self, train_only: bool = False) -> t.Dict[str, Any]:
        return {
            k: v.state_dict() for k, v in self._attrs.items() 
            if (train_only and isinstance(v, Param))
            or not train_only
        }

    def load_state_dict(self, state_dict: t.Dict[str, Any]) -> None:
        for k, v in state_dict.items():
            self._attrs[k].load_state_dict(v)


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

    def state_dict(self, train_only = False):
        pass

    def load_state_dict(self, state_dict):
        pass

    def to_spec(
        self, 
        context: 'BuildContext'=None
    ) -> 'BuildContext':
        pass

    def from_spec(
        cls, 
        spec: BaseSpec,
        context: 'BuildContext'=None
    ) -> typing.Self:
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


class Shared(
    BaseItem, typing.Generic[V]
):
    """Shared is used to specify state within a system that will be shared
    across multiple processes or components.
    """
    def __init__(
        self, 
        name: str, 
        data: V
    ):
        """

        Args:
            name (str): The shared name
            data (Trainable): the data in the shared
        """
        self.name = name
        self.data = data

    def from_spec(
        cls, 
        spec: BaseSpec,
        context: 'BuildContext' = None
    ) -> typing.Self:
        pass

    def to_spec(
        self,
        context: 'BuildContext' = None
    ) -> BaseSpec:
        pass

    def render(self) -> str:
        return render(self.data)


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

    @classmethod
    def from_spec(
        cls, 
        spec: BaseSpec,
        context: 'BuildContext' = None
    ) -> typing.Self:
        pass

    def to_spec(cls, spec, context: 'BuildContext' = None):
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

    @classmethod
    def from_spec(
        cls, 
        spec: BaseSpec,
        context: 'BuildContext' = None
    ) -> typing.Self:
        pass

    def to_spec(cls, spec, context: 'BuildContext' = None):
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
