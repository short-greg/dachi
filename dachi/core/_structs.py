from __future__ import annotations

# 1st party
import typing as t
import types
from typing import TypeVar, Iterable, ClassVar, Iterator, Optional
from dataclasses import InitVar

# 2nd Party
from pydantic import BaseModel

# Local
from ._base import BaseModule, BaseSpec, registry, BaseFieldDescriptor, RestrictedSchemaMixin, UNDEFINED
from dachi.utils import is_primitive

V_co = t.TypeVar("V_co", bound=BaseModule, covariant=True)
V = t.TypeVar("V", bound=BaseModule)
T = TypeVar("T", bound=BaseModule)


class ModuleList(BaseModule, t.Generic[V]):
    """
    A list-like container whose elements are themselves `BaseModule`
    instances.  Works seamlessly with the new serialization / dedup rules.
    """
    __spec_hooks__: ClassVar[t.List[str]] = ["items"]
    items: InitVar[list[V]]

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

        # manually register generic alias machinery
        # cls.__parameters__ = (V,)  # or t.get_args(cls) if dynamic
        # cls.__orig_bases__ = (t.Generic[V],) + tuple(b for b in cls.__bases__ if b != t.Generic)
        # cls.__class_getitem__ = t.Generic.__class_getitem__

    def __post_init__(self, items: Optional[Iterable[T]] = None):
        self._module_list = []

        if items is not None:
            for m in items:
                self.append(m)

    @classmethod
    def __build_schema_hook__(
        cls, name: str, type_: t.Any, default: t.Any
    ):
        if name != "items":
            raise ValueError(f"No hook specified for {name}")
        return list[BaseSpec]

    def __len__(self) -> int:  # Positive test: len reflects number added
        return len(self._module_list)

    def __iter__(self) -> Iterator[V]:  # Positive test: order preserved
        return iter(self._module_list)

    def __getitem__(self, idx: int) -> V:  # Edge test: negative index ok
        return self._module_list[idx]

    def __setitem__(self, idx: int, value: V):
        if not isinstance(value, BaseModule):
            raise TypeError("ModuleList accepts only BaseModule instances")
        if idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds.")
        # unregister old, register new
        old_key = str(idx)
        del self._modules[old_key]
        self._module_list[idx] = value
        self.register_module(old_key, value)

    # public API – intentionally *append‑only*
    def append(self, module: V):

        if not isinstance(module, BaseModule):
            raise TypeError("ModuleList accepts only BaseModule instances")
        key = str(len(self._module_list))
        self._module_list.append(module)
        self.register_module(key, module)

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
        if name == "items":
            # Special case for _items, which is a list of modules
            if isinstance(val, list):
                val = [
                    item.spec(to_dict=to_dict) 
                    for item in self._module_list
                ]
            else:
                raise TypeError(f"Expected _items to be a list, got {type(val)}")
        else:
            raise ValueError(f"Unknown spec hook name: {name}")
        return val

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
        res = None
        if name == "items":
            if isinstance(val, list):
                res = []
                for item in val:
                    cur_item = ctx.get(item.id)
                    if cur_item is None:
                        cur_item = registry[item.kind].obj.from_spec(item, ctx) 
                        ctx[item.id] = cur_item
                    res.append(cur_item)
                
            else:
                raise TypeError(f"Expected _items to be a list, got {type(val)}")
        else: 
            raise ValueError(f"Unknown spec hook name: {name}")
        return res
    
    @property
    def module_list(self) -> list[V]:
        """
        Expose the internal list of modules.
        This is useful for iterating over the modules directly.
        """
        return [*self._module_list]
    

class ModuleDict(BaseModule, t.Generic[V, T]):
    """
    A dict-like container whose values are themselves `BaseModule`
    instances. Keys must be strings.
    """
    __spec_hooks__: ClassVar[t.List[str]] = ["items"]
    items: InitVar[dict[str, BaseModule | t.Any]] = {}

    def __post_init__(self, items: Optional[dict[str, BaseModule | t.Any]] = None):

        super().__post_init__()
        self._module_dict = {}

        if items is not None:
            for k, m in items.items():
                self[k] = m

    @classmethod
    def __build_schema_hook__(cls, name: str, type_: t.Any, default: t.Any):
        """
        Build a schema for the module dict.
        """
        if name != "items":
            raise ValueError(f"No hook specified for {name}")
        return dict[str, BaseSpec]

    def __getitem__(self, key: str) -> V:
        """Get an item from the module dict.

        Args:
            key (str): The key of the item to retrieve.

        Returns:
            V: The item associated with the key.
        """
        return self._module_dict[key]

    def __setitem__(self, key: str, val: V):
        """Set an item in the module dict.

        Args:
            key (str): The key of the item to set.
            val (V): The item to set.

        Raises:
            TypeError: If the key is not a string.
            TypeError: If the value is not a BaseModule instance or primitive.
        """
        if not isinstance(key, str):
            raise TypeError("Keys must be strings")
        
        if not isinstance(val, BaseModule) and not is_primitive(val):
            raise TypeError("Values must be BaseModule instances or primitives")
        self._module_dict[key] = val
        if isinstance(val, BaseModule):
            self.register_module(key, val)

    def __iter__(self):
        return iter(self._module_dict)

    def __len__(self):
        return len(self._module_dict)
    
    def __delattr__(self, name):
        
        if isinstance(self._module_dict[name], BaseModule):
            del self._modules[name]
        del self._module_dict[name]

    def keys(self):
        return self._module_dict.keys()

    def values(self):
        return self._module_dict.values()

    def items(self):
        return self._module_dict.items()

    def spec_hook(
        self,
        *,
        name: str,
        val: t.Any,
        to_dict: bool = False,
    ):
        """
        Spec hook for the module dict.
        """
        if name == "items":
            return {
                k: v.spec(to_dict=to_dict)
                for k, v in self._module_dict.items()
            }
        raise ValueError(f"Unknown spec hook name: {name}")

    @classmethod
    def from_spec_hook(
        cls,
        name: str,
        val: t.Any,
        ctx: "dict | None" = None,
    ) -> dict[str, V]:
        if name != "items":
            raise ValueError(f"Unknown spec hook name: {name}")
        if not isinstance(val, dict):
            raise TypeError(f"Expected a dict for 'items', got {type(val)}")

        ctx = ctx or {}
        out = {}
        for k, v in val.items():
            if isinstance(v, BaseSpec):
                id_ = v.id
                if id_ in ctx:
                    out[k] = ctx[id_]
                else:
                    mod = registry[v.kind].obj.from_spec(v, ctx)
                    ctx[id_] = mod
                    out[k] = mod
            else:
                raise TypeError(f"Expected BaseSpec values in dict, got {type(v)}")
        return out
    
    def get(self, key: str, default: Optional[V] = None) -> Optional[V]:
        """Get an item from the module dict, returning a default if not found.
        Args:
            key (str): The key of the item to retrieve.
            default (Optional[V], optional): The default value to return if the key is not found
                Defaults to None.
        Returns:
            Optional[V]: The item associated with the key, or the default value if not found.
        """
        return self._module_dict.get(key, default)


class SerialDict(BaseModule):
    """
    Dict-like container.

    • Public runtime attribute  : **`storage`**   (actual data)
    • Spec / constructor field : **`items`**

      – `items` is an `InitVar[dict[str, Any]]`; it is written into
        `self.storage` during `__post_init__`, and returned by
        `spec_hook`.  Thus the external (spec) name remains *items*,
        while internal logic uses *storage*.
    """

    # Spec hook refers to *items*
    __spec_hooks__: ClassVar[list[str]] = ["data"]
    data: InitVar[dict[str, t.Any] | None] = None

    def __post_init__(self, data: dict[str, t.Any] | None):
        self._storage = {}
        if data is not None:
            self._storage.update(data)

        for k, v in self._storage.items():
            if not isinstance(k, str):
                raise TypeError("SerialDict keys must be strings")
            if isinstance(v, BaseModule):
                self.register_module(k, v)

    # mapping helpers
    def __getitem__(self, k: str): return self._storage[k]
    def __setitem__(self, k: str, v: t.Any):
        if not isinstance(k, str):
            raise TypeError("SerialDict keys must be strings")
        if isinstance(v, BaseModule):
            self.register_module(k, v)
        elif k in self._modules:          # replacing a former module
            del self._modules[k]
        self._storage[k] = v

    def __iter__(self): return iter(self._storage)
    def __len__(self):  return len(self._storage)
    def keys(self):     return self._storage.keys()
    def values(self):   return self._storage.values()
    def items(self):    return self._storage.items()

    @classmethod
    def __build_schema_hook__(
        cls, name: str, typ: t.Any, default: t.Any
    ):
        if name != "data":
            raise ValueError(f"No spec-hook for {name}")
        return dict[str, t.Any]

    def spec_hook(self, *, name: str, val: t.Any, to_dict: bool = False):
        if name != "data":
            raise ValueError
        out: dict[str, t.Any] = {}
        for k, v in self._storage.items():
            if isinstance(v, BaseModule):
                out[k] = v.spec(to_dict=to_dict)
            elif isinstance(v, BaseModel) and to_dict:
                out[k] = v.model_dump()
            else:
                out[k] = v
        return out

    @classmethod
    def from_spec_hook(cls, name: str, val: t.Any, ctx: dict | None = None):
        if name != "data":
            raise ValueError
        if not isinstance(val, dict):
            raise TypeError("'items' must be a dict")

        ctx = ctx or {}
        rebuilt: dict[str, t.Any] = {}
        for k, v in val.items():
            if isinstance(v, BaseSpec) or (isinstance(v, dict) and "kind" in v):
                spec_obj = v if isinstance(v, BaseSpec) else \
                           registry[v["kind"]].obj.schema().model_validate(v)
                if spec_obj.id in ctx:
                    rebuilt[k] = ctx[spec_obj.id]
                else:
                    mod_cls = registry[spec_obj.kind].obj
                    rebuilt[k] = mod_cls.from_spec(spec_obj, ctx)
                    ctx[spec_obj.id] = rebuilt[k]
            else:
                rebuilt[k] = v
        return rebuilt


# SerialTuple – ordered sequence container
class SerialTuple(BaseModule):
    """
    An *ordered* container of heterogeneous, serialisable items.

    • Accepts anything Pydantic can serialise **or** any `BaseModule`.
    • Preserves insertion order.
    • Round-trips through `.spec()` / `.from_spec()` just like ModuleList,
      but without type restrictions on the stored values.
    """

    __spec_hooks__: ClassVar[list[str]] = ["data"]
    data: InitVar[tuple[t.Any, ...] | list[t.Any]]

    # -------------------------------------------------- runtime constructor
    def __post_init__(self, data: tuple[t.Any, ...] | list[t.Any] | None = None):
        self._storage: list[t.Any] = []
        if data:
            for item in data:
                self.append(item)

    @classmethod
    def __build_schema_hook__(cls, name: str, typ: t.Any, default: t.Any):
        if name != "data":
            raise ValueError(f"No spec-hook for {name}")
        # The list elements may be BaseSpec OR primitives, so we use Any
        return list[t.Any]

    def __len__(self):
        """
        Get the number of items in the container.
        """
        return len(self._storage)

    def __iter__(self):
        """
        Get an iterator over the items in the container.
        """
        return iter(self._storage)

    def __getitem__(self, idx):
        """
        Get an item from the container by index.
        """
        return self._storage[idx]

    # Only mutation we expose is append – tuples are conceptually immutable
    def append(self, value: t.Any):
        """
        Append an item to the container.
        """
        if isinstance(value, BaseModule):
            self.register_module(str(len(self._storage)), value)
        self._storage.append(value)

    # spec serialisation
    def spec_hook(self, *, name: str, val: t.Any, to_dict: bool = False):
        if name != "data":
            raise ValueError(f"Unknown spec-hook name {name}")

        out: list[t.Any] = []
        for v in self._storage:
            if isinstance(v, BaseModule):
                out.append(v.spec(to_dict=to_dict))
            elif isinstance(v, BaseModel) and to_dict:
                out.append(v.model_dump())
            else:
                out.append(v)
        return out

    # spec deserialisation
    @classmethod
    def from_spec_hook(
        cls,
        name: str,
        val: t.Any,
        ctx: dict | None = None,
    ) -> list[t.Any]:
        if name != "data":
            raise ValueError(f"Unknown spec-hook name {name}")
        if not isinstance(val, list):
            raise TypeError(f"'data' must be list[Any], got {type(val)}")

        ctx = ctx or {}
        out: list[t.Any] = []

        for item in val:
            # BaseModule spec?
            if isinstance(item, BaseSpec) or (isinstance(item, dict) and "kind" in item):
                spec_obj: BaseSpec = (
                    item if isinstance(item, BaseSpec)
                    else registry[item["kind"]].obj.schema().model_validate(item)
                )
                if spec_obj.id in ctx:
                    out.append(ctx[spec_obj.id])
                else:
                    mod_cls = registry[spec_obj.kind].obj
                    module_instance = mod_cls.from_spec(spec_obj, ctx)
                    ctx[spec_obj.id] = module_instance
                    out.append(module_instance)
            else:
                out.append(item)

        return out

# ============================================================================
# Module Field Descriptors
# ============================================================================


class ModListFieldDescriptor(BaseFieldDescriptor):
    """Descriptor for module list field."""

    def get_default(self):
        """Get default value, converting list to ModuleList if needed."""
        if self.default_factory is not UNDEFINED:
            val = self.default_factory()
            # If default_factory returns a list, convert to ModuleList
            if isinstance(val, list):
                return ModuleList(items=val)
            return val
        elif self.default is not UNDEFINED:
            return self.default
        else:
            raise TypeError(f"No default value for field")

    def validate_annotation(self, annotation) -> None:
        """Validate annotation, allowing ModuleList[T]."""
        # Call parent to extract and store types
        super().validate_annotation(annotation)

    def _extract_types_from_annotation(self, annotation) -> list:
        """Extract inner types from ModuleList[T] or ModuleList[Union[T1, T2]].

        ModuleList[Task] -> [Task]
        ModuleList[Task1 | Task2] -> [Task1, Task2]
        """
        origin = t.get_origin(annotation)

        # Handle ModuleList[T] (ModuleList is defined in this file)
        if origin is not None and isinstance(origin, type):
            try:
                is_module_list = issubclass(origin, ModuleList)
            except TypeError:
                is_module_list = False

            if is_module_list:
                args = t.get_args(annotation)
                if len(args) == 1:
                    inner_type = args[0]
                    # If inner type is Union, extract union members
                    inner_origin = t.get_origin(inner_type)
                    if inner_origin is t.Union or (isinstance(inner_origin, type) and issubclass(inner_origin, types.UnionType)):
                        return list(t.get_args(inner_type))
                    # Single inner type
                    return [inner_type]
                else:
                    raise TypeError(
                        f"modlistfield() annotation must be ModuleList[T], got {annotation}"
                    )

        # Fall back to base implementation (for non-generic cases)
        return super()._extract_types_from_annotation(annotation)

    def restricted_schema(
        self,
        *,
        filter_schema_cls: t.Type[RestrictedSchemaMixin] = type,
        variants: list | None = None,
        _profile: str = "shared",
        _seen: dict | None = None,
        **kwargs
    ) -> tuple[dict, dict]:
        """Generate restricted schema for list field."""
        if variants is None:
            base_schema = self._owner.schema()
            return (base_schema["properties"][self._name], {})

        # Process variants
        variant_schemas = self._schema_process_variants(
            variants,
            restricted_schema_cls=filter_schema_cls,
            _seen=_seen,
            **kwargs
        )

        # Build entries
        entries = [(self._schema_name_from_dict(s), s) for s in variant_schemas]

        # Build union
        if _profile == "shared":
            union_name = self._schema_allowed_union_name(self._name)
            defs = {union_name: {"oneOf": self._schema_build_refs(entries)}}
            for name, schema in entries:
                defs[name] = schema

            field_schema = {
                "type": "array",
                "items": {"$ref": f"#/$defs/{union_name}"}
            }
            return (field_schema, defs)
        else:
            # Build defs for individual schemas (but not union)
            defs = {name: schema for name, schema in entries}

            field_schema = {
                "type": "array",
                "items": self._schema_make_union_inline(entries)
            }
            return (field_schema, defs)


class ModDictFieldDescriptor(BaseFieldDescriptor):
    """Descriptor for module dict field."""

    def get_default(self):
        """Get default value, converting dict to ModuleDict if needed."""
        if self.default_factory is not UNDEFINED:
            val = self.default_factory()
            # If default_factory returns a dict, convert to ModuleDict
            if isinstance(val, dict):
                return ModuleDict(items=val)
            return val
        elif self.default is not UNDEFINED:
            return self.default
        else:
            raise TypeError(f"No default value for field")

    def _extract_types_from_annotation(self, annotation) -> list:
        """Extract value types from ModuleDict[K, V] or ModuleDict[K, Union[V1, V2]].

        ModuleDict[str, State] -> [State]
        ModuleDict[str, State1 | State2] -> [State1, State2]
        """
        origin = t.get_origin(annotation)

        # Handle ModuleDict[K, V] (ModuleDict is defined in this file)
        if origin is not None:
            if isinstance(origin, type) and issubclass(origin, ModuleDict):
                args = t.get_args(annotation)
                if len(args) == 2:
                    key_type, value_type = args
                    # Validate key type
                    if key_type not in (str, int):
                        raise TypeError(
                            f"moddictfield() key type must be str or int, got {key_type}"
                        )
                    # If value type is Union, extract union members
                    value_origin = t.get_origin(value_type)
                    if value_origin is t.Union or (isinstance(value_origin, type) and issubclass(value_origin, types.UnionType)):
                        return list(t.get_args(value_type))
                    # Single value type
                    return [value_type]
            else:
                raise TypeError(
                    f"moddictfield() annotation must be ModuleDict[K, V], got {annotation}"
                )

        # Fall back to base implementation (for non-generic cases)
        return super()._extract_types_from_annotation(annotation)

    def validate_annotation(self, annotation) -> None:
        """Validate dict annotation (key type validation happens in _extract_types_from_annotation)."""
        super().validate_annotation(annotation)

    def restricted_schema(
        self,
        *,
        filter_schema_cls: t.Type[RestrictedSchemaMixin] = type,
        variants: list | None = None,
        _profile: str = "shared",
        _seen: dict | None = None,
        **kwargs
    ) -> tuple[dict, dict]:
        """Generate restricted schema for dict field."""
        if variants is None:
            base_schema = self._owner.schema()
            return (base_schema["properties"][self._name], {})

        # Process variants
        variant_schemas = self._schema_process_variants(
            variants,
            restricted_schema_cls=filter_schema_cls,
            _seen=_seen,
            **kwargs
        )

        # Build entries
        entries = [(self._schema_name_from_dict(s), s) for s in variant_schemas]

        # Build union
        if _profile == "shared":
            union_name = self._schema_allowed_union_name(self._name)
            defs = {union_name: {"oneOf": self._schema_build_refs(entries)}}
            for name, schema in entries:
                defs[name] = schema

            field_schema = {
                "type": "object",
                "additionalProperties": {"$ref": f"#/$defs/{union_name}"}
            }
            return (field_schema, defs)
        else:
            # Build defs for individual schemas (but not union)
            defs = {name: schema for name, schema in entries}

            field_schema = {
                "type": "object",
                "additionalProperties": self._schema_make_union_inline(entries)
            }
            return (field_schema, defs)


def modlistfield(typ=UNDEFINED, default=UNDEFINED, default_factory=UNDEFINED) -> ModListFieldDescriptor:
    """Mark field as containing a list of BaseModule instances."""
    return ModListFieldDescriptor(typ=typ, default=default, default_factory=default_factory)


def moddictfield(typ=UNDEFINED, default=UNDEFINED, default_factory=UNDEFINED) -> ModDictFieldDescriptor:
    """Mark field as containing a dict of BaseModule instances."""
    return ModDictFieldDescriptor(typ=typ, default=default, default_factory=default_factory)
