from __future__ import annotations
import typing as t
from pydantic import BaseModel
from ._base4 import BaseModule, Param, State, Shared, BuildContext, BaseSpec, registry  # adjust import path
from typing import TypeVar, Generic, Iterable, ClassVar, Iterator
from pydantic import BaseModel, Field, ConfigDict, create_model, field_validator

from typing import Callable, Any, Dict, Optional, Union, List, Iterator, Iterable
from dataclasses import InitVar

from uuid import uuid4

from pydantic import create_model

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

    def __post_init__(self, items: Optional[Iterable[T]] = None):
        self._module_list = []

        if items is not None:
            for m in items:
                self.append(m)

    @classmethod
    def __build_schema_hook__(cls, name: str, type_: t.Any, default: t.Any):
        if name != "items":
            raise ValueError(f"No hook specified for {name}")
        return list[BaseSpec]

    def __len__(self) -> int:  # Positive test: len reflects number added
        return len(self._module_list)

    def __iter__(self) -> Iterator[T]:  # Positive test: order preserved
        return iter(self._module_list)

    def __getitem__(self, idx: int) -> T:  # Edge test: negative index ok
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
        if name == "items":
            if isinstance(val, list):
                val = [
                    registry[item.kind].obj.from_spec(item, ctx) 
                    for item in val
                ]
            else:
                raise TypeError(f"Expected _items to be a list, got {type(val)}")
        else: 
            raise ValueError(f"Unknown spec hook name: {name}")
        return val

    # @classmethod
    # def __build_schema__(cls) -> None:
    #     """
    #     Build a ModuleListSpec with modules: list[BaseSpec].
    #     Each element carries its own spec with kind, so we don't need to know the element type here.
    #     """
    #     cls.__spec__ = create_model(
    #         f"{cls.__name__}Spec",
    #         __base__=BaseSpec,
    #         model_config=ConfigDict(arbitrary_types_allowed=True),
    #         modules=(list[BaseSpec], ...)
    #     )

    # def __setitem__(self, idx: int, module: T):
    #     if not isinstance(module, BaseModule):
    #         raise TypeError("ModuleList elements must be BaseModule instances")

    #     try:
    #         self._module_list[idx]
    #     except IndexError as e:  # Negative test: out‑of‑range
    #         raise e

    #     name = str(idx)  # stable name equals list position of first insert

    #     # Unregister the old child (important for attr cleanliness)
    #     if hasattr(self, name):
    #         delattr(self, name)
    #     self._modules.pop(name, None)

    #     # Replace in the underlying list and registry

    #     self._module_list[idx] = module
    #     self.register_module(name, module)

    # def append(self, module: T):  # Positive & duplicate‑name bug fixed test
    #     if not isinstance(module, BaseModule):
    #         raise TypeError("ModuleList elements must be BaseModule instances")

    #     name = str(self._next_idx)
    #     self._next_idx += 1

    #     self._module_list.append(module)
    #     self.register_module(name, module)

    # ------------------------------------------------------------------
    # Spec / schema helpers – mostly defer to BaseModule but with clearer
    # diagnostics when the generic parameter is missing.
    # ------------------------------------------------------------------
    # @classmethod
    # def schema(cls) -> type[BaseSpec]:  # Negative test: raw class raises
    #     try:
    #         child_type = cls.__orig_bases__[0].__args__[0]
    #     except (AttributeError, IndexError):
    #         raise TypeError(
    #             "ModuleList must be parametrised like ModuleList[MyModule] "
    #             "to derive a schema"
    #         ) from None
    #     return child_type.schema().__class__  # type: ignore[attr-defined]

    # def spec(self) -> BaseSpec:
    #     """
    #     Serialize the ModuleList into a single spec object with modules: list[BaseSpec].
    #     Each child's spec carries its own kind, id, and fields.
    #     """
    #     return self.__class__.__spec__(
    #         kind=self.__class__.__qualname__,
    #         modules=[child.spec() for child in self._module_list]
    #     )


    # def state_dict(
    #     self,
    #     *,
    #     recurse: bool = True,
    #     train: bool = True,
    #     runtime: bool = True,
    # ) -> dict[str, t.Any]:
    #     out: dict[str, t.Any] = {}

    #     if train:
    #         for name, param in self._parameters.items():
    #             out[name] = param.data

    #     if runtime:
    #         for name, state in self._states.items():
    #             out[name] = state.data

    #     # Recurse into child BaseItems
    #     if recurse:
    #         for name, child in self._modules.items():
    #             child_sd = child.state_dict(recurse=True, train=train, runtime=runtime)
    #             for sub_name, value in child_sd.items():
    #                 out[f"{name}.{sub_name}"] = value

    #     return out

    # def load_state_dict(self, sd: list[t.Any], *,
    #                     recurse: bool = True, train: bool=True,
    #                     runtime: bool=True, strict: bool = True):
    #     if not isinstance(sd, list):
    #         raise TypeError("Expected state_dict to be a list")
    #     if strict and len(sd) != len(self._module_list):
    #         raise KeyError("Length mismatch in ItemList.load_state_dict")
    #     for child, child_sd in zip(self._module_list, sd):
    #         child.load_state_dict(child_sd, recurse=recurse,
    #                               train=train, runtime=runtime, strict=strict)


    # @classmethod
    # def from_spec(cls, spec: Union[BaseSpec, dict], context: Optional["BuildContext"] = None) -> "ModuleList":
    #     """
    #     Deserialize a ModuleList from a spec (BaseSpec or dict).
    #     Each module element is deserialized based on its `kind`.
    #     """
    #     context = context or BuildContext()  # ensure context exists

    #     # Parse dict into spec if needed
    #     if isinstance(spec, dict):
    #         spec_obj = cls.__spec__.model_validate(spec)
    #     else:
    #         spec_obj = spec

    #     modules = []
    #     for module_spec in spec_obj.modules:
    #         if isinstance(module_spec, dict):
    #             kind = module_spec.get("kind")
    #             if not kind:
    #                 raise ValueError("Missing 'kind' in module spec")
    #             mod_cls = registry[kind].obj
    #             module = mod_cls.from_spec(module_spec, context)
    #         else:
    #             # If module_spec is already a BaseSpec, use its kind
    #             mod_cls = registry[module_spec.kind].obj
    #             module = mod_cls.from_spec(module_spec, context)
    #         modules.append(module)

    #     return cls(modules)

    # def spec(self, *, context: BuildContext | None = None, to_dict=False):
    #     context = context or BuildContext()
    #     # BaseModule.spec() uses id(self) but we prefer a stable UUID
    #     self_id = getattr(self, "_spec_uuid", None) or str(uuid4())
    #     self._spec_uuid = self_id
    #     # mark dedup before serialising children
    #     context.put(self_id, self)
    #     return super().spec(context=context, to_dict=to_dict)
    # --------------------------------------------------
    # state handling
    # --------------------------------------------------
