from __future__ import annotations
import typing as t
from pydantic import BaseModel
from ._base4 import BaseModule, Param, State, Shared, BuildContext, BaseSpec, registry  # adjust import path
from typing import TypeVar, Generic, Iterable, Iterator
from pydantic import BaseModel, Field, ConfigDict, create_model, field_validator

from typing import Callable, Any, Dict, Optional, Union, List


from pydantic import create_model, field_validator

V_co = t.TypeVar("V_co", bound=BaseModule, covariant=True)


T = TypeVar("T", bound=BaseModule)


class ModuleList(BaseModule, Generic[T]):
    """A *sequential* container for child *BaseModule* objects.

    *   Preserves insertion order (iteration behaves like list).
    *   Names for registered sub‑modules are **stable**; they never
        change after insertion and are guaranteed unique for the life
        of the object.
    *   Supports ``append`` and ``__setitem__``.  Removal methods are
        intentionally *not* implemented – the container is conceptually
        *append‑only* (like PyTorch's ``nn.ModuleList``).
    """

    # NOTE: we expose the internal list only for typing purposes; users
    # should treat it as read‑only.
    _module_list: List[T]
    _next_idx: int  # monotonically‑increasing counter for unique names

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __init__(self, modules: Optional[Iterable[T]] = None):
        super().__init__()
        self._module_list = []
        self._next_idx = 0

        if modules is not None:
            for m in modules:
                self.append(m)

    # ------------------------------------------------------------------
    # Python list emulation (partial)
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # Positive test: len reflects number added
        return len(self._module_list)

    def __iter__(self) -> Iterator[T]:  # Positive test: order preserved
        return iter(self._module_list)

    def __getitem__(self, idx: int) -> T:  # Edge test: negative index ok
        return self._module_list[idx]

    def __setitem__(self, idx: int, module: T):
        if not isinstance(module, BaseModule):
            raise TypeError("ModuleList elements must be BaseModule instances")

        try:
            self._module_list[idx]
        except IndexError as e:  # Negative test: out‑of‑range
            raise e

        name = str(idx)  # stable name equals list position of first insert

        # Unregister the old child (important for attr cleanliness)
        if hasattr(self, name):
            delattr(self, name)
        self._modules.pop(name, None)

        # Replace in the underlying list and registry

        self._module_list[idx] = module
        self.register_module(name, module)

    # public API – intentionally *append‑only*
    def append(self, module: T):  # Positive & duplicate‑name bug fixed test
        if not isinstance(module, BaseModule):
            raise TypeError("ModuleList elements must be BaseModule instances")

        name = str(self._next_idx)
        self._next_idx += 1

        self._module_list.append(module)
        self.register_module(name, module)

    # ------------------------------------------------------------------
    # Spec / schema helpers – mostly defer to BaseModule but with clearer
    # diagnostics when the generic parameter is missing.
    # ------------------------------------------------------------------
    @classmethod
    def schema(cls) -> type[BaseSpec]:  # Negative test: raw class raises
        try:
            child_type = cls.__orig_bases__[0].__args__[0]
        except (AttributeError, IndexError):
            raise TypeError(
                "ModuleList must be parametrised like ModuleList[MyModule] "
                "to derive a schema"
            ) from None
        return child_type.schema().__class__  # type: ignore[attr-defined]


    @classmethod
    def __build_schema__(cls) -> None:
        """
        Build a ModuleListSpec with modules: list[BaseSpec].
        Each element carries its own spec with kind, so we don't need to know the element type here.
        """
        cls.__spec__ = create_model(
            f"{cls.__name__}Spec",
            __base__=BaseSpec,
            model_config=ConfigDict(arbitrary_types_allowed=True),
            modules=(list[BaseSpec], ...)
        )

    def spec(self) -> BaseSpec:
        """
        Serialize the ModuleList into a single spec object with modules: list[BaseSpec].
        Each child's spec carries its own kind, id, and fields.
        """
        return self.__class__.__spec__(
            kind=self.__class__.__qualname__,
            modules=[child.spec() for child in self._module_list]
        )

    # --------------------------------------------------
    # state handling
    # --------------------------------------------------
    def state_dict(self, *, recurse: bool = True,
                   train: bool = True, runtime: bool = True) -> list[t.Any]:
        return [child.state_dict(recurse=recurse, train=train, runtime=runtime)
                for child in self._module_list]

    def load_state_dict(self, sd: list[t.Any], *,
                        recurse: bool = True, train: bool=True,
                        runtime: bool=True, strict: bool = True):
        if not isinstance(sd, list):
            raise TypeError("Expected state_dict to be a list")
        if strict and len(sd) != len(self._module_list):
            raise KeyError("Length mismatch in ItemList.load_state_dict")
        for child, child_sd in zip(self._module_list, sd):
            child.load_state_dict(child_sd, recurse=recurse,
                                  train=train, runtime=runtime, strict=strict)

    @classmethod
    def from_spec(cls, spec: Union[BaseSpec, dict], context: Optional["BuildContext"] = None) -> "ModuleList":
        """
        Deserialize a ModuleList from a spec (BaseSpec or dict).
        Each module element is deserialized based on its `kind`.
        """
        context = context or BuildContext()  # ensure context exists

        # Parse dict into spec if needed
        if isinstance(spec, dict):
            spec_obj = cls.__spec__.model_validate(spec)
        else:
            spec_obj = spec

        modules = []
        for module_spec in spec_obj.modules:
            if isinstance(module_spec, dict):
                kind = module_spec.get("kind")
                if not kind:
                    raise ValueError("Missing 'kind' in module spec")
                mod_cls = registry[kind].obj
                module = mod_cls.from_spec(module_spec, context)
            else:
                # If module_spec is already a BaseSpec, use its kind
                mod_cls = registry[module_spec.kind].obj
                module = mod_cls.from_spec(module_spec, context)
            modules.append(module)

        return cls(modules)
