from __future__ import annotations
import typing as t
from pydantic import BaseModel
from ._base4 import BaseModule, Param, State, Shared, BuildContext, BaseSpec, registry  # adjust import path

from pydantic import BaseModel, Field, ConfigDict, create_model, field_validator

from typing import Callable, Any, Dict, Optional, Union, List


from pydantic import create_model, field_validator

V_co = t.TypeVar("V_co", bound=BaseModule, covariant=True)


class ModuleList(BaseModule, t.Generic[V_co]):
    """
    A sequential container of BaseItem objects behaving like a Python list.
    Each child is registered under a stringified index: `"0"`, `"1"`, …
    """

    # --------------------------------------------------
    # constructor
    # --------------------------------------------------
    def __init__(self, modules: t.Iterable[V_co]):
        modules = list(modules)
        self._parameters: dict[str, Param] = {}  # registry for parameters
        self._states: dict[str, State] = {}  # registry for states
        if not modules:
            raise ValueError("ItemList cannot be empty")
        if not all(isinstance(i, BaseModule) for i in modules):
            raise TypeError("All elements must be BaseItem instances.")
        self._module_list: list[V_co] = []
        self._modules: dict[str, V_co] = {}  # registry for child modules
        for itm in modules:
            self.append(itm)   # register each

    # --------------------------------------------------
    # python list interface
    # --------------------------------------------------
    def __getitem__(self, idx: int) -> V_co:
        return self._module_list[idx]

    def __setitem__(self, idx: int, value: V_co):
        if not isinstance(value, BaseModule):
            raise TypeError("Item must be a BaseItem")
        # deregister old
        old_name = str(idx)
        if old_name in self._module_dict():
            del self._module_dict()[old_name]
        # register new
        self._module_list[idx] = value
        self.register_module(old_name, value)

    def __len__(self):
        return len(self._module_list)

    def __iter__(self):
        return iter(self._module_list)

    def append(self, item: V_co):
        if not isinstance(item, BaseModule):
            raise TypeError("Item must be a BaseItem")
        name = str(len(self._module_list))
        self._module_list.append(item)
        self.register_module(name, item)

    # --------------------------------------------------
    # helpers
    # --------------------------------------------------
    def _module_dict(self) -> dict[str, BaseModule]:
        """Shortcut to the internal child registry."""
        return self._modules   # already stored by index order

    # --------------------------------------------------
    # spec / schema
    # --------------------------------------------------
    @classmethod
    def schema(cls) -> t.Type[BaseSpec]:
        """
        Return a *single* spec model describing this ModuleList.
        The dynamic Pydantic model looks like:
            class ModuleListSpec(BaseSpec):
                modules: list[ElemSpec]
        """
        if getattr(cls, "__spec__", None) is not None:
            return cls.__spec__

        # Resolve element type
        try:
            elem_type = cls.__orig_bases__[0].__args__[0]
        except Exception:
            raise TypeError("ModuleList must be parametrised, e.g. ModuleList[MyMod]")

        if not issubclass(elem_type, BaseModule):
            raise TypeError("Element type must inherit from BaseModule")

        elem_spec_type = elem_type.schema()

        cls.__spec__ = create_model(  # directly create the dynamic model
            f"{cls.__name__}Spec",
            __base__=BaseSpec,
            modules=(list[elem_spec_type], ...)
        )
        return cls.__spec__

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
