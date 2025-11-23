from __future__ import annotations

# 1st party
import typing as t
import types
from typing import TypeVar, Iterator, Optional

# Local
from ._base import Module
from dachi.utils import is_primitive

V_co = t.TypeVar("V_co", bound=Module, covariant=True)
V = t.TypeVar("V", bound=Module)
T = TypeVar("T", bound=Module)

import pydantic


class ModuleList(Module, t.Generic[V]):
    """
    A list-like container whose elements are themselves `BaseModule`
    instances.  Works seamlessly with the new serialization / dedup rules.
    """
    items: list[V] = pydantic.Field(default_factory=list)

    # Positive test: len reflects number added
    def __len__(self) -> int:  
        return len(self.items)

    # Positive test: order preserved
    def __iter__(self) -> Iterator[V]:  
        return iter(self.items)

    def __getitem__(self, idx: int) -> V:  # Edge test: negative index ok
        return self.items[idx]

    def __setitem__(self, idx: int, value: V):
        if not isinstance(value, Module):
            raise TypeError("ModuleList accepts only BaseModule instances")
        if idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds.")
        # unregister old, register new
        old_key = str(idx)
        del self._modules[old_key]
        self.items[idx] = value

    # public API – intentionally *append‑only*
    def append(self, module: V):

        if not isinstance(module, Module):
            raise TypeError("ModuleList accepts only BaseModule instances")
        # key = str(len(self.items))
        self.items.append(module)
    
    @property
    def aslist(self) -> list[V]:
        """
        Expose the internal list of modules.
        This is useful for iterating over the modules directly.
        """
        return [*self.items]
    
    def modules(self, *, recurse = True, f = None):
        yield from super().modules(recurse=recurse, f=f)

        if recurse:
            for module in self.items:
                yield from module.modules(recurse=recurse, f=f)
    
    def parameters(self, *, recurse = True, _seen = None, with_annotations = False):
        if _seen is None:
            _seen = set()
        
        if id(self) in _seen:
            return
        _seen.add(id(self))

        for param in super().parameters(recurse=recurse, _seen=_seen, with_annotations=with_annotations):
            yield param
        for module in self.items:
            for param in module.parameters(recurse=recurse, _seen=_seen, with_annotations=with_annotations):
                yield param

    def named_parameters(self, *, recurse = True, _seen = None, prefix = ""):
        if _seen is None:
            _seen = set()
        if id(self) in _seen:
            return
        _seen.add(id(self))
        yield from super().named_parameters(recurse=recurse, _seen=_seen, prefix=prefix)
        for i, module in enumerate(self.items):
            child_prefix = f"{prefix}{i}."
            for name, param in module.named_parameters(recurse=recurse, _seen=_seen, prefix=child_prefix):
                yield name, param

    def children(self):
        """Immediate child modules (non-recursive)."""
        yield from super().children()
        for module in self.items:
            yield module
    
    def named_children(self):
        """Immediate child modules (non-recursive) as (name, module) pairs."""
        yield from super().named_children()
        for i, module in enumerate(self.items):
            yield str(i), module

    def apply(self, fn, *, recurse = True, include = None):
        super().apply(fn, recurse=recurse, include=include)

        for module in self.items:
            module.apply(fn, recurse=recurse, include=include)

    def state_dict(self, *, recurse = True, train = True, runtime = True):
        d = super().state_dict(recurse=recurse, train=train, runtime=runtime)
        for module in self.items:
            d["items." + module._module_key] = module.state_dict(recurse=recurse, train=train, runtime=runtime)
        return d
    
    def load_state_dict(self, sd, *, recurse = True, train = True, runtime = True, strict = True):
        super().load_state_dict(sd, recurse=recurse, train=train, runtime=runtime, strict=strict)
        for module in self.items:
            module.load_state_dict(
                sd.get("items." + module._module_key, {}),
                recurse=recurse, train=train, runtime=runtime, strict=strict
            )
    

class ModuleDict(Module, t.Generic[V]):
    """
    A dict-like container whose values are themselves `BaseModule`
    instances. Keys must be strings.
    """
    # __spec_hooks__: ClassVar[t.List[str]] = ["items"]
    # items: InitVar[dict[str, BaseModule | t.Any]] = {}
    items: dict[str | int, V] = pydantic.Field(default_factory=dict)

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
        
        if not isinstance(val, Module) and not is_primitive(val):
            raise TypeError("Values must be Module instances or primitives")
        self.items[key] = val
        if isinstance(val, Module):
            self.register_module(key, val)

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)
    
    def __delattr__(self, name):
        
        if isinstance(self.items[name], Module):
            del self._modules[name]
        del self.items[name]

    def keys(self):
        return self.items.keys()

    def values(self):
        return self.items.values()

    def items(self):
        return self.items.items()
    
    def get(self, key: str, default: Optional[V] = None) -> Optional[V]:
        """Get an item from the module dict, returning a default if not found.
        Args:
            key (str): The key of the item to retrieve.
            default (Optional[V], optional): The default value to return if the key is not found
                Defaults to None.
        Returns:
            Optional[V]: The item associated with the key, or the default value if not found.
        """
        return self.items.get(key, default)
    
    def parameters(self, *, recurse = True, _seen = None, with_annotations = False):
        for param in super().parameters(recurse=recurse, _seen=_seen, with_annotations=with_annotations):
            yield param
        
        for module in self.items.values():
            if isinstance(module, Module):
                for param in module.parameters(recurse=recurse, _seen=_seen, with_annotations=with_annotations):
                    yield param
    
    def named_parameters(self, *, recurse = True, _seen = None, prefix = ""):

        _seen = _seen or set()

        for param in super().named_parameters(recurse=recurse, _seen=_seen):
            yield param
        for k, module in self.items.items():
            if isinstance(module, Module):
                child_prefix = f"{prefix}{k}."
                for param in module.named_parameters(recurse=recurse, _seen=_seen, prefix=child_prefix):
                    yield param

    def children(self):
        """Immediate child modules (non-recursive)."""
        for module in self.items.values():
            if isinstance(module, Module):
                yield module

    def named_children(self):
        """Immediate child modules (non-recursive) as (name, module) pairs."""
        for k, module in self.items.items():
            if isinstance(module, Module):
                yield k, module

    def state_dict(self, *, recurse = True, train = True, runtime = True):
        d = super().state_dict(recurse=recurse, train=train, runtime=runtime)
        for k, module in self.items.items():
            if isinstance(module, Module):
                d["items." + str(k)] = module.state_dict(recurse=recurse, train=train, runtime=runtime)
        return d
    
    def load_state_dict(self, sd, *, recurse = True, train = True, runtime = True, strict = True):
        super().load_state_dict(sd, recurse=recurse, train=train, runtime=runtime, strict=strict)
        for k, module in self.items.items():
            if isinstance(module, Module):
                module.load_state_dict(
                    sd.get("items." + str(k), {}),
                    recurse=recurse, train=train, runtime=runtime, strict=strict
                )
