from __future__ import annotations

# 1st party
import typing as t
from typing import Iterator, Optional

# Local
from ._module import Module, MODULE
from dachi.utils.store._store import is_primitive
import pydantic


class ModuleList(Module, t.Generic[MODULE]):
    """
    A list-like container whose elements are themselves `BaseModule`
    instances.  Works seamlessly with the new serialization / dedup rules.
    """
    vals: list[MODULE] = pydantic.Field(default_factory=list)

    # Positive test: len reflects number added
    def __len__(self) -> int:  
        """
        Returns the number of modules in the list.
        """
        return len(self.vals)

    # Positive test: order preserved
    def __iter__(self) -> Iterator[MODULE]:  
        """
        Returns an iterator over the modules in the list.
        """
        return iter(self.vals)

    def __getitem__(self, idx: int) -> MODULE:  # Edge test: negative index ok
        return self.vals[idx]

    def __setitem__(self, idx: int, value: MODULE):
        if not isinstance(value, Module):
            raise TypeError("ModuleList accepts only BaseModule instances")
        if idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds.")
        # unregister old, register new
        self.vals[idx] = value

    def items(self) -> Iterator[str | int, MODULE]:
        for idx, module in enumerate(self.vals):
            yield idx, module

    # public API – intentionally *append‑only*
    def append(self, module: MODULE):

        if not isinstance(module, Module):
            raise TypeError("ModuleList accepts only BaseModule instances")
        # key = str(len(self.items))
        self.vals.append(module)
    
    @property
    def aslist(self) -> list[MODULE]:
        """
        Expose the internal list of modules.
        This is useful for iterating over the modules directly.
        """
        return [*self.vals]
    
    def modules(
        self,
        *,
        recurse: bool = True,
        f: t.Callable[['Module'], bool] | None = None,
        _seen: t.Optional[set[int]] = None,
        _skip_self: bool = False
    ):
        _seen = _seen or set()
        yield from super().modules(recurse=recurse, f=f, _seen=_seen, _skip_self=_skip_self)

        for module in self.vals:
            
            if not isinstance(module, Module):
                continue
            if id(module) in _seen:
                continue
            _seen.add(id(module))
            if f is None or f(module):
                yield module
            if recurse:
                yield from module.modules(recurse=recurse, f=f, _seen=_seen, _skip_self=True)
    
    def named_modules(
        self,
        *,
        recurse: bool = True,
        prefix: str = "",
        f: t.Callable[['Module'], bool] | None = None,
        _seen: t.Optional[set[int]] = None,
        _skip_self: bool = False
    ):
        if _seen is None:
            _seen = set()
        yield from super().named_modules(recurse=recurse, prefix=prefix, f=f, _seen=_seen, _skip_self=_skip_self)

        for idx, module in enumerate(self.vals):
            if not isinstance(module, Module):
                continue
            child_prefix = f"{prefix}{idx}."
            if id(module) in _seen:
                continue
            _seen.add(id(module))
            if f is None or f(module):
                yield child_prefix.rstrip("."), module
            if recurse:
                yield from module.named_modules(recurse=recurse, prefix=child_prefix, f=f, _seen=_seen)

    def state_dict(self, *, recurse = True, train = True, runtime = True):
        d = super().state_dict(recurse=recurse, train=train, runtime=runtime)
        for module in self.vals:
            d["items." + module._module_key] = module.state_dict(recurse=recurse, train=train, runtime=runtime)
        return d
    
    def load_state_dict(self, sd, *, recurse = True, train = True, runtime = True, strict = True):
        super().load_state_dict(sd, recurse=recurse, train=train, runtime=runtime, strict=strict)
        for module in self.vals:
            module.load_state_dict(
                sd.get("items." + module._module_key, {}),
                recurse=recurse, train=train, runtime=runtime, strict=strict
            )
    
    def __contains__(self, item: t.Any) -> bool:
        return item in self.vals


class ModuleDict(Module, t.Generic[MODULE]):
    """
    A dict-like container whose values are themselves `BaseModule`
    instances. Keys must be strings.
    """
    # __spec_hooks__: ClassVar[t.List[str]] = ["items"]
    # items: InitVar[dict[str, BaseModule | t.Any]] = {}
    vals: dict[str | int, MODULE] = pydantic.Field(default_factory=dict)

    def __getitem__(self, key: str) -> MODULE:
        """Get an item from the module dict.

        Args:
            key (str): The key of the item to retrieve.

        Returns:
            V: The item associated with the key.
        """
        return self.vals[key]

    def __setitem__(self, key: str, val: MODULE):
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
        self.vals[key] = val

    def __iter__(self):
        return iter(self.vals)

    def __len__(self):
        return len(self.vals)

    def __contains__(self, item: t.Any) -> bool:
        return item in self.vals

    def __delattr__(self, name):
        
        if isinstance(self.vals[name], Module):
            del self._modules[name]
        del self.vals[name]

    def keys(self):
        return self.vals.keys()

    def values(self):
        return self.vals.values()

    def items(self):
        return self.vals.items()
    
    def get(self, key: str, default: Optional[MODULE] = None) -> Optional[MODULE]:
        """Get an item from the module dict, returning a default if not found.
        Args:
            key (str): The key of the item to retrieve.
            default (Optional[V], optional): The default value to return if the key is not found
                Defaults to None.
        Returns:
            Optional[V]: The item associated with the key, or the default value if not found.
        """
        return self.vals.get(key, default)

    def modules(self, *, recurse = True, f = None, _seen = None, _skip_self = False):

        if _seen is None:
            _seen = set()

        yield from super().modules(recurse=recurse, f=f, _seen=_seen, _skip_self=_skip_self)
        for module in self.vals.values():
            if not isinstance(module, Module):
                continue
            if id(module) in (_seen := _seen or set()):
                continue
            _seen.add(id(module))
            if f is None or f(module):
                yield module
            if recurse:
                yield from module.modules(recurse=recurse, f=f, _seen=_seen, _skip_self=True)
    
    def named_modules(
        self,
        *,
        recurse: bool = True,
        prefix: str = "",
        f: t.Callable[['Module'], bool] | None = None,
        _seen: t.Optional[set[int]] = None,
        _skip_self: bool = False
    ):
        
        if _seen is None:
            _seen = set()
        yield from super().named_modules(recurse=recurse, prefix=prefix, f=f, _seen=_seen, _skip_self=_skip_self)

        for name, module in self.vals.items():
            child_prefix = f"{prefix}{name}."
            if not isinstance(module, Module):
                continue
            if id(module) in (_seen := _seen or set()):
                continue
            _seen.add(id(module))
            if f is None or f(module):
                yield child_prefix.rstrip("."), module
            if recurse:
                yield from module.named_modules(recurse=recurse, prefix=child_prefix, f=f, _seen=_seen, _skip_self=True)
