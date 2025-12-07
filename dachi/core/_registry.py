# 1st party
from __future__ import annotations
import inspect
from typing import Union, Callable, Any, Dict, Optional, Union, List
import typing as t
from typing import Union
import inspect
from ._base import to_kind

T = t.TypeVar("T")


class RegistryEntry(t.Generic[T]):
    def __init__(self,
                 obj: T,
                 obj_type: str,
                 tags: Dict[str, Any],
                 package: str,
                 description: Optional[str] = None):
        self.obj = obj
        self.type = obj_type
        self.tags = tags
        self.package = package
        self.description = description


# ============================================================================
# Module Field Descriptors
# ============================================================================

class Registry(t.Generic[T]):
    """Registry for BaseModule classes and functions.
    Allows registration, filtering, and retrieval of objects
    by name, type, tags, and package.
    """
    def __init__(self):
        self._entries: Dict[str, RegistryEntry[T]] = {}

    def register_item(
        self,
        item: T,
        name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None
    ) -> None:
        """
        Register an object in the registry.

        Args:
            key: The name of the object.
            obj: The object to register.
            obj_type: The type of the object.
            tags: A dictionary of tags associated with the object.
            package: The package of the object.
            description: A description of the object.
        """
        return self.register(
            name=name,
            tags=tags,
            description=description
        )(item)

    def register(
        self,
        name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None
    ) -> Callable[[Union[type, Callable]], Union[type, Callable]]:
        """
        Register a Module in the registry.

        Args:
            name: The name of the module.
            tags: A dictionary of tags associated with the module.
            description: A description of the module.
        """
        def decorator(obj: Union[type, Callable]) -> Union[type, Callable]:

            key: str = name or to_kind(obj)
            obj_type: str = "class" if inspect.isclass(obj) else "function"
            module: str = obj.__module__

            if key in self._entries:
                print(f"Warning: Overwriting existing entry '{key}'")

            self._entries[key] = RegistryEntry[T](
                obj=obj,
                obj_type=obj_type,
                tags=tags or {},
                package=module,
                description=description
            )
            return obj
        return decorator

    def filter(self,
               obj_type: Optional[str] = None,
               tags: Optional[Dict[str, Any]] = None,
               package: Optional[str] = None) -> Dict[str, RegistryEntry[T]]:
        """
        Filter the registry entries based on the given criteria.
        Args:
            obj_type: The type of the object to filter by.
            tags: A dictionary of tags to filter by.
            package: The package to filter by.

        Returns:
            A dictionary of matching registry entries.
        """
        results: Dict[str, RegistryEntry[T]] = {}
        for k, v in self._entries.items():
            if obj_type and v.type != obj_type:
                continue
            if tags and not all(item in v.tags.items() for item in tags.items()):
                continue
            if package and v.package != package:
                continue
            results[k] = v
        return results

    def __getitem__(self, key: Union[str, List[str]]) -> Union[RegistryEntry[T], Dict[str, RegistryEntry[T]]]:
        """Retrieve a single entry by key or a list of entries by keys."""
        try: 
            if isinstance(key, list):
                return {k: self._entries[k] for k in key if k in self._entries}
            return self._entries[key]
        except KeyError:
            raise KeyError(f"Registry entry '{key}' not found. Available entries: {list(self._entries.keys())}")

    def deregister(self, key: str) -> None:
        """Remove an entry from the registry by key."""
        if key in self._entries:
            del self._entries[key]

    def list_entries(self) -> List[str]:
        """
        List all registered entries in the registry.
        """
        return list(self._entries.keys())

    def list_types(self) -> List[str]:
        """
        List all unique types of registered entries.
        """
        return list(set(v.type for v in self._entries.values()))

    def list_packages(self) -> List[str]:
        """
        List all unique packages of registered entries.
        """
        return list(set(v.package for v in self._entries.values()))

    def list_tags(self) -> List[str]:
        """
        List all unique tags of registered entries.
        """
        tags: set[str] = set()
        for v in self._entries.values():
            tags.update(v.tags.keys())
        return list(tags)
    
    def __call__(self,
                 name: Optional[str] = None,
                 tags: Optional[Dict[str, Any]] = None,
                 description: Optional[str] = None) -> Callable[[Union[type, Callable]], Union[type, Callable]]:
        """
        Create a decorator to register a module.
        Args:
            name: The name of the module.
            tags: A dictionary of tags associated with the module.
            description: A description of the module.

        Returns:
            A decorator that registers the module.
        """
        return self.register(
            name=name,
            tags=tags,
            description=description
        )
