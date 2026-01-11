from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ConfigNode:
    def __init__(self, value: Any, frozen: bool = False):
        object.__setattr__(self, "_value", value)
        object.__setattr__(self, "_frozen", frozen)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._value, name)
        return self._wrap(attr)

    def __getitem__(self, key: str) -> Any:
        attr = getattr(self._value, key)
        return self._wrap(attr)

    def __setattr__(self, name: str, value: Any) -> None:
        if object.__getattribute__(self, "_frozen"):
            raise RuntimeError("Configuration is read-only")
        setattr(self._value, name, value)

    def __setitem__(self, key: str, value: Any) -> None:
        if object.__getattribute__(self, "_frozen"):
            raise RuntimeError("Configuration is read-only")
        setattr(self._value, key, value)

    def _wrap(self, attr: Any) -> Any:
        if isinstance(attr, BaseModel):
            return ConfigNode(attr, object.__getattribute__(self, "_frozen"))
        return attr


class ConfigAccessor(ConfigNode):
    def __init__(self, value: Any):
        super().__init__(value, False)

    def freeze(self) -> None:
        object.__setattr__(self, "_frozen", True)
