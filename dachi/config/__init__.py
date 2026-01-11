from __future__ import annotations

from typing import Any, Type, TypeVar

from ._accessor import ConfigAccessor
from ._models import DachiConfig

T = TypeVar("T", bound=DachiConfig)

_global_config: ConfigAccessor | None = None
_config_class: Type[DachiConfig] = DachiConfig


def _build_config_class(base_cls: Type[T], yaml_file: str | None) -> Type[T]:
    if yaml_file is None:
        return base_cls
    config_dict: dict[str, Any] = dict(base_cls.model_config)
    return type(
        f"{base_cls.__name__}WithYaml",
        (base_cls,),
        {"_yaml_file_override": yaml_file, "model_config": config_dict},
    )


def init_config(config_path: str | None = None, config_class: Type[T] | None = None) -> ConfigAccessor:
    global _global_config, _config_class
    if _global_config is not None:
        raise RuntimeError("Config already initialized. Call reset_config() first.")
    base_class: Type[T]
    if config_class is not None:
        if not issubclass(config_class, DachiConfig):
            raise TypeError("config_class must extend DachiConfig")
        _config_class = config_class
        base_class = config_class
    else:
        base_class = _config_class  # type: ignore[assignment]
    effective_class = _build_config_class(base_class, config_path)
    config_instance = effective_class()
    accessor = ConfigAccessor(config_instance)
    accessor.freeze()
    _global_config = accessor
    return accessor


def get_config() -> ConfigAccessor:
    global _global_config
    if _global_config is None:
        _global_config = init_config()
    return _global_config


def reset_config() -> None:
    global _global_config, _config_class
    _global_config = None
    _config_class = DachiConfig


class _ConfigProxy:
    def __getattr__(self, name: str) -> Any:
        return getattr(get_config(), name)

    def __getitem__(self, key: str) -> Any:
        return get_config()[key]


config = _ConfigProxy()


__all__ = [
    "config",
    "init_config",
    "get_config",
    "reset_config",
    "DachiConfig",
    "ConfigAccessor",
]
