"""Argument model generation for Process classes.

This module provides utilities for automatically generating Pydantic models
from Python function signatures, supporting all parameter kinds including
positional-only, keyword-only, *args, and **kwargs.
"""

import inspect
import typing as t
import dataclasses
import pydantic


@dataclasses.dataclass
class Ref:
    """Reference to the output of another process"""
    name: str


V = t.TypeVar('V')


class KWOnly(pydantic.BaseModel, t.Generic[V]):
    """Wrapper for keyword-only arguments"""
    data: V


class PosArgs(pydantic.BaseModel, t.Generic[V]):
    """Wrapper for VAR_POSITIONAL (*args) arguments"""
    data: t.List[V]


class KWArgs(pydantic.BaseModel, t.Generic[V]):
    """Wrapper for VAR_KEYWORD (**kwargs) arguments"""
    data: t.Dict[str, V]


class BaseArgs(pydantic.BaseModel):
    """Base class for generated argument models"""

    def get_args(self, by: t.Dict[str, t.Any]) -> t.Tuple[t.List[t.Any], t.Dict[str, t.Any]]:
        """Get the args and kwargs with references resolved

        Args:
            by: The mapping to resolve references from

        Returns:
            Tuple of (positional_args, keyword_args)
        """
        args = []
        pos_args = []
        kw_only = {}
        kwargs = {}

        for k, _ in self.model_fields.items():
            value = getattr(self, k)

            if isinstance(value, PosArgs):
                pos_args = value.data
            elif isinstance(value, KWOnly):
                if isinstance(value.data, Ref):
                    value = by[value.data.name]
                else:
                    value = value.data
                kw_only[k] = value
            elif isinstance(value, KWArgs):
                kwargs = value.data
            elif isinstance(value, Ref):
                args.append(by[value.name])
            else:
                args.append(value)

        return (
            [*args, *pos_args],
            {**kw_only, **kwargs}
        )

    @classmethod
    def build(cls, *args, **kwargs) -> 'BaseArgs':
        """Build an argument model instance from args and kwargs.
        Args:
            args: Positional arguments
            kwargs: Keyword arguments
        Returns:
            An instance of the argument model
        """
        init_kwargs = {}
        field_items = list(cls.model_fields.items())

        def is_wrapper_type(annotation, wrapper_cls):
            """Check if annotation is a wrapper type (KWOnly, PosArgs, KWArgs)"""
            try:
                return isinstance(annotation, type) and issubclass(annotation, wrapper_cls)
            except TypeError:
                return False

        # Count regular positional fields (not KWOnly, PosArgs, or KWArgs)
        num_pos_args = sum(
            1 for _, field_info in field_items
            if not (is_wrapper_type(field_info.annotation, KWOnly) or
                   is_wrapper_type(field_info.annotation, PosArgs) or
                   is_wrapper_type(field_info.annotation, KWArgs))
        )

        # Check for special field types
        has_pos_args = any(is_wrapper_type(f.annotation, PosArgs) for _, f in field_items)
        has_kwargs = any(is_wrapper_type(f.annotation, KWArgs) for _, f in field_items)

        field_names = {name for name, _ in field_items}

        # Assign positional args to regular fields in order
        pos_field_idx = 0
        for name, field_info in field_items:
            if not (is_wrapper_type(field_info.annotation, KWOnly) or
                   is_wrapper_type(field_info.annotation, PosArgs) or
                   is_wrapper_type(field_info.annotation, KWArgs)):
                if pos_field_idx < len(args):
                    init_kwargs[name] = args[pos_field_idx]
                    pos_field_idx += 1

        # Handle excess positional args (*args)
        if has_pos_args:
            pos_args_name = next(n for n, f in field_items if is_wrapper_type(f.annotation, PosArgs))
            posargs_annotation = cls.model_fields[pos_args_name].annotation
            init_kwargs[pos_args_name] = posargs_annotation(data=list(args[num_pos_args:]))

        # Handle keyword arguments
        for k, v in kwargs.items():
            if k in field_names:
                field_info = cls.model_fields[k]
                if is_wrapper_type(field_info.annotation, KWOnly):
                    # Keyword-only arg: wrap in the specific KWOnly subclass
                    init_kwargs[k] = field_info.annotation(data=v)
                elif not is_wrapper_type(field_info.annotation, KWArgs):
                    # Regular field provided as kwarg
                    if k in init_kwargs:
                        raise TypeError(f"Got multiple values for argument '{k}'")
                    init_kwargs[k] = v

        # Handle excess keyword args (**kwargs)
        if has_kwargs:
            kwargs_name = next(n for n, f in field_items if is_wrapper_type(f.annotation, KWArgs))
            kwargs_annotation = cls.model_fields[kwargs_name].annotation
            true_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in field_names
            }
            init_kwargs[kwargs_name] = kwargs_annotation(data=true_kwargs)

        return cls(**init_kwargs)

# TODO: Determine how to handle arguments that start with _ or __
# Currently, we will not be able to create a model from these
def func_arg_model(cls: type, cls_f, with_ref: bool=False) -> type[pydantic.BaseModel]:
    """Build a Pydantic model from a function's signature.

    Inspects the function signature and creates a Pydantic model with fields
    corresponding to the function's parameters. Handles all Python parameter
    kinds including positional-only, keyword-only, *args.

    Args:
        cls: The class containing the method (used for model naming)
        cls_f: The function/method to inspect
        with_ref: If True, wrap all annotations in Union[Ref, T] to allow
                 references to other process outputs

    Returns:
        A Pydantic BaseModel subclass with fields matching the function signature

    Parameter Handling:
        - POSITIONAL_ONLY: Regular fields (x: int, /)
        - POSITIONAL_OR_KEYWORD: Regular fields (x: int = 5)
        - KEYWORD_ONLY: Wrapped in KWOnly (*, x: int)
        - VAR_POSITIONAL: Wrapped in PosArgs (*args)
        - VAR_KEYWORD: Wrapped in KWArgs (**kwargs)
    """
    sig = inspect.signature(cls_f)
    fields: dict[str, tuple[t.Any, t.Any]] = {}

    positional_only = {}
    var_positional = None
    # defines the name and the default value
    positional_or_keyword = {}
    keyword_only = {}
    var_keyword = None

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        anno = param.annotation

        # Handle missing annotations
        if with_ref:
            if anno == inspect._empty:
                anno = t.Any
            anno = t.Union[Ref, anno]
        elif anno == inspect._empty:
            anno = t.Any

        default = param.default if param.default is not inspect._empty else ...

        if param.kind == param.POSITIONAL_OR_KEYWORD:
            positional_or_keyword[name] = (anno, default)
        elif param.kind == param.POSITIONAL_ONLY:
            positional_only[name] = (anno, default)
        elif param.kind == param.VAR_POSITIONAL:
            var_positional = (anno, name)
        elif param.kind == param.KEYWORD_ONLY:
            keyword_only[name] = (anno, default)
        elif param.kind == param.VAR_KEYWORD:
            var_keyword = (anno, name)

    # Build fields dict for create_model
    # Positional-only params come first
    for name, (anno, default) in positional_only.items():
        fields[name] = (anno, default)

    # Then positional-or-keyword params
    for name, (anno, default) in positional_or_keyword.items():
        fields[name] = (anno, default)

    # Keyword-only params wrapped in KWOnly
    for name, (anno, default) in keyword_only.items():
        wrapped_anno = KWOnly[anno]
        if default is ...:
            # Required field
            fields[name] = (wrapped_anno, ...)
        else:
            # Field with default
            fields[name] = (wrapped_anno, pydantic.Field(default_factory=lambda v=default, a=anno: KWOnly[a](data=v)))

    # VAR_POSITIONAL (*args) wrapped in PosArgs
    if var_positional is not None:
        anno, name = var_positional
        fields[name] = (PosArgs[anno], pydantic.Field(default_factory=lambda a=anno: PosArgs[a](data=[])))

    # VAR_KEYWORD (**kwargs) wrapped in KWArgs
    if var_keyword is not None:
        anno, name = var_keyword
        fields[name] = (KWArgs[anno], pydantic.Field(default_factory=lambda a=anno: KWArgs[a](data={})))

    model_name = f"{cls.__name__}Args"
    created = pydantic.create_model(
        model_name,
        __base__=BaseArgs,
        **fields
    )  # type: ignore[call-arg]

    return created
