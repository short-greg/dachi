from __future__ import annotations

import sys
import types
from typing import Any, TypeVar, get_args, get_origin

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_raw_annotation(defining_cls: type[BaseModel], raw: Any) -> Any:
    """
    Resolve a raw annotation value in the defining class's module/namespace.

    - If it's not a string, just return it as-is.
    - If it's a string, eval it in the module + class namespace.
    - On any error, fall back to the raw value (don't crash).
    """
    if raw is None:
        return None

    if not isinstance(raw, str):
        return raw

    module_globals = vars(sys.modules[defining_cls.__module__])
    localns = dict(vars(defining_cls))

    try:
        return eval(raw, module_globals, localns)
    except Exception:
        # NameError, SyntaxError, etc. – keep the original string
        return raw


def _get_typevar_map_for_model(model_cls: type[BaseModel]) -> dict[TypeVar, Any]:
    """
    For a *concrete* generic model, derive mappings like {T: bool}.

    We first look at the model's own __pydantic_generic_metadata__.
    If that doesn't give us anything (e.g. for a non-generic subclass of
    a concrete generic base), we fall back to inspecting its BaseModel bases.
    """
    tv_map: dict[TypeVar, Any] = {}

    # 1) Try the model itself, e.g. HasParam[bool]
    md = getattr(model_cls, "__pydantic_generic_metadata__", None)
    if md:
        origin = md.get("origin")
        args = tuple(md.get("args") or ())
        if origin and args:
            origin_md = getattr(origin, "__pydantic_generic_metadata__", None)
            if origin_md:
                params = tuple(origin_md.get("parameters") or ())
                tv_map.update(dict(zip(params, args)))
                if tv_map:
                    return tv_map

    # 2) Fallback: check generic BaseModel bases, e.g. class Child(HasParam[bool])
    for base in model_cls.__bases__:
        bmd = getattr(base, "__pydantic_generic_metadata__", None)
        if not bmd:
            continue

        origin = bmd.get("origin")
        args = tuple(bmd.get("args") or ())
        if not origin or not args:
            continue

        origin_md = getattr(origin, "__pydantic_generic_metadata__", None)
        if not origin_md:
            continue

        params = tuple(origin_md.get("parameters") or ())
        if not params:
            continue

        tv_map.update(dict(zip(params, args)))
        if tv_map:
            return tv_map

    return tv_map


def _lookup_typevar(tv: TypeVar, tv_map: dict[TypeVar, Any]) -> Any:
    """
    Lookup a TypeVar in tv_map.

    First try identity (tv_map[tv]).
    If that fails, fall back to matching by TypeVar name, to handle cases
    where the same logical TypeVar is re-used or re-imported.
    """
    if tv in tv_map:
        return tv_map[tv]

    # Fallback: match by name
    for k, v in tv_map.items():
        if isinstance(k, TypeVar) and getattr(k, "__name__", None) == getattr(tv, "__name__", None):
            return v

    return tv  # unresolved


def _rebuild_union(origin: Any, args: tuple[Any, ...]) -> Any:
    """
    Rebuild a typing / PEP 604 union from new args.
    """
    import typing as _typing

    # typing.Union[...]
    if origin is _typing.Union:
        return _typing.Union[args]

    # PEP 604 union (T | U | ...)
    if origin is types.UnionType:  # Python 3.10+
        it = iter(args)
        try:
            first = next(it)
        except StopIteration:
            raise TypeError("Cannot build union with no args")
        acc = first
        for a in it:
            acc = acc | a
        return acc

    # Fallback: try normal reconstruction
    if len(args) == 1:
        return origin[args[0]]
    return origin[args]


def _substitute_typevars(tp: Any, tv_map: dict[TypeVar, Any]) -> Any:
    """
    Substitute TypeVars in a type according to tv_map.

    Handles:
    - direct TypeVar
    - Pydantic generic models (MyModel, MyModel[T], MyModel[int])
    - standard typing generics (list[T], dict[str, T], ...)
    - PEP 604 unions (T | None)
    """
    from typing import TypeVar as _TypeVar

    # Direct TypeVar: replace with mapped value if present
    if isinstance(tp, _TypeVar):
        return _lookup_typevar(tp, tv_map)

    # Pydantic generic model class (e.g. Param, Param[T], Param[Foo | None])
    try:
        is_model_sub = isinstance(tp, type) and issubclass(tp, BaseModel)
    except TypeError:
        is_model_sub = False

    if is_model_sub:
        md = getattr(tp, "__pydantic_generic_metadata__", None)
        if not md:
            return tp

        origin = md.get("origin") or tp
        params = tuple(md.get("parameters") or ())
        args = tuple(md.get("args") or ())

        # Case 1: already specialized (Param[Something]) – substitute inside args
        if args:
            new_args = tuple(_substitute_typevars(a, tv_map) for a in args)
            if new_args == args:
                return tp
            if len(new_args) == 1:
                return origin[new_args[0]]
            return origin[tuple(new_args)]

        # Case 2: unspecialized generic (Param) – specialize using tv_map
        if params:
            new_args = tuple(_lookup_typevar(p, tv_map) for p in params)
            # If nothing actually changed, keep the original
            if all(isinstance(a, _TypeVar) and a in params for a in new_args):
                return tp
            if len(new_args) == 1:
                return origin[new_args[0]]
            return origin[tuple(new_args)]

        return tp

    # Standard typing generics / unions
    origin = get_origin(tp)
    if origin is None:
        return tp

    args = get_args(tp)
    if not args:
        return tp

    new_args = tuple(_substitute_typevars(a, tv_map) for a in args)
    if new_args == args:
        return tp

    try:
        # Unions need special handling
        if origin is types.UnionType:
            return _rebuild_union(origin, new_args)
        import typing as _typing
        if origin is _typing.Union:
            return _rebuild_union(origin, new_args)

        # Normal typing generics: origin[args]
        if len(new_args) == 1:
            return origin[new_args[0]]
        return origin[tuple(new_args)]
    except TypeError:
        # Some weird alias we can't reconstruct – keep original
        return tp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_all_private_attr_annotations(model_cls: type[BaseModel]) -> dict[str, Any]:
    """
    Return {name: resolved_annotation} for all *private* attributes on a
    Pydantic v2 model class, including those declared on base classes.

    Behaviour:
    - Uses model_cls.__private_attributes__ for the set of private names
      (this already includes inherited private attributes).
    - Walks the MRO to find the class that actually defines each annotation.
    - Resolves string annotations in that class's module/namespace.
    - Builds a TypeVar -> concrete type mapping from the model's generic
      metadata (and, if necessary, its generic BaseModel bases).
    - Substitutes TypeVars even inside unions
      (e.g. Param[T | None] -> Param[Foo | None]).
    - Never raises due to unresolved names; falls back to raw annotations.

    Caveats:
    - If the model itself is still generic (no concrete args), you'll still see
      TypeVars in the result; there's nothing to substitute yet.
    - If an annotation refers to a name that doesn't exist in the module/class
      namespace, it will stay as a string.
    """
    private_attrs = getattr(model_cls, "__private_attributes__", {})
    if not private_attrs:
        return {}

    tv_map = _get_typevar_map_for_model(model_cls)
    result: dict[str, Any] = {}

    for name in private_attrs.keys():
        raw = None
        defining_cls: type[BaseModel] | None = None

        # Find the class in the MRO that actually defines the annotation
        for cls in model_cls.__mro__:
            anns = getattr(cls, "__annotations__", {})
            if name in anns:
                raw = anns[name]
                defining_cls = cls
                break

        if defining_cls is None:
            result[name] = None
            continue

        # 1) Resolve any string / forward-ref style annotation
        resolved = _resolve_raw_annotation(defining_cls, raw)
        # 2) Substitute TypeVars using this model's concrete mapping
        print('Resolved before substitution: ', resolved)
        resolved = _substitute_typevars(resolved, tv_map)
        print('Resolved after substitution: ', resolved)

        result[name] = resolved

    return result
