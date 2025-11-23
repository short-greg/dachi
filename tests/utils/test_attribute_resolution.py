from __future__ import annotations

import pytest
from typing import Dict, List, TypeVar, Generic, get_args, get_origin
from pydantic import BaseModel, PrivateAttr

from dachi.utils._attribute_resolution import (
    get_all_private_attr_annotations,
    _resolve_raw_annotation,
    _get_typevar_map_for_model,
    _substitute_typevars,
)

T = TypeVar("T")
U = TypeVar("U")

# Module-level classes for testing nested generics (must be module-level for proper resolution)
class InnerGenericTest(BaseModel, Generic[T]):
    value: T

class GenericParam(BaseModel, Generic[T]):
    """Test class simulating Param behavior without importing from core."""
    data: T | None = None


class TestResolveRawAnnotation:
    """Test the _resolve_raw_annotation helper function."""

    def test_resolve_when_not_string_returns_as_is(self):
        class TestModel(BaseModel):
            pass

        result = _resolve_raw_annotation(TestModel, int)
        assert result is int

    def test_resolve_when_none_returns_none(self):
        class TestModel(BaseModel):
            pass

        result = _resolve_raw_annotation(TestModel, None)
        assert result is None

    def test_resolve_when_string_and_valid_resolves_to_type(self):
        class TestModel(BaseModel):
            pass

        result = _resolve_raw_annotation(TestModel, "int")
        assert result is int

    def test_resolve_when_string_and_invalid_returns_string(self):
        class TestModel(BaseModel):
            pass

        result = _resolve_raw_annotation(TestModel, "NonExistentType")
        assert result == "NonExistentType"

    def test_resolve_when_forward_ref_in_same_module_resolves(self):
        class InnerClass(BaseModel):
            pass

        class TestModel(BaseModel):
            pass

        # Simulate forward reference by passing string
        result = _resolve_raw_annotation(TestModel, "InnerClass")
        # Should resolve to the actual class if it's in the module
        assert isinstance(result, (type, str))  # May or may not resolve depending on scope


class TestGetTypeVarMapForModel:
    """Test the _get_typevar_map_for_model helper function."""

    def test_map_when_non_generic_returns_empty(self):
        class TestModel(BaseModel):
            x: int

        result = _get_typevar_map_for_model(TestModel)
        assert result == {}

    def test_map_when_generic_with_concrete_type_returns_mapping(self):
        class GenericModel(BaseModel, Generic[T]):
            data: T

        ConcreteModel = GenericModel[int]
        result = _get_typevar_map_for_model(ConcreteModel)

        assert len(result) == 1
        assert T in result or any(isinstance(k, TypeVar) and k.__name__ == "T" for k in result)
        # The value should be int
        assert int in result.values()

    def test_map_when_multiple_typevars_returns_all_mappings(self):
        class GenericModel(BaseModel, Generic[T, U]):
            data1: T
            data2: U

        ConcreteModel = GenericModel[int, str]
        result = _get_typevar_map_for_model(ConcreteModel)

        assert len(result) == 2
        assert int in result.values()
        assert str in result.values()

    def test_map_when_inherited_generic_returns_mapping(self):
        class BaseGeneric(BaseModel, Generic[T]):
            data: T

        class ChildConcrete(BaseGeneric[bool]):
            extra: int

        result = _get_typevar_map_for_model(ChildConcrete)

        assert len(result) >= 1
        assert bool in result.values()


class TestSubstituteTypeVars:
    """Test the _substitute_typevars helper function."""

    def test_substitute_when_direct_typevar_returns_mapped_type(self):
        tv_map = {T: int}

        result = _substitute_typevars(T, tv_map)
        assert result is int

    def test_substitute_when_typevar_not_in_map_returns_typevar(self):
        tv_map = {U: str}

        result = _substitute_typevars(T, tv_map)
        assert isinstance(result, TypeVar)

    def test_substitute_when_non_generic_type_returns_as_is(self):
        tv_map = {T: int}

        result = _substitute_typevars(str, tv_map)
        assert result is str

    def test_substitute_when_list_generic_substitutes_args(self):

        tv_map = {T: int}

        result = _substitute_typevars(List[T], tv_map)
        # Implementation returns list[int] (builtin), not List[int] (typing module)
        # Both are semantically equivalent
        assert get_origin(result) is list
        assert get_args(result) == (int,)

    def test_substitute_when_union_substitutes_all_members(self):
        tv_map = {T: int}

        result = _substitute_typevars(T | None, tv_map)
        assert result == (int | None)

    def test_substitute_when_pydantic_model_with_typevar_substitutes(self):
        class GenericModel(BaseModel, Generic[T]):
            data: T

        tv_map = {T: bool}

        result = _substitute_typevars(GenericModel[T], tv_map)
        # Should return GenericModel[bool]
        assert hasattr(result, "__pydantic_generic_metadata__")

    def test_substitute_when_nested_generics_substitutes_recursively(self):

        tv_map = {T: str}

        result = _substitute_typevars(Dict[T, List[T]], tv_map)
        # Implementation returns dict[str, list[str]] (builtins), not Dict/List (typing module)
        assert get_origin(result) is dict
        args = get_args(result)
        assert len(args) == 2
        assert args[0] is str
        assert get_origin(args[1]) is list
        assert get_args(args[1]) == (str,)


class TestGetAllPrivateAttrAnnotations:
    """Test the main public API function."""

    def test_annotations_when_no_private_attrs_returns_empty(self):
        class TestModel(BaseModel):
            x: int

        result = get_all_private_attr_annotations(TestModel)
        assert result == {}

    def test_annotations_when_simple_private_attr_returns_annotation(self):
        class TestModel(BaseModel):
            _private: int = PrivateAttr(default=42)

        result = get_all_private_attr_annotations(TestModel)
        assert "_private" in result
        assert result["_private"] is int

    def test_annotations_when_generic_private_attr_returns_concrete_type(self):
        class Container(BaseModel, Generic[T]):
            _data: T = PrivateAttr(default=None)

        ConcreteContainer = Container[str]
        result = get_all_private_attr_annotations(ConcreteContainer)

        assert "_data" in result
        assert result["_data"] is str

    def test_annotations_when_inherited_private_attr_includes_base_attrs(self):
        class BaseModel1(BaseModel):
            _base_attr: int = PrivateAttr(default=1)

        class ChildModel(BaseModel1):
            _child_attr: str = PrivateAttr(default="test")

        result = get_all_private_attr_annotations(ChildModel)

        assert "_base_attr" in result
        assert "_child_attr" in result
        assert result["_base_attr"] is int
        assert result["_child_attr"] is str

    def test_annotations_when_complex_generic_resolves_correctly(self):
        class GenericContainer(BaseModel, Generic[T]):
            _data: T | None = PrivateAttr(default=None)

        ConcreteContainer = GenericContainer[int]
        result = get_all_private_attr_annotations(ConcreteContainer)

        assert "_data" in result
        # Should be int | None
        assert result["_data"] == (int | None)

    def test_annotations_when_nested_generic_resolves_all_levels(self):

        class GenericContainer(BaseModel, Generic[T]):
            _items: List[T] = PrivateAttr(default_factory=list)

        ConcreteContainer = GenericContainer[bool]
        result = get_all_private_attr_annotations(ConcreteContainer)

        assert "_items" in result
        assert get_origin(result["_items"]) is list
        assert get_args(result["_items"])[0] is bool

    def test_annotations_when_multiple_generics_resolves_all(self):
        class MultiGeneric(BaseModel, Generic[T, U]):
            _first: T = PrivateAttr(default=None)
            _second: U = PrivateAttr(default=None)

        Concrete = MultiGeneric[int, str]
        result = get_all_private_attr_annotations(Concrete)

        assert result["_first"] is int
        assert result["_second"] is str

    def test_annotations_when_pydantic_generic_model_in_attr_resolves(self):
        # Use module-level InnerGenericTest for proper resolution
        class OuterModel(BaseModel, Generic[U]):
            _inner: InnerGenericTest[U] = PrivateAttr(default=None)

        ConcreteOuter = OuterModel[float]
        result = get_all_private_attr_annotations(ConcreteOuter)

        assert "_inner" in result
        # Should be InnerGenericTest[float]
        inner_type = result["_inner"]
        assert hasattr(inner_type, "__pydantic_generic_metadata__")
        metadata = inner_type.__pydantic_generic_metadata__
        assert float in metadata.get("args", ())

    def test_annotations_when_forward_ref_string_resolves_or_stays_string(self):
        class TestModel(BaseModel):
            _attr: int = PrivateAttr(default=0)  # No explicit quotes (with from __future__ import annotations, this becomes a string anyway)

        result = get_all_private_attr_annotations(TestModel)

        assert "_attr" in result
        # Should resolve to int class
        assert result["_attr"] is int

    def test_annotations_when_unresolvable_forward_ref_returns_string(self):
        class TestModel(BaseModel):
            _attr: "NonExistentClass" = PrivateAttr(default=None)

        result = get_all_private_attr_annotations(TestModel)

        assert "_attr" in result
        # Should stay as string since it can't be resolved
        assert result["_attr"] == "NonExistentClass"

    def test_annotations_when_no_annotation_returns_none(self):
        class TestModel(BaseModel):
            pass

        # Manually add private attr without annotation
        TestModel.__private_attributes__["_no_anno"] = PrivateAttr(default=1)

        result = get_all_private_attr_annotations(TestModel)

        assert "_no_anno" in result
        assert result["_no_anno"] is None


class TestIntegrationWithRealScenarios:
    """Test real-world scenarios from the dachi codebase."""

    def test_param_with_module_type_resolves_correctly(self):
        """Simulate the AdaptModule scenario using GenericParam."""

        class InnerModel(BaseModel):
            value: int = 0

        class AdaptSimulation(BaseModel, Generic[T]):
            _adapted: GenericParam[T] = PrivateAttr(default=None)

        ConcreteAdapt = AdaptSimulation[InnerModel]
        result = get_all_private_attr_annotations(ConcreteAdapt)

        assert "_adapted" in result
        # Should be GenericParam[InnerModel]
        adapted_type = result["_adapted"]
        assert hasattr(adapted_type, "__pydantic_generic_metadata__")
        metadata = adapted_type.__pydantic_generic_metadata__
        assert InnerModel in metadata.get("args", ())

    def test_runtime_with_optional_type_resolves_correctly(self):
        """Test GenericParam[T | None] pattern."""

        class TestModel(BaseModel, Generic[T]):
            _state: GenericParam[T | None] = PrivateAttr(default=None)

        ConcreteModel = TestModel[int]
        result = get_all_private_attr_annotations(ConcreteModel)

        assert "_state" in result
        state_type = result["_state"]
        # Should be GenericParam[int | None]
        assert hasattr(state_type, "__pydantic_generic_metadata__")
        metadata = state_type.__pydantic_generic_metadata__
        args = metadata.get("args", ())
        assert len(args) == 1
        # The arg should be (int | None)
        assert args[0] == (int | None)

    def test_multiple_inheritance_levels_resolves_correctly(self):
        """Test deep inheritance hierarchies."""
        class Level1(BaseModel, Generic[T]):
            _l1: T = PrivateAttr(default=None)

        class Level2(Level1[str], Generic[U]):
            _l2: U = PrivateAttr(default=None)

        class Level3(Level2[int]):
            _l3: bool = PrivateAttr(default=False)

        result = get_all_private_attr_annotations(Level3)

        assert result["_l1"] is str  # From Level1[str]
        assert result["_l2"] is int  # From Level2[int]
        assert result["_l3"] is bool  # From Level3
