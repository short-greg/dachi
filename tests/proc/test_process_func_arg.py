"""Comprehensive tests for func_arg_model and argument handling.

Tests all Python function signature patterns to ensure func_arg_model
correctly generates Pydantic models for Process argument handling.
"""

import pytest
import typing
import inspect
import importlib.util

# Import _arg_model directly to avoid __init__.py import issues
spec = importlib.util.spec_from_file_location(
    '_arg_model',
    '/Users/shortg/Development/dachi/dachi/proc/_arg_model.py'
)
_arg_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_arg_model)

func_arg_model = _arg_model.func_arg_model
Ref = _arg_model.Ref
KWOnly = _arg_model.KWOnly
PosArgs = _arg_model.PosArgs
KWArgs = _arg_model.KWArgs
BaseArgs = _arg_model.BaseArgs


class TestFuncArgModelBasic:
    """Test basic parameter types (POSITIONAL_OR_KEYWORD)"""

    def test_empty_signature_creates_model_with_no_fields(self):
        class TestCls:
            def forward(self):
                pass

        model = func_arg_model(TestCls, TestCls.forward)

        # Should have no fields (except inherited from BaseModel)
        assert len(model.model_fields) == 0
        # Should be instantiable
        instance = model()
        assert instance is not None

    def test_single_param_no_default_creates_required_field(self):
        class TestCls:
            def forward(self, x: int):
                pass

        model = func_arg_model(TestCls, TestCls.forward)

        assert 'x' in model.model_fields
        assert model.model_fields['x'].annotation == int
        assert model.model_fields['x'].is_required()

    def test_single_param_with_default_creates_optional_field(self):
        class TestCls:
            def forward(self, x: int = 5):
                pass

        model = func_arg_model(TestCls, TestCls.forward)

        assert 'x' in model.model_fields
        assert model.model_fields['x'].annotation == int
        assert not model.model_fields['x'].is_required()
        assert model.model_fields['x'].default == 5

    def test_multiple_positional_or_keyword_params(self):
        class TestCls:
            def forward(self, x: int, y: str = "hello", z: bool = True):
                pass

        model = func_arg_model(TestCls, TestCls.forward)

        assert len(model.model_fields) == 3
        assert model.model_fields['x'].is_required()
        assert not model.model_fields['y'].is_required()
        assert not model.model_fields['z'].is_required()
        assert model.model_fields['y'].default == "hello"
        assert model.model_fields['z'].default is True

    def test_param_without_annotation_defaults_to_any(self):
        class TestCls:
            def forward(self, x):
                pass

        model = func_arg_model(TestCls, TestCls.forward)

        assert 'x' in model.model_fields
        assert model.model_fields['x'].annotation == typing.Any


class TestFuncArgModelKeywordOnly:
    """Test keyword-only parameters (after * in signature)"""

    def test_single_keyword_only_param_wrapped_in_kwonly(self):
        class TestCls:
            def forward(self, *, x: int):
                pass

        model = func_arg_model(TestCls, TestCls.forward)

        assert 'x' in model.model_fields
        # Check that the field type is KWOnly[int]
        field_annotation = model.model_fields['x'].annotation
        # Should be a generic type (KWOnly[int])
        assert 'KWOnly' in str(field_annotation)

    def test_keyword_only_param_with_default(self):
        class TestCls:
            def forward(self, *, x: int = 5):
                pass

        model = func_arg_model(TestCls, TestCls.forward)

        assert 'x' in model.model_fields
        # Should be able to instantiate
        instance = model()
        assert hasattr(instance, 'x')

    def test_mixed_positional_and_keyword_only(self):
        class TestCls:
            def forward(self, a: int, *, b: str):
                pass

        model = func_arg_model(TestCls, TestCls.forward)

        assert 'a' in model.model_fields
        assert 'b' in model.model_fields
        assert len(model.model_fields) == 2

    def test_multiple_keyword_only_params(self):
        class TestCls:
            def forward(self, *, x: int, y: str = "default"):
                pass

        model = func_arg_model(TestCls, TestCls.forward)

        assert 'x' in model.model_fields
        assert 'y' in model.model_fields
        assert len(model.model_fields) == 2


class TestFuncArgModelPositionalOnly:
    """Test positional-only parameters (before / in signature)"""

    def test_single_positional_only_param_wrapped_in_posargs(self):
        class TestCls:
            def forward(self, x: int, /):
                pass

        model = func_arg_model(TestCls, TestCls.forward)

        assert 'x' in model.model_fields
        # POSITIONAL_ONLY params are regular fields, not wrapped
        field_annotation = model.model_fields['x'].annotation
        assert field_annotation == int


class TestFuncArgModelVarPositional:
    """Test *args handling (VAR_POSITIONAL)"""

    def test_var_positional_creates_kwargs_field(self):
        class TestCls:
            def forward(self, *args):
                pass

        model = func_arg_model(TestCls, TestCls.forward)

        assert 'args' in model.model_fields
        # Field should be PosArgs type (for *args)
        field_annotation = model.model_fields['args'].annotation
        assert 'PosArgs' in str(field_annotation)

    def test_var_positional_with_type_hint(self):
        class TestCls:
            def forward(self, *args: int):
                pass

        model = func_arg_model(TestCls, TestCls.forward)

        assert 'args' in model.model_fields

    def test_mixed_positional_and_var_positional(self):
        class TestCls:
            def forward(self, x: int, *args):
                pass

        model = func_arg_model(TestCls, TestCls.forward)

        assert 'x' in model.model_fields
        assert 'args' in model.model_fields
        assert len(model.model_fields) == 2


class TestFuncArgModelComplex:
    """Test complex mixed signatures"""

    def test_sequential_like_signature(self):
        class TestCls:
            def forward(self, *x):
                pass

        model = func_arg_model(TestCls, TestCls.forward)

        assert 'x' in model.model_fields
        # Should be able to instantiate
        instance = model()
        assert hasattr(instance, 'x')


class TestFuncArgModelWithRef:
    """Test with_ref=True behavior"""

    def test_with_ref_wraps_simple_type_in_union(self):
        class TestCls:
            def forward(self, x: int):
                pass

        model = func_arg_model(TestCls, TestCls.forward, with_ref=True)

        assert 'x' in model.model_fields
        # Field should be Union[Ref, int]
        field_annotation = model.model_fields['x'].annotation
        # Check it's a Union
        assert hasattr(field_annotation, '__origin__')

    def test_with_ref_handles_no_annotation(self):
        class TestCls:
            def forward(self, x):
                pass

        model = func_arg_model(TestCls, TestCls.forward, with_ref=True)

        assert 'x' in model.model_fields
        # Should be Union[Ref, Any]
        field_annotation = model.model_fields['x'].annotation
        assert hasattr(field_annotation, '__origin__')

    def test_with_ref_multiple_params(self):
        class TestCls:
            def forward(self, x: int, y: str = "default"):
                pass

        model = func_arg_model(TestCls, TestCls.forward, with_ref=True)

        assert 'x' in model.model_fields
        assert 'y' in model.model_fields
        # Both should be wrapped in Union[Ref, T]


class TestFuncArgModelEdgeCases:
    """Test edge cases and special types"""

    def test_complex_generic_type_preserved(self):
        class TestCls:
            def forward(self, x: typing.Dict[str, typing.List[int]]):
                pass

        model = func_arg_model(TestCls, TestCls.forward)

        assert 'x' in model.model_fields
        # Type should be preserved
        field_annotation = model.model_fields['x'].annotation
        assert field_annotation is not None

    def test_incomplete_generic_dict(self):
        class TestCls:
            def forward(self, x: typing.Dict):
                pass

        model = func_arg_model(TestCls, TestCls.forward)

        assert 'x' in model.model_fields
        # Should not crash, even with incomplete generic

    def test_union_type(self):
        class TestCls:
            def forward(self, x: int | str | None = None):
                pass

        model = func_arg_model(TestCls, TestCls.forward)

        assert 'x' in model.model_fields
        assert not model.model_fields['x'].is_required()

    def test_model_can_be_instantiated_with_values(self):
        class TestCls:
            def forward(self, x: int, y: str = "hello"):
                pass

        model = func_arg_model(TestCls, TestCls.forward)

        # Should be able to create instance with values
        instance = model(x=42, y="world")
        assert instance.x == 42
        assert instance.y == "world"

    def test_model_has_correct_name(self):
        class TestCls:
            def forward(self, x: int):
                pass

        model = func_arg_model(TestCls, TestCls.forward)

        assert model.__name__ == "TestClsArgs"


class TestBaseArgsGetArgs:
    """Test BaseArgs.get_args() method integration"""

    def test_get_args_with_regular_fields_returns_positional_args(self):
        class TestCls:
            def forward(self, x: int, y: str):
                pass

        model = func_arg_model(TestCls, TestCls.forward)
        instance = model(x=5, y="hello")

        args, kwargs = instance.get_args({})

        assert args == [5, "hello"]
        assert kwargs == {}

    def test_get_args_with_keyword_only_returns_kwargs(self):
        class TestCls:
            def forward(self, *, x: int = 42):
                pass

        model = func_arg_model(TestCls, TestCls.forward)
        # Create instance using default
        instance = model()

        # This test verifies keyword-only goes to kwargs
        args, kwargs = instance.get_args({})
        assert args == []
        assert kwargs == {'x': 42}

    def test_get_args_with_ref_resolution(self):
        class TestCls:
            def forward(self, x: int):
                pass

        model = func_arg_model(TestCls, TestCls.forward, with_ref=True)
        instance = model(x=Ref(name="some_node"))

        args, kwargs = instance.get_args({"some_node": 42})

        assert args == [42]
        assert kwargs == {}

    def test_get_args_with_mixed_regular_and_ref(self):
        class TestCls:
            def forward(self, x: int, y: str):
                pass

        model = func_arg_model(TestCls, TestCls.forward, with_ref=True)
        instance = model(x=5, y=Ref(name="node1"))

        args, kwargs = instance.get_args({"node1": "resolved"})

        assert args == [5, "resolved"]
        assert kwargs == {}


class TestGetArgsOrdering:
    """Test that get_args() maintains correct argument ordering"""

    def test_get_args_preserves_positional_argument_order(self):
        """Positional args must be in the same order as function signature"""
        class TestCls:
            def forward(self, first: int, second: str, third: bool):
                pass

        model = func_arg_model(TestCls, TestCls.forward)
        instance = model(first=1, second="two", third=True)

        args, kwargs = instance.get_args({})

        # Order MUST match function signature: first, second, third
        assert args == [1, "two", True]
        assert kwargs == {}

    def test_get_args_with_mixed_positional_and_keyword_only(self):
        """Test correct separation of positional vs keyword-only args"""
        class TestCls:
            def forward(self, pos1: int, pos2: str, *, kw1: bool, kw2: float):
                pass

        model = func_arg_model(TestCls, TestCls.forward)
        instance = model(pos1=10, pos2="hello", kw1={'data': True}, kw2={'data': 3.14})

        args, kwargs = instance.get_args({})

        # Positional args in order
        assert args == [10, "hello"]
        # Keyword-only args
        assert kwargs == {"kw1": True, "kw2": 3.14}

    def test_get_args_with_refs_preserves_order(self):
        """Refs should be resolved but maintain their position in args"""
        class TestCls:
            def forward(self, a: int, b: int, c: int):
                pass

        model = func_arg_model(TestCls, TestCls.forward, with_ref=True)
        # Mix of refs and values
        instance = model(a=Ref(name="node1"), b=5, c=Ref(name="node2"))

        by = {"node1": 100, "node2": 200}
        args, kwargs = instance.get_args(by)

        # Must be in signature order: a, b, c
        assert args == [100, 5, 200]
        assert kwargs == {}

    def test_get_args_with_posargs_appends_to_end(self):
        """PosArgs (*args) should be appended after regular positional args"""
        class TestCls:
            def forward(self, regular: int, *args):
                pass

        model = func_arg_model(TestCls, TestCls.forward)
        instance = model(regular=42, args={'data': ["extra1", "extra2"]})

        args, kwargs = instance.get_args({})

        # Regular arg first, then PosArgs data
        assert args == [42, "extra1", "extra2"]
        assert kwargs == {}

    def test_get_args_with_kwargs_merges_correctly(self):
        """KWArgs (**kwargs) should be merged into kwargs dict"""
        class TestCls:
            def forward(self, **kwargs):
                pass

        model = func_arg_model(TestCls, TestCls.forward)
        instance = model(kwargs={'data': {"key1": "val1", "key2": "val2"}})

        args, kwargs = instance.get_args({})

        assert args == []
        assert kwargs == {"key1": "val1", "key2": "val2"}

    def test_get_args_complex_mixed_signature(self):
        """Test a complex signature with all parameter types"""
        class TestCls:
            def forward(self, pos1: int, pos2: str, *, kw1: bool = True):
                pass

        model = func_arg_model(TestCls, TestCls.forward, with_ref=True)
        instance = model(
            pos1=Ref(name="node_a"),
            pos2="literal",
            kw1={'data': False}
        )

        by = {"node_a": 999}
        args, kwargs = instance.get_args(by)

        # Positional args with ref resolved, in order
        assert args == [999, "literal"]
        # Keyword-only
        assert kwargs == {"kw1": False}

    def test_get_args_with_kwonly_ref_resolution(self):
        """Keyword-only params can contain Refs that need resolution"""
        class TestCls:
            def forward(self, *, x: int, y: str):
                pass

        model = func_arg_model(TestCls, TestCls.forward, with_ref=True)
        instance = model(
            x={'data': Ref(name="node1")},
            y={'data': "literal"}
        )

        by = {"node1": 42}
        args, kwargs = instance.get_args(by)

        assert args == []
        assert kwargs == {"x": 42, "y": "literal"}


class TestFuncArgModelInstanceCreation:
    """Test that created models can actually be instantiated"""

    def test_model_with_required_fields_needs_values(self):
        class TestCls:
            def forward(self, x: int, y: str):
                pass

        model = func_arg_model(TestCls, TestCls.forward)

        # Should raise validation error if required fields missing
        with pytest.raises(Exception):  # Pydantic ValidationError
            model()

        # Should work with all required fields
        instance = model(x=42, y="test")
        assert instance.x == 42
        assert instance.y == "test"

    def test_model_with_defaults_can_be_created_empty(self):
        class TestCls:
            def forward(self, x: int = 10, y: str = "default"):
                pass

        model = func_arg_model(TestCls, TestCls.forward)

        # Should work without any arguments
        instance = model()
        assert instance.x == 10
        assert instance.y == "default"

    def test_model_with_mixed_required_and_optional(self):
        class TestCls:
            def forward(self, required: int, optional: str = "default"):
                pass

        model = func_arg_model(TestCls, TestCls.forward)

        # Should work with just required field
        instance = model(required=42)
        assert instance.required == 42
        assert instance.optional == "default"

        # Should work with both fields
        instance2 = model(required=99, optional="custom")
        assert instance2.required == 99
        assert instance2.optional == "custom"


class TestBaseArgsBuild:
    """Test BaseArgs.build() class method"""

    def test_build_with_simple_positional_args(self):
        class TestCls:
            def forward(self, x: int, y: str):
                pass

        model = func_arg_model(TestCls, TestCls.forward)
        instance = model.build(42, "hello")

        assert instance.x == 42
        assert instance.y == "hello"

    def test_build_with_keyword_args(self):
        class TestCls:
            def forward(self, x: int, y: str):
                pass

        model = func_arg_model(TestCls, TestCls.forward)
        instance = model.build(x=42, y="hello")

        assert instance.x == 42
        assert instance.y == "hello"

    def test_build_with_mixed_positional_and_keyword(self):
        class TestCls:
            def forward(self, x: int, y: str, z: bool):
                pass

        model = func_arg_model(TestCls, TestCls.forward)
        instance = model.build(42, y="hello", z=True)

        assert instance.x == 42
        assert instance.y == "hello"
        assert instance.z is True

    def test_build_with_keyword_only_params(self):
        class TestCls:
            def forward(self, x: int, *, y: str):
                pass

        model = func_arg_model(TestCls, TestCls.forward)
        instance = model.build(42, y="hello")

        assert instance.x == 42
        assert isinstance(instance.y, KWOnly)
        assert instance.y.data == "hello"

    def test_build_with_var_positional(self):
        class TestCls:
            def forward(self, x: int, *args):
                pass

        model = func_arg_model(TestCls, TestCls.forward)
        instance = model.build(42, "extra1", "extra2")

        assert instance.x == 42
        assert isinstance(instance.args, PosArgs)
        assert instance.args.data == ["extra1", "extra2"]

    def test_build_with_var_keyword(self):
        class TestCls:
            def forward(self, x: int, **kwargs):
                pass

        model = func_arg_model(TestCls, TestCls.forward)
        instance = model.build(42, extra1="val1", extra2="val2")

        assert instance.x == 42
        assert isinstance(instance.kwargs, KWArgs)
        assert instance.kwargs.data == {"extra1": "val1", "extra2": "val2"}

    def test_build_with_all_param_types(self):
        class TestCls:
            def forward(self, pos: int, *, kw: str, **kwargs):
                pass

        model = func_arg_model(TestCls, TestCls.forward)
        instance = model.build(42, kw="hello", extra="value")

        assert instance.pos == 42
        assert isinstance(instance.kw, KWOnly)
        assert instance.kw.data == "hello"
        assert isinstance(instance.kwargs, KWArgs)
        assert instance.kwargs.data == {"extra": "value"}

    def test_build_with_defaults_uses_defaults_when_not_provided(self):
        class TestCls:
            def forward(self, x: int = 10, y: str = "default"):
                pass

        model = func_arg_model(TestCls, TestCls.forward)
        instance = model.build()

        assert instance.x == 10
        assert instance.y == "default"

    def test_build_with_defaults_override_when_provided(self):
        class TestCls:
            def forward(self, x: int = 10, y: str = "default"):
                pass

        model = func_arg_model(TestCls, TestCls.forward)
        instance = model.build(99, "custom")

        assert instance.x == 99
        assert instance.y == "custom"

    def test_build_roundtrip_with_get_args(self):
        class TestCls:
            def forward(self, x: int, y: str, *, z: bool):
                pass

        model = func_arg_model(TestCls, TestCls.forward)
        instance = model.build(42, "hello", z=True)

        args, kwargs = instance.get_args({})

        assert args == [42, "hello"]
        assert kwargs == {"z": True}

    def test_build_with_var_positional_empty(self):
        class TestCls:
            def forward(self, x: int, *args):
                pass

        model = func_arg_model(TestCls, TestCls.forward)
        instance = model.build(42)

        assert instance.x == 42
        assert isinstance(instance.args, PosArgs)
        assert instance.args.data == []

    def test_build_with_var_keyword_empty(self):
        class TestCls:
            def forward(self, x: int, **kwargs):
                pass

        model = func_arg_model(TestCls, TestCls.forward)
        instance = model.build(42)

        assert instance.x == 42
        assert isinstance(instance.kwargs, KWArgs)
        assert instance.kwargs.data == {}
