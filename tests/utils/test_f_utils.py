import pytest
from dachi.utils import extract_parameter_types


class TestExtractParameterTypes:

    def test_extract_parameter_types_returns_dict(self):
        def example(x: int, y: str) -> float:
            pass

        result = extract_parameter_types(example)
        assert result == {'x': int, 'y': str}

    def test_extract_parameter_types_excludes_self(self):
        class Example:
            def method(self, x: int, y: str):
                pass

        result = extract_parameter_types(Example.method)
        assert result == {'x': int, 'y': str}

    def test_extract_parameter_types_excludes_cls(self):
        class Example:
            @classmethod
            def method(cls, x: int, y: str):
                pass

        result = extract_parameter_types(Example.method)
        assert result == {'x': int, 'y': str}

    def test_extract_parameter_types_excludes_custom_params(self):
        def example(ctx, x: int, y: str):
            pass

        result = extract_parameter_types(example, exclude_params={'ctx'})
        assert result == {'x': int, 'y': str}

    def test_extract_parameter_types_skips_args_kwargs(self):
        def example(x: int, *args, **kwargs):
            pass

        result = extract_parameter_types(example)
        assert result == {'x': int}

    def test_extract_parameter_types_raises_when_missing_annotation(self):
        def untyped(x, y: int):
            pass

        with pytest.raises(TypeError, match="Parameter 'x'.*must have a type annotation"):
            extract_parameter_types(untyped, require_annotations=True)

    def test_extract_parameter_types_allows_missing_when_not_required(self):
        def partial(x, y: int):
            pass

        result = extract_parameter_types(partial, require_annotations=False)
        assert result == {'y': int}

    def test_extract_parameter_types_empty_function(self):
        def no_params():
            pass

        result = extract_parameter_types(no_params)
        assert result == {}

    def test_extract_parameter_types_with_defaults(self):
        def with_defaults(x: int = 5, y: str = "hello"):
            pass

        result = extract_parameter_types(with_defaults)
        assert result == {'x': int, 'y': str}

    def test_extract_parameter_types_complex_types(self):
        from typing import List, Dict, Optional

        def complex_sig(items: List[int], mapping: Dict[str, float], opt: Optional[str]):
            pass

        result = extract_parameter_types(complex_sig)
        assert 'items' in result
        assert 'mapping' in result
        assert 'opt' in result

    def test_extract_parameter_types_async_function(self):
        async def async_func(x: int, y: str) -> float:
            return float(x)

        result = extract_parameter_types(async_func)
        assert result == {'x': int, 'y': str}
