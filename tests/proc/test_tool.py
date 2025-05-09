from dachi import msg as Msg
import pytest
from dachi.proc import ToolDef
from pydantic import BaseModel
from typing import Callable, Any, Optional, Type
from inspect import signature, Parameter
from typing import get_type_hints, Dict
from dachi.proc import make_tool_def

class MockInputModel(BaseModel):
    field: str

def mock_function(input_data: MockInputModel) -> str:
    return f"Processed {input_data.field}"


class TestToolDef:

    def test_tooldef_initialization(self):
        tool_def = ToolDef(
            name="TestTool",
            description="A test tool",
            fn=mock_function,
            input_model=MockInputModel,
            return_type=str,
            version="1.0.0"
        )
        assert tool_def.name == "TestTool"
        assert tool_def.description == "A test tool"
        assert tool_def.fn == mock_function
        assert tool_def.input_model == MockInputModel
        assert tool_def.return_type == str
        assert tool_def.version == "1.0.0"

    def test_tooldef_missing_optional_fields(self):
        tool_def = ToolDef(
            name="TestTool",
            description="A test tool",
            fn=mock_function,
            input_model=MockInputModel
        )
        assert tool_def.name == "TestTool"
        assert tool_def.description == "A test tool"
        assert tool_def.fn == mock_function
        assert tool_def.input_model == MockInputModel
        assert tool_def.return_type is None
        assert tool_def.version is None

    def test_tooldef_function_execution(self):
        tool_def = ToolDef(
            name="TestTool",
            description="A test tool",
            fn=mock_function,
            input_model=MockInputModel
        )
        input_data = MockInputModel(field="test")
        result = tool_def.fn(input_data)
        assert result == "Processed test"

    # def test_tooldef_invalid_input_model(self):
    #     tool_def = ToolDef(
    #         name="TestTool",
    #         description="A test tool",
    #         fn=mock_function,
    #         input_model=MockInputModel
    #     )
    #     with pytest.raises(ValueError):
    #         tool_def.fn({"field": "test"})  # Invalid input, not a MockInputModel instance

class TestMakeToolDef:

    def test_make_tool_def_with_valid_function(self):
        def sample_function(field: str) -> str:
            """Sample function for testing."""
            return f"Hello, {field}!"

        tool_def = make_tool_def(sample_function)

        assert tool_def.name == "sample_function"
        assert tool_def.description == "Sample function for testing."
        assert tool_def.fn == sample_function
        assert tool_def.input_model.__name__ == "Sample_FunctionInputs"
        assert tool_def.return_type == str

        input_data = tool_def.input_model(field="World")
        result = tool_def.fn(**input_data.model_dump())
        assert result == "Hello, World!"

    def test_make_tool_def_with_missing_docstring(self):
        def no_doc_function(field: int) -> int:
            return field * 2

        tool_def = make_tool_def(no_doc_function)

        assert tool_def.name == "no_doc_function"
        assert tool_def.description == "Tool for no_doc_function"
        assert tool_def.fn == no_doc_function
        assert tool_def.input_model.__name__ == "No_Doc_FunctionInputs"
        assert tool_def.return_type == int

        input_data = tool_def.input_model(field=5)
        result = tool_def.fn(**input_data.model_dump())
        assert result == 10

    def test_make_tool_def_with_no_parameters(self):
        def no_param_function() -> str:
            return "No parameters here!"

        tool_def = make_tool_def(no_param_function)

        assert tool_def.name == "no_param_function"
        assert tool_def.description == "Tool for no_param_function"
        assert tool_def.fn == no_param_function
        assert tool_def.input_model.__name__ == "No_Param_FunctionInputs"
        assert tool_def.return_type == str

        input_data = tool_def.input_model()
        result = tool_def.fn(**input_data.model_dump())
        assert result == "No parameters here!"

    def test_make_tool_def_with_optional_parameters(self):
        def optional_param_function(field: Optional[str] = None) -> str:
            return f"Field is {field or 'empty'}"

        tool_def = make_tool_def(optional_param_function)

        assert tool_def.name == "optional_param_function"
        assert tool_def.description == "Tool for optional_param_function"
        assert tool_def.fn == optional_param_function
        assert tool_def.input_model.__name__ == "Optional_Param_FunctionInputs"
        assert tool_def.return_type == str

        input_data = tool_def.input_model(field="test")
        result = tool_def.fn(**input_data.model_dump())
        assert result == "Field is test"

        input_data = tool_def.input_model()
        result = tool_def.fn(**input_data.model_dump())
        assert result == "Field is empty"

    def test_make_tool_def_with_invalid_function(self):
        with pytest.raises(TypeError):
            make_tool_def("not_a_function")

    def test_make_tool_def_with_varargs(self):
        def varargs_function(*args: int) -> int:
            return sum(args)

        with pytest.raises(TypeError):
            make_tool_def(varargs_function)

    def test_make_tool_def_with_kwargs(self):
        def kwargs_function(**kwargs: str) -> str:
            return ", ".join(f"{k}={v}" for k, v in kwargs.items())

        with pytest.raises(TypeError):
            make_tool_def(kwargs_function)


class TestToolCall:

    def test_tool_call_with_two_args(self):
        def varargs_function(x: int, y: int) -> int:
            return x + y

        tool_def = make_tool_def(varargs_function)

        tool_call = tool_def.to_tool_call(2, 3)
        assert tool_call() == 5

    def test_tool_call_with_keyword_args(self):
        def varargs_function(x: int, y: int) -> int:
            return x + y

        tool_def = make_tool_def(varargs_function)

        tool_call = tool_def.to_tool_call(x=2, y=3)
        assert tool_call() == 5

