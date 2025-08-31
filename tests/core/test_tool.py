import pytest
from dachi.core import BaseTool, register_tool
from pydantic import BaseModel
from typing import Optional


class MockInputModel(BaseModel):
    field: str

def mock_function(input_data: MockInputModel) -> str:
    return f"Processed {input_data.field}"


class TestToolDef:

    def test_tooldef_initialization(self):
        from dachi.core import Tool
        tool_def = Tool(
            name="TestTool",
            description="A test tool",
            input_model=MockInputModel,
            return_type=str,
            version="1.0.0"
        )
        assert tool_def.name == "TestTool"
        assert tool_def.description == "A test tool"
        assert tool_def.input_model == MockInputModel
        assert tool_def.return_type == str
        assert tool_def.version == "1.0.0"

    def test_tooldef_missing_optional_fields(self):
        from dachi.core import Tool
        tool_def = Tool(
            name="TestTool",
            description="A test tool",
            input_model=MockInputModel
        )
        assert tool_def.name == "TestTool"
        assert tool_def.description == "A test tool"
        assert tool_def.input_model == MockInputModel
        assert tool_def.return_type is None
        assert tool_def.version is None

    def test_tooldef_function_execution(self):
        def test_func(field: str) -> str:
            return f"Processed {field}"
        
        # Register the function and test execution through the Tool
        from dachi.core import register_tool
        tool_def = register_tool(test_func)
        result = tool_def(field="test")
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

        tool_def = register_tool(sample_function)

        assert tool_def.name == "TestMakeToolDef.test_make_tool_def_with_valid_function.<locals>.sample_function"
        assert tool_def.description == "Sample function for testing."
        assert tool_def.input_model.__name__ == "TestMakeToolDef.test_make_tool_def_with_valid_function.<locals>.sample_functionInputs"
        assert tool_def.return_type == str

        input_data = tool_def.input_model(field="World")
        result = tool_def(field="World")
        assert result == "Hello, World!"

    def test_make_tool_def_with_missing_docstring(self):
        def no_doc_function(field: int) -> int:
            return field * 2

        tool_def = register_tool(no_doc_function)

        assert tool_def.name == "TestMakeToolDef.test_make_tool_def_with_missing_docstring.<locals>.no_doc_function"
        assert tool_def.description.startswith("Tool for TestMakeToolDef.test_make_tool_def_with_missing_docstring.<locals>.no_doc_function")
        # Input model name is generated from function qualname
        assert tool_def.return_type == int

        input_data = tool_def.input_model(field=5)
        result = tool_def(field=5)
        assert result == 10

    def test_make_tool_def_with_no_parameters(self):
        def no_param_function() -> str:
            return "No parameters here!"

        tool_def = register_tool(no_param_function)

        assert tool_def.name == "TestMakeToolDef.test_make_tool_def_with_no_parameters.<locals>.no_param_function"
        assert tool_def.description.startswith("Tool for TestMakeToolDef.test_make_tool_def_with_no_parameters.<locals>.no_param_function")
        # Input model name is generated from function qualname
        assert tool_def.return_type == str

        input_data = tool_def.input_model()
        result = tool_def()
        assert result == "No parameters here!"

    def test_make_tool_def_with_optional_parameters(self):
        def optional_param_function(field: Optional[str] = None) -> str:
            return f"Field is {field or 'empty'}"

        tool_def = register_tool(optional_param_function)

        assert tool_def.name == "TestMakeToolDef.test_make_tool_def_with_optional_parameters.<locals>.optional_param_function"
        assert tool_def.description.startswith("Tool for TestMakeToolDef.test_make_tool_def_with_optional_parameters.<locals>.optional_param_function")
        # Input model name is generated from function qualname
        assert tool_def.return_type == str

        input_data = tool_def.input_model(field="test")
        result = tool_def(field="test")
        assert result == "Field is test"

        input_data = tool_def.input_model()
        result = tool_def()
        assert result == "Field is empty"

    def test_make_tool_def_with_invalid_function(self):
        with pytest.raises(TypeError):
            register_tool("not_a_function")

    def test_make_tool_def_with_varargs(self):
        def varargs_function(*args: int) -> int:
            return sum(args)

        with pytest.raises(TypeError):
            register_tool(varargs_function)

    def test_make_tool_def_with_kwargs(self):
        def kwargs_function(**kwargs: str) -> str:
            return ", ".join(f"{k}={v}" for k, v in kwargs.items())

        with pytest.raises(TypeError):
            register_tool(kwargs_function)


class TestToolCall:

    def test_tool_call_with_two_args(self):
        def varargs_function(x: int, y: int) -> int:
            return x + y

        tool_def = register_tool(varargs_function)

        tool_call = tool_def.to_tool_call(2, 3, tool_id="x")
        assert tool_call() == 5

    def test_tool_call_with_keyword_args(self):
        def varargs_function(x: int, y: int) -> int:
            return x + y

        tool_def = register_tool(varargs_function)

        tool_call = tool_def.to_tool_call(x=2, y=3, tool_id="x")
        assert tool_call() == 5


class TestToolChunk:
    """Test ToolChunk creation and validation."""
    
    def test_toolchunk_basic_creation(self):
        from dachi.core import ToolChunk
        
        chunk = ToolChunk(
            id="call_123",
            name="test_tool",
            args_text_delta='{"x": 5',
            done=False
        )
        
        assert chunk.id == "call_123"
        assert chunk.name == "test_tool"
        assert chunk.args_text_delta == '{"x": 5'
        assert chunk.done == False
    
    def test_toolchunk_routing_by_id(self):
        from dachi.core import ToolChunk
        
        chunk1 = ToolChunk(id="call_1", name="tool_a")
        chunk2 = ToolChunk(id="call_2", name="tool_b")
        
        # Different IDs should create different routing keys
        assert chunk1.id != chunk2.id
    
    def test_toolchunk_routing_by_index(self):
        from dachi.core import ToolChunk
        
        chunk1 = ToolChunk(turn_index=0, call_index=0, name="tool_a")
        chunk2 = ToolChunk(turn_index=0, call_index=1, name="tool_b")
        
        # Different call_index should create different routing
        assert chunk1.call_index != chunk2.call_index
    
    def test_toolchunk_optional_fields(self):
        from dachi.core import ToolChunk
        
        # Should work with minimal fields
        chunk = ToolChunk()
        assert chunk.id is None
        assert chunk.name is None
        assert chunk.done == False


class TestToolBuffer:
    """Test ToolBuffer streaming accumulation."""
    
    def setup_method(self):
        """Setup test tools."""
        def add_numbers(x: int, y: int) -> int:
            return x + y
            
        def multiply_numbers(a: int, b: int) -> int:
            return a * b
        
        self.add_tool = register_tool(add_numbers)
        self.multiply_tool = register_tool(multiply_numbers)
        self.tools = [self.add_tool, self.multiply_tool]
    
    def test_toolbuffer_single_tool_complete(self):
        """Test complete single tool call streaming."""
        from dachi.core import ToolBuffer, ToolChunk
        
        buffer = ToolBuffer(tools=self.tools)
        
        # Stream tool call in chunks
        chunks = [
            ToolChunk(id="call_1", name=self.add_tool.name),
            ToolChunk(id="call_1", args_text_delta='{"x": 5'),
            ToolChunk(id="call_1", args_text_delta=', "y": 3}'),
            ToolChunk(id="call_1", done=True)
        ]
        
        results = []
        for chunk in chunks:
            completed = buffer.append(chunk)
            if completed:
                results.append(completed)
        
        # Should have one completed tool call
        assert len(results) == 1
        # Should have processed the tool call correctly
        assert len(buffer._calls) > 0
    
    def test_toolbuffer_parallel_tool_calls(self):
        """Test parallel tool calls streaming simultaneously."""
        from dachi.core import ToolBuffer, ToolChunk
        
        buffer = ToolBuffer(tools=self.tools)
        
        # Interleaved chunks from two parallel tool calls
        chunks = [
            # Start tool 1
            ToolChunk(id="call_1", name=self.add_tool.name),
            # Start tool 2  
            ToolChunk(id="call_2", name=self.multiply_tool.name),
            # Continue tool 1
            ToolChunk(id="call_1", args_text_delta='{"x": 5'),
            # Continue tool 2
            ToolChunk(id="call_2", args_text_delta='{"a": 3'),
            # Complete tool 1
            ToolChunk(id="call_1", args_text_delta=', "y": 3}'),
            ToolChunk(id="call_1", done=True),
            # Complete tool 2
            ToolChunk(id="call_2", args_text_delta=', "b": 4}'),
            ToolChunk(id="call_2", done=True)
        ]
        
        completed_count = 0
        for chunk in chunks:
            if buffer.append(chunk):
                completed_count += 1
        
        # Should have completed both tool calls
        assert completed_count == 2
        assert len(buffer._calls) == 2
    
    def test_toolbuffer_key_based_routing(self):
        """Test that different routing keys create separate accumulators."""
        from dachi.core import ToolBuffer, ToolChunk
        
        buffer = ToolBuffer(tools=self.tools)
        
        # Test ID-based routing
        chunk1 = ToolChunk(id="call_1", name=self.add_tool.name, args_text_delta='{"x":')
        chunk2 = ToolChunk(id="call_2", name=self.multiply_tool.name, args_text_delta='{"a":')
        
        buffer.append(chunk1)
        buffer.append(chunk2)
        
        # Should have two separate accumulators
        assert len(buffer._acc) == 2
        
        # Test index-based routing (when no ID)
        chunk3 = ToolChunk(turn_index=0, call_index=0, name=self.add_tool.name)
        chunk4 = ToolChunk(turn_index=0, call_index=1, name=self.multiply_tool.name)
        
        buffer.append(chunk3)
        buffer.append(chunk4)
        
        # Should now have 4 separate accumulators
        assert len(buffer._acc) == 4
    
    def test_toolbuffer_json_accumulation_text_fragments(self):
        """Test JSON accumulation via text fragments."""
        from dachi.core import ToolBuffer, ToolChunk
        
        buffer = ToolBuffer(tools=self.tools)
        
        chunks = [
            ToolChunk(id="call_1", name=self.add_tool.name),
            ToolChunk(id="call_1", args_text_delta='{"x"'),
            ToolChunk(id="call_1", args_text_delta=': 5, '),
            ToolChunk(id="call_1", args_text_delta='"y": 3}'),
            ToolChunk(id="call_1", done=True)
        ]
        
        completed_calls = []
        for chunk in chunks:
            result = buffer.append(chunk)
            if result:
                completed_calls.append(result)
        
        # Should have completed one tool call
        assert len(completed_calls) == 1
        assert len(buffer._calls) == 1
        
        # The accumulator should be cleared after completion
        assert len(buffer._acc) == 0
    
    def test_toolbuffer_kv_patch_accumulation(self):
        """Test JSON accumulation via key-value patches."""
        from dachi.core import ToolBuffer, ToolChunk
        
        buffer = ToolBuffer(tools=self.tools)
        
        chunks = [
            ToolChunk(id="call_1", name=self.add_tool.name),
            ToolChunk(id="call_1", args_kv_patch={"x": 5}),
            ToolChunk(id="call_1", args_kv_patch={"y": 3}),
            ToolChunk(id="call_1", done=True)
        ]
        
        completed_calls = []
        for chunk in chunks:
            result = buffer.append(chunk)
            if result:
                completed_calls.append(result)
        
        # Should have completed one tool call
        assert len(completed_calls) == 1
        assert len(buffer._calls) == 1
        
        # The accumulator should be cleared after completion
        assert len(buffer._acc) == 0
        
        # Verify the tool call was created with correct inputs
        tool_call = buffer._calls[0]
        assert tool_call.inputs.x == 5
        assert tool_call.inputs.y == 3
    
    def test_toolbuffer_unknown_tool_error(self):
        """Test error handling for unknown tools."""
        from dachi.core import ToolBuffer, ToolChunk
        
        buffer = ToolBuffer(tools=self.tools)
        
        # Try to use unknown tool
        chunk = ToolChunk(
            id="call_1", 
            name="unknown_tool",
            args_text_delta='{"x": 1}',
            done=True
        )
        
        with pytest.raises(KeyError, match="unknown tool"):
            buffer.append(chunk)
    
    def test_toolbuffer_incomplete_chunks(self):
        """Test handling of incomplete tool call sequences.""" 
        from dachi.core import ToolBuffer, ToolChunk
        
        buffer = ToolBuffer(tools=self.tools)
        
        # Add chunks but never mark as done
        chunks = [
            ToolChunk(id="call_1", name=self.add_tool.name),
            ToolChunk(id="call_1", args_text_delta='{"x": 5')
            # Missing closing and done=True
        ]
        
        completed_count = 0
        for chunk in chunks:
            if buffer.append(chunk):
                completed_count += 1
        
        # Should not have completed any tool calls
        assert completed_count == 0
        assert len(buffer._calls) == 0
        # But should have partial accumulator
        assert len(buffer._acc) == 1
    
    def test_toolbuffer_make_key_logic(self):
        """Test the key generation logic for routing."""
        from dachi.core import ToolBuffer, ToolChunk
        
        buffer = ToolBuffer(tools=self.tools)
        
        # Test ID-based key
        chunk1 = ToolChunk(id="call_123")
        key1 = buffer._make_key(chunk1)
        assert key1 == ("call_123", None, None)
        
        # Test index-based key 
        chunk2 = ToolChunk(turn_index=0, call_index=1)
        key2 = buffer._make_key(chunk2)
        assert key2 == (None, 0, 1)
        
        # Test ID takes precedence over indices
        chunk3 = ToolChunk(id="call_456", turn_index=0, call_index=1)
        key3 = buffer._make_key(chunk3)
        assert key3 == ("call_456", None, None)


class TestToolBufferEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Setup test tools."""
        def simple_tool(value: str) -> str:
            return f"processed: {value}"
        
        self.tool = register_tool(simple_tool)
        self.tools = [self.tool]
    
    def test_malformed_json_handling(self):
        """Test handling of malformed JSON in chunks."""
        from dachi.core import ToolBuffer, ToolChunk
        
        buffer = ToolBuffer(tools=self.tools)
        
        chunks = [
            ToolChunk(id="call_1", name=self.tool.name),
            ToolChunk(id="call_1", args_text_delta='{"value": "test"'),  # Missing closing
            ToolChunk(id="call_1", done=True)
        ]
        
        # This should handle malformed JSON gracefully
        # Either by repair or error handling
        completed_calls = []
        for chunk in chunks:
            result = buffer.append(chunk)
            if result:
                completed_calls.append(result)
        
        # Should either succeed with repair or fail gracefully
        # (Exact behavior depends on implementation strategy)
    
    def test_empty_chunks(self):
        """Test handling of empty or minimal chunks."""
        from dachi.core import ToolBuffer, ToolChunk
        
        buffer = ToolBuffer(tools=self.tools)
        
        # Empty chunk
        empty_chunk = ToolChunk()
        result = buffer.append(empty_chunk)
        assert result == False  # Should not complete
        
        # Chunk with only done flag
        done_chunk = ToolChunk(done=True)
        result = buffer.append(done_chunk)
        assert result == False  # Should not complete without name/id
    
    def test_out_of_order_chunks(self):
        """Test handling of chunks arriving out of expected order."""
        from dachi.core import ToolBuffer, ToolChunk
        
        buffer = ToolBuffer(tools=self.tools)
        
        # Send done before name/args
        chunks = [
            ToolChunk(id="call_1", args_text_delta='{"value": "test"}'),  # Args first
            ToolChunk(id="call_1", name=self.tool.name),  # Name after
            ToolChunk(id="call_1", done=True)  # Done last
        ]
        
        completed_count = 0
        for chunk in chunks:
            if buffer.append(chunk):
                completed_count += 1
        
        # Should handle gracefully (exact behavior TBD)
        # At minimum, should not crash
    
    def test_duplicate_completion_signals(self):
        """Test handling of multiple done=True signals."""
        from dachi.core import ToolBuffer, ToolChunk
        
        buffer = ToolBuffer(tools=self.tools)
        
        chunks = [
            ToolChunk(id="call_1", name=self.tool.name),
            ToolChunk(id="call_1", args_text_delta='{"value": "test"}'),
            ToolChunk(id="call_1", done=True),  # First completion
            ToolChunk(id="call_1", done=True)   # Duplicate completion
        ]
        
        completed_count = 0
        for chunk in chunks:
            if buffer.append(chunk):
                completed_count += 1
        
        # Should only complete once, not twice
        assert completed_count <= 1


class TestToolBufferIntegration:
    """Integration tests simulating real provider scenarios."""
    
    def setup_method(self):
        """Setup realistic tools."""
        def calculate(operation: str, x: float, y: float) -> float:
            if operation == "add":
                return x + y
            elif operation == "multiply": 
                return x * y
            else:
                raise ValueError(f"Unknown operation: {operation}")
        
        def get_weather(city: str, units: str = "celsius") -> dict:
            return {"city": city, "temp": 22, "units": units}
        
        self.calc_tool = register_tool(calculate)
        self.weather_tool = register_tool(get_weather)
        self.tools = [self.calc_tool, self.weather_tool]
    
    def test_openai_style_streaming(self):
        """Simulate OpenAI-style streaming chunks."""
        from dachi.core import ToolBuffer, ToolChunk
        
        buffer = ToolBuffer(tools=self.tools)
        
        # Simulate OpenAI parallel tool calls
        openai_chunks = [
            # Call 1 starts
            ToolChunk(id="call_abc123", call_index=0, name=self.calc_tool.name),
            # Call 2 starts  
            ToolChunk(id="call_def456", call_index=1, name=self.weather_tool.name),
            # Call 1 args
            ToolChunk(id="call_abc123", call_index=0, args_text_delta='{"operation": "add", "x": 10'),
            # Call 2 args
            ToolChunk(id="call_def456", call_index=1, args_text_delta='{"city": "Tokyo"'),
            # Call 1 continues
            ToolChunk(id="call_abc123", call_index=0, args_text_delta=', "y": 5}'),
            # Call 2 continues  
            ToolChunk(id="call_def456", call_index=1, args_text_delta=', "units": "fahrenheit"}'),
            # Call 1 completes
            ToolChunk(id="call_abc123", call_index=0, done=True),
            # Call 2 completes
            ToolChunk(id="call_def456", call_index=1, done=True)
        ]
        
        completed_calls = []
        for chunk in openai_chunks:
            result = buffer.append(chunk)
            if result:
                completed_calls.append(result)
        
        assert len(completed_calls) == 2
        assert len(buffer._calls) == 2
    
    def test_claude_style_streaming(self):
        """Simulate Claude-style event streaming."""
        from dachi.core import ToolBuffer, ToolChunk
        
        buffer = ToolBuffer(tools=self.tools)
        
        # Simulate Claude content blocks
        claude_chunks = [
            ToolChunk(turn_index=0, call_index=0, name=self.calc_tool.name),
            ToolChunk(turn_index=0, call_index=0, args_text_delta='{"operation": "multiply"'),
            ToolChunk(turn_index=0, call_index=0, args_text_delta=', "x": 7, "y": 6}'),
            ToolChunk(turn_index=0, call_index=0, done=True)
        ]
        
        completed_calls = []
        for chunk in claude_chunks:
            result = buffer.append(chunk)
            if result:
                completed_calls.append(result)
        
        assert len(completed_calls) == 1
    
    def test_gemini_style_kv_updates(self):
        """Simulate Gemini-style key-value updates."""
        from dachi.core import ToolBuffer, ToolChunk
        
        buffer = ToolBuffer(tools=self.tools)
        
        # Simulate Gemini sending complete arg objects
        gemini_chunks = [
            ToolChunk(turn_index=0, call_index=0, name=self.weather_tool.name),
            ToolChunk(turn_index=0, call_index=0, args_kv_patch={"city": "London"}),
            ToolChunk(turn_index=0, call_index=0, args_kv_patch={"units": "celsius"}),
            ToolChunk(turn_index=0, call_index=0, done=True)
        ]
        
        completed_calls = []
        for chunk in gemini_chunks:
            result = buffer.append(chunk)
            if result:
                completed_calls.append(result)
        
        assert len(completed_calls) == 1
