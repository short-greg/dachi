# 1st party
import typing as t
import pytest

# 3rd party
import pydantic

# local
from dachi.core import Msg, Resp, ListDialog
from dachi.core._tool import tool
from dachi.proc.openai import OpenAIBase, OpenAIChat, OpenAIResp
from dachi.proc._resp import TextOut


# Test fixtures and mock classes
class MockOpenAI:
    """Mock OpenAI client for testing"""
    
    def __init__(self, *args, **kwargs):
        self.calls: list[tuple[tuple, dict]] = []
        self.chat = MockChatCompletions()
        self.responses = MockResponses()


class MockChatCompletions:
    def __init__(self):
        self.completions = MockCompletionsCreate()


class MockCompletionsCreate:
    def __init__(self):
        self.calls: list[dict] = []
    
    def create(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello there!"
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }


class MockResponses:
    def __init__(self):
        self.calls: list[dict] = []
    
    def create(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "id": "resp-123", 
            "object": "response",
            "model": "o1-preview",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello there!"
                },
                "finish_reason": "stop"
            }],
            "reasoning": "The user greeted me, so I should respond politely.",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }


class MockAsyncOpenAI:
    """Mock async OpenAI client for testing"""
    
    def __init__(self, *args, **kwargs):
        self.calls: list[tuple[tuple, dict]] = []
        self.chat = MockAsyncChatCompletions()
        self.responses = MockAsyncResponses()


class MockAsyncChatCompletions:
    def __init__(self):
        self.completions = MockAsyncCompletionsCreate()


class MockAsyncCompletionsCreate:
    def __init__(self):
        self.calls: list[dict] = []
    
    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion", 
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello there!"
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }


class MockAsyncResponses:
    def __init__(self):
        self.calls: list[dict] = []
    
    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "id": "resp-123",
            "object": "response", 
            "model": "o1-preview",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello there!"
                },
                "finish_reason": "stop"
            }],
            "reasoning": "The user greeted me, so I should respond politely.",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }


# Test data structures
@tool
def test_function(query: str) -> str:
    """A test function for tool testing"""
    return f"Result: {query}"


class _TestModel(pydantic.BaseModel):
    name: str
    age: int


class TestOpenAIBase:
    """Test OpenAIBase shared functionality"""
    
    def test_set_tool_arg_with_none_does_nothing(self):
        base = OpenAIBase()
        kwargs = {}
        base.set_tool_arg(None, kwargs)
        assert kwargs == {}
    
    def test_set_tool_arg_with_tools_adds_openai_format(self):
        base = OpenAIBase()
        kwargs = {}
        tools = [test_function]
        
        base.set_tool_arg(tools, kwargs)
        
        assert 'tools' in kwargs
        assert len(kwargs['tools']) == 1
        tool_schema = kwargs['tools'][0]
        assert tool_schema['type'] == 'function'
        assert tool_schema['function']['name'] == 'test_function'
        assert tool_schema['function']['description'] == 'A test function for tool testing'
        assert 'parameters' in tool_schema['function']
    


class TestOpenAIChat:
    """Test OpenAIChat Chat Completions API adapter"""
    
    def setup_method(self):
        """Setup mock clients for each test"""
        self.chat = OpenAIChat()
        self.chat.client = MockOpenAI()
        self.chat.async_client = MockAsyncOpenAI()
    
    def test_to_input_basic_message_creates_messages_array(self):
        msg = Msg(role="user", text="Hello")
        result = self.chat.to_input(msg)
        
        assert 'messages' in result
        assert len(result['messages']) == 1
        assert result['messages'][0]['role'] == 'user'
        assert result['messages'][0]['content'] == 'Hello'
    
    def test_to_input_dialog_creates_multiple_messages(self):
        dialog = ListDialog(messages=[
            Msg(role="user", text="Hello"),
            Msg(role="assistant", text="Hi there!")
        ])
        result = self.chat.to_input(dialog)
        
        assert len(result['messages']) == 2
        assert result['messages'][0]['role'] == 'user'
        assert result['messages'][1]['role'] == 'assistant'
    
    def test_from_output_creates_resp_with_message(self):
        openai_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {"role": "assistant", "content": "Hello there!"},
                "finish_reason": "stop"
            }],
            "usage": {"total_tokens": 15}
        }
        
        resp = self.chat.to_output(openai_response)
        
        assert isinstance(resp, Resp)
        assert resp.msg.role == "assistant"
        assert resp.msg.text == "Hello there!"
        assert resp.finish_reason == "stop"
        assert resp.model == "gpt-4"
        assert resp.response_id == "chatcmpl-123"
    
    def test_forward_with_model_parameter_passes_to_api(self):
        msg = Msg(role="user", text="Hello")
        
        self.chat.forward(msg, model="gpt-4-turbo")
        
        api_calls = self.chat.client.chat.completions.calls
        assert len(api_calls) == 1
        assert api_calls[0]['model'] == 'gpt-4-turbo'
    
    def test_forward_with_tools_parameter_converts_to_openai_format(self):
        msg = Msg(role="user", text="Hello")
        tools = [test_function]
        
        self.chat.forward(msg, tools=tools)
        
        api_calls = self.chat.client.chat.completions.calls
        assert len(api_calls) == 1
        assert 'tools' in api_calls[0]
        assert len(api_calls[0]['tools']) == 1
        assert api_calls[0]['tools'][0]['function']['name'] == 'test_function'
    
    def test_forward_with_structured_true_adds_json_object(self):
        msg = Msg(role="user", text="Hello")
        
        self.chat.forward(msg, structured=True)
        
        api_calls = self.chat.client.chat.completions.calls
        assert len(api_calls) == 1
        assert api_calls[0]['response_format'] == {"type": "json_object"}
    
    def test_forward_with_structured_pydantic_adds_json_schema(self):
        msg = Msg(role="user", text="Hello")
        
        self.chat.forward(msg, structured=_TestModel)
        
        api_calls = self.chat.client.chat.completions.calls
        assert len(api_calls) == 1
        assert 'response_format' in api_calls[0]
        assert api_calls[0]['response_format']['type'] == 'json_schema'
    
    def test_forward_with_out_parameter_processes_response(self):
        msg = Msg(role="user", text="Hello")
        
        resp = self.chat.forward(msg, out=TextOut())
        
        assert hasattr(resp, 'out')
        assert resp.out == "Hello there!"  # Based on mock response
    
    @pytest.mark.asyncio
    async def test_aforward_calls_async_client(self):
        msg = Msg(role="user", text="Hello")
        
        resp = await self.chat.aforward(msg)
        
        api_calls = self.chat.async_client.chat.completions.calls
        assert len(api_calls) == 1
        assert isinstance(resp, Resp)
        assert resp.msg.text == "Hello there!"
    
    def test_from_streamed_accumulates_text_content(self):
        chunk1 = {
            "choices": [{
                "delta": {"role": "assistant", "content": "Hello"},
                "finish_reason": None
            }]
        }
        chunk2 = {
            "choices": [{
                "delta": {"content": " there!"},
                "finish_reason": "stop"
            }]
        }
        
        msg = Msg(role="user", text="Test")
        resp1 = self.chat.from_streamed(chunk1, msg)
        resp2 = self.chat.from_streamed(chunk2, msg, resp1)
        
        assert resp1.msg.text == "Hello"
        assert resp2.msg.text == "Hello there!"
        assert resp2.delta.finish_reason == "stop"
    
    def test_forward_without_optional_params_works(self):
        msg = Msg(role="user", text="Hello")
        
        resp = self.chat.forward(msg)
        
        api_calls = self.chat.client.chat.completions.calls
        assert len(api_calls) == 1
        assert isinstance(resp, Resp)
        # Should not add None values to API call
        assert 'tools' not in api_calls[0] or api_calls[0]['tools'] is None
        assert 'response_format' not in api_calls[0] or api_calls[0]['response_format'] is None

    def test_set_structured_output_arg_with_none_does_nothing(self):
        kwargs = {}
        self.chat.set_structured_output_arg(None, kwargs)
        assert kwargs == {}
    
    def test_set_structured_output_arg_with_true_adds_json_object(self):
        kwargs = {}
        self.chat.set_structured_output_arg(True, kwargs)
        assert kwargs['response_format'] == {"type": "json_object"}
    
    def test_set_structured_output_arg_with_dict_passes_through(self):
        kwargs = {}
        custom_format = {"type": "custom_format", "schema": "test"}
        self.chat.set_structured_output_arg(custom_format, kwargs)
        assert kwargs['response_format'] == custom_format
    
    def test_set_structured_output_arg_with_pydantic_creates_json_schema(self):
        kwargs = {}
        self.chat.set_structured_output_arg(_TestModel, kwargs)
        
        assert 'response_format' in kwargs
        response_format = kwargs['response_format']
        assert response_format['type'] == 'json_schema'
        assert response_format['json_schema']['name'] == '_TestModel'
        assert response_format['json_schema']['strict'] is True
        assert 'schema' in response_format['json_schema']
    
    def test_set_structured_output_arg_with_invalid_type_raises_error(self):
        kwargs = {}
        with pytest.raises(ValueError, match="Unsupported structured output type"):
            self.chat.set_structured_output_arg("invalid", kwargs)


class TestOpenAIResp:
    """Test OpenAIResp Responses API adapter"""
    
    def setup_method(self):
        """Setup mock clients for each test"""
        self.resp_adapter = OpenAIResp()
        self.resp_adapter.client = MockOpenAI()
        self.resp_adapter.async_client = MockAsyncOpenAI()
    
    def test_to_input_single_user_message_creates_input_field(self):
        msg = Msg(role="user", text="Hello")
        result = self.resp_adapter.to_input(msg)
        
        # For Responses API with single user message, should use 'input' field
        if 'input' in result:
            assert result['input'] == 'Hello'
        else:
            # Otherwise should create messages array
            assert 'messages' in result
    
    def test_from_output_captures_reasoning_field(self):
        openai_response = {
            "id": "resp-123",
            "model": "o1-preview", 
            "choices": [{
                "message": {"role": "assistant", "content": "Hello there!"},
                "finish_reason": "stop"
            }],
            "reasoning": "The user greeted me, so I should respond politely.",
            "usage": {"total_tokens": 15}
        }
        
        resp = self.resp_adapter.to_output(openai_response)
        
        assert isinstance(resp, Resp)
        assert resp.thinking == "The user greeted me, so I should respond politely."
        assert resp.msg.text == "Hello there!"
    
    def test_forward_with_reasoning_summary_request_passes_to_api(self):
        msg = Msg(role="user", text="Hello")
        
        self.resp_adapter.forward(msg, reasoning_summary_request=True)
        
        api_calls = self.resp_adapter.client.responses.calls
        assert len(api_calls) == 1
        assert api_calls[0]['reasoning_summary_request'] is True
    
    def test_forward_with_model_and_tools_parameters(self):
        msg = Msg(role="user", text="Hello")
        tools = [test_function]
        
        self.resp_adapter.forward(msg, model="o1-preview", tools=tools)
        
        api_calls = self.resp_adapter.client.responses.calls
        assert len(api_calls) == 1
        assert api_calls[0]['model'] == 'o1-preview'
        assert 'tools' in api_calls[0]
    
    @pytest.mark.asyncio
    async def test_aforward_calls_async_responses_client(self):
        msg = Msg(role="user", text="Hello")
        
        resp = await self.resp_adapter.aforward(msg)
        
        api_calls = self.resp_adapter.async_client.responses.calls
        assert len(api_calls) == 1
        assert isinstance(resp, Resp)
        assert resp.thinking == "The user greeted me, so I should respond politely."

    def test_set_structured_output_arg_with_none_does_nothing(self):
        kwargs = {}
        self.resp_adapter.set_structured_output_arg(None, kwargs)
        assert kwargs == {}
    
    def test_set_structured_output_arg_with_true_adds_text_format(self):
        kwargs = {}
        self.resp_adapter.set_structured_output_arg(True, kwargs)
        assert kwargs['text'] == {"format": {"type": "json_object"}}
    
    def test_set_structured_output_arg_with_dict_passes_through(self):
        kwargs = {}
        custom_format = {"type": "custom_format", "schema": "test"}
        self.resp_adapter.set_structured_output_arg(custom_format, kwargs)
        assert kwargs['text'] == {"format": custom_format}
    
    def test_set_structured_output_arg_with_pydantic_creates_json_schema(self):
        kwargs = {}
        self.resp_adapter.set_structured_output_arg(_TestModel, kwargs)
        
        assert 'text' in kwargs
        text_format = kwargs['text']
        assert text_format['format']['type'] == 'json_schema'
        assert 'schema' in text_format['format']
    
    def test_set_structured_output_arg_with_invalid_type_raises_error(self):
        kwargs = {}
        with pytest.raises(ValueError, match="Unsupported structured output type"):
            self.resp_adapter.set_structured_output_arg("invalid", kwargs)