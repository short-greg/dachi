# # 1st party
# import pytest

# # 3rd party
# import pydantic

# # local
# from dachi.core import Inp, Resp, ListDialog, Inp
# from dachi.core._tool import tool
# from dachi.proc.openai import OpenAIChat, OpenAIResp, build_openai_response_format, build_openai_text_format
# from dachi.proc._resp import TextOut


# # Test fixtures and mock classes
# class MockOpenAI:
#     """Mock OpenAI client for testing"""
    
#     def __init__(self, *_args, **_kwargs):
#         self.calls: list[tuple[tuple, dict]] = []
#         self.chat = MockChatCompletions()
#         self.responses = MockResponses()


# class MockChatCompletions:
#     def __init__(self):
#         self.completions = MockCompletionsCreate()


# class MockCompletionsCreate:
#     def __init__(self):
#         self.calls: list[dict] = []
    
#     def create(self, **kwargs):
#         self.calls.append(kwargs)
#         return {
#             "id": "chatcmpl-123",
#             "object": "chat.completion",
#             "model": "gpt-4",
#             "choices": [{
#                 "index": 0,
#                 "message": {
#                     "role": "assistant",
#                     "content": "Hello there!"
#                 },
#                 "finish_reason": "stop"
#             }],
#             "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
#         }


# class MockResponses:
#     def __init__(self):
#         self.calls: list[dict] = []
    
#     def create(self, **kwargs):
#         self.calls.append(kwargs)
#         return {
#             "id": "resp-123", 
#             "object": "response",
#             "model": "o1-preview",
#             "choices": [{
#                 "index": 0,
#                 "message": {
#                     "role": "assistant",
#                     "content": "Hello there!"
#                 },
#                 "finish_reason": "stop"
#             }],
#             "reasoning": "The user greeted me, so I should respond politely.",
#             "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
#         }


# class MockAsyncOpenAI:
#     """Mock async OpenAI client for testing"""
    
#     def __init__(self, *_args, **_kwargs):
#         self.calls: list[tuple[tuple, dict]] = []
#         self.chat = MockAsyncChatCompletions()
#         self.responses = MockAsyncResponses()


# class MockAsyncChatCompletions:
#     def __init__(self):
#         self.completions = MockAsyncCompletionsCreate()


# class MockAsyncCompletionsCreate:
#     def __init__(self):
#         self.calls: list[dict] = []
    
#     async def create(self, **kwargs):
#         self.calls.append(kwargs)
#         return {
#             "id": "chatcmpl-123",
#             "object": "chat.completion", 
#             "model": "gpt-4",
#             "choices": [{
#                 "index": 0,
#                 "message": {
#                     "role": "assistant",
#                     "content": "Hello there!"
#                 },
#                 "finish_reason": "stop"
#             }],
#             "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
#         }


# class MockAsyncResponses:
#     def __init__(self):
#         self.calls: list[dict] = []
    
#     async def create(self, **kwargs):
#         self.calls.append(kwargs)
#         return {
#             "id": "resp-123",
#             "object": "response", 
#             "model": "o1-preview",
#             "choices": [{
#                 "index": 0,
#                 "message": {
#                     "role": "assistant",
#                     "content": "Hello there!"
#                 },
#                 "finish_reason": "stop"
#             }],
#             "reasoning": "The user greeted me, so I should respond politely.",
#             "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
#         }


# # Test data structures
# @tool
# def tool_test_function(query: str) -> str:
#     """A test function for tool testing"""
#     return f"Result: {query}"


# class _TestModel(pydantic.BaseModel):
#     name: str
#     age: int


# class TestOpenAIChat:
#     """Test OpenAIChat Chat Completions API adapter"""
    
#     def setup_method(self):
#         """Setup adapter for each test"""
#         self.chat = OpenAIChat()
    
#     def test_to_input_basic_message_creates_messages_array(self):
#         msg = Inp(role="user", text="Hello")
#         result = self.chat.to_input(msg)
        
#         assert 'messages' in result
#         assert len(result['messages']) == 1
#         assert result['messages'][0]['role'] == 'user'
#         assert result['messages'][0]['content'] == 'Hello'
    
#     def test_to_input_dialog_creates_multiple_messages(self):
#         dialog = ListDialog(messages=[
#             Inp(role="user", text="Hello"),
#             Inp(role="assistant", text="Hi there!")
#         ])
#         result = self.chat.to_input(dialog)
        
#         assert len(result['messages']) == 2
#         assert result['messages'][0]['role'] == 'user'
#         assert result['messages'][1]['role'] == 'assistant'
    
#     def test_from_result_creates_resp_with_message(self):
#         openai_response = {
#             "id": "chatcmpl-123",
#             "model": "gpt-4",
#             "choices": [{
#                 "message": {"role": "assistant", "content": "Hello there!"},
#                 "finish_reason": "stop"
#             }],
#             "usage": {"total_tokens": 15}
#         }
#         msg = Inp(role="user", text="Hello")
        
#         resp = self.chat.from_result(openai_response, msg)
        
#         assert isinstance(resp, Resp)
#         assert resp.role == "assistant"
#         assert resp.text == "Hello there!"
#         assert resp.finish_reason == "stop"
#         assert resp.model == "gpt-4"
#         assert resp.id == "chatcmpl-123"
    
#     def test_forward_with_model_parameter_passes_to_api(self):
#         msg = Inp(role="user", text="Hello")
#         mock_client = MockOpenAI()
        
#         self.chat.forward(mock_client.chat.completions.create, msg, model="gpt-4-turbo")
        
#         api_calls = mock_client.chat.completions.calls
#         assert len(api_calls) == 1
#         assert api_calls[0]['model'] == 'gpt-4-turbo'
    
#     def test_to_input_with_tools_parameter_converts_to_openai_format(self):
#         msg = Inp(role="user", text="Hello", tools=[tool_test_function])
        
#         result = self.chat.to_input(msg)

#         assert 'tools' in result
#         assert len(result['tools']) == 1
#         assert result['tools'][0]['function']['name'] == 'tool_test_function'
    
#     def test_to_input_with_format_override_json_adds_json_object(self):
#         msg = Inp(role="user", text="Hello", format_override="json")
        
#         result = self.chat.to_input(msg)
        
#         assert 'response_format' in result
#         assert result['response_format'] == {"type": "json_object"}
    
#     def test_to_input_with_format_override_pydantic_adds_json_schema(self):
#         msg = Inp(role="user", text="Hello", format_override=_TestModel)
        
#         result = self.chat.to_input(msg)
        
#         assert 'response_format' in result
#         assert result['response_format']['type'] == 'json_schema'
    
#     def test_forward_with_out_parameter_processes_response(self):
#         msg = Inp(role="user", text="Hello")
#         mock_client = MockOpenAI()
        
#         resp = self.chat.forward(mock_client.chat.completions.create, msg, out=TextOut())
        
#         assert hasattr(resp, 'out')
#         assert resp.out == "Hello there!"  # Based on mock response
    
#     @pytest.mark.asyncio
#     async def test_aforward_calls_async_client(self):
#         msg = Inp(role="user", text="Hello")
#         mock_client = MockAsyncOpenAI()
        
#         resp = await self.chat.aforward(mock_client.chat.completions.create, msg)
        
#         api_calls = mock_client.chat.completions.calls
#         assert len(api_calls) == 1
#         assert isinstance(resp, Resp)
#         assert resp.text == "Hello there!"
    
#     def test_from_streamed_result_accumulates_text_content(self):
#         chunk1 = {
#             "choices": [{
#                 "delta": {"role": "assistant", "content": "Hello"},
#                 "finish_reason": None
#             }]
#         }
#         chunk2 = {
#             "choices": [{
#                 "delta": {"content": " there!"},
#                 "finish_reason": "stop"
#             }]
#         }
        
#         msg = Inp(role="user", text="Test")
#         resp1, _ = self.chat.from_streamed_result(chunk1, msg, None)
#         resp2, delta2 = self.chat.from_streamed_result(chunk2, msg, resp1)
        
#         assert resp1.text == "Hello"
#         assert resp2.text == "Hello there!"
#         assert delta2.finish_reason == "stop"
    
#     def test_forward_without_optional_params_works(self):
#         msg = Inp(role="user", text="Hello")
#         mock_client = MockOpenAI()
        
#         resp = self.chat.forward(mock_client.chat.completions.create, msg)
        
#         api_calls = mock_client.chat.completions.calls
#         assert len(api_calls) == 1
#         assert isinstance(resp, Resp)
#         # Should not add None values to API call
#         assert 'tools' not in api_calls[0] or api_calls[0]['tools'] is None
#         assert 'response_format' not in api_calls[0] or api_calls[0]['response_format'] is None

#     def test_build_openai_response_format_with_none_does_nothing(self):
#         result = build_openai_response_format(None)
#         assert result == {}
    
#     def test_build_openai_response_format_with_true_adds_json_object(self):
#         result = build_openai_response_format(True)
#         assert result['response_format'] == {"type": "json_object"}
    
#     def test_build_openai_response_format_with_dict_passes_through(self):
#         custom_format = {"type": "json_schema", "schema": "test"}
#         result = build_openai_response_format(custom_format)
#         assert result['response_format'] == custom_format
    
#     def test_build_openai_response_format_with_pydantic_creates_json_schema(self):
#         result = build_openai_response_format(_TestModel)
        
#         assert 'response_format' in result
#         response_format = result['response_format']
#         assert response_format['type'] == 'json_schema'
#         assert response_format['json_schema']['name'] == '_TestModel'
#         assert response_format['json_schema']['strict'] is True
#         assert 'schema' in response_format['json_schema']
    
#     def test_build_openai_response_format_with_invalid_type_raises_error(self):
#         with pytest.raises(ValueError, match="Unsupported format_override type"):
#             build_openai_response_format("invalid")


# class TestOpenAIResp:
#     """Test OpenAIResp Responses API adapter"""
    
#     def setup_method(self):
#         """Setup adapter for each test"""
#         self.resp_adapter = OpenAIResp()
    
#     def test_to_input_single_user_message_creates_input_field(self):
#         msg = Inp(role="user", text="Hello")
#         result = self.resp_adapter.to_input(msg)
        
#         # For Responses API with single user message, should use 'input' field
#         if 'input' in result:
#             assert result['input'] == 'Hello'
#         else:
#             # Otherwise should create messages array
#             assert 'messages' in result
    
#     def test_from_result_captures_reasoning_field(self):
#         openai_response = {
#             "id": "resp-123",
#             "model": "o1-preview", 
#             "choices": [{
#                 "message": {"role": "assistant", "content": "Hello there!"},
#                 "finish_reason": "stop"
#             }],
#             "reasoning": "The user greeted me, so I should respond politely.",
#             "usage": {"total_tokens": 15}
#         }
#         msg = Inp(role="user", text="Hello")
        
#         resp = self.resp_adapter.from_result(openai_response, msg)
        
#         assert isinstance(resp, Resp)
#         assert resp.thinking == "The user greeted me, so I should respond politely."
#         assert resp.text == "Hello there!"
    
#     def test_forward_with_reasoning_summary_request_passes_to_api(self):
#         msg = Inp(role="user", text="Hello")
#         mock_client = MockOpenAI()
        
#         self.resp_adapter.forward(mock_client.responses.create, msg, reasoning_summary_request=True)
        
#         api_calls = mock_client.responses.calls
#         assert len(api_calls) == 1
#         assert api_calls[0]['reasoning_summary_request'] is True
    
#     def test_to_input_with_model_and_tools_parameters(self):
#         msg = Inp(role="user", text="Hello", tools=[tool_test_function])
        
#         result = self.resp_adapter.to_input(msg, model="o1-preview")
        
#         assert result['model'] == 'o1-preview'
#         assert 'tools' in result
    
#     @pytest.mark.asyncio
#     async def test_aforward_calls_async_responses_client(self):
#         msg = Inp(role="user", text="Hello")
#         mock_client = MockAsyncOpenAI()
        
#         resp = await self.resp_adapter.aforward(mock_client.responses.create, msg)
        
#         api_calls = mock_client.responses.calls
#         assert len(api_calls) == 1
#         assert isinstance(resp, Resp)
#         assert resp.thinking == "The user greeted me, so I should respond politely."

#     def test_build_openai_text_format_with_none_does_nothing(self):
#         result = build_openai_text_format(None)
#         assert result == {}
    
#     def test_build_openai_text_format_with_true_adds_text_format(self):
#         result = build_openai_text_format(True)
#         assert result['text'] == {"format": {"type": "json_object"}}
    
#     def test_build_openai_text_format_with_dict_passes_through(self):
#         custom_format = {"type": "custom_format", "schema": "test"}
#         result = build_openai_text_format(custom_format)
#         assert result['text'] == {"format": custom_format}
    
#     def test_build_openai_text_format_with_pydantic_creates_json_schema(self):
#         result = build_openai_text_format(_TestModel)
        
#         assert 'text' in result
#         text_format = result['text']
#         assert text_format['format']['type'] == 'json_schema'
#         assert 'schema' in text_format['format']
    
#     def test_build_openai_text_format_with_invalid_type_raises_error(self):
#         with pytest.raises(ValueError, match="Unsupported format_override type"):
#             build_openai_text_format("invalid")