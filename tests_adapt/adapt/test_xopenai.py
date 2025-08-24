# """Tests for Dachi's OpenAI adapter layer."""

# from __future__ import annotations

# import json
# import types
# from types import SimpleNamespace

# import pytest
# import pydantic
# import sys

# # Dachi imports
# from dachi.core import Resp, RespDelta, END_TOK, ToolDef, Msg, ToolCall, OpenAIChat, OpenAIResp
# from dachi.utils import UNDEFINED


# def _patch_module(monkeypatch, name, replacement):
#     """Register *replacement* under *name* in sys.modules (sub-modules too)."""
#     monkeypatch.setitem(sys.modules, name, replacement)
#     parts = name.split(".")
#     for i in range(1, len(parts)):
#         parent = ".".join(parts[:i])
#         if parent not in sys.modules:
#             monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
#     monkeypatch.setitem(sys.modules, name, replacement)


# class _MockChat:
#     """Holds a single `.create` attr that we patch in every test."""
#     def __init__(self):
#         self.completions = SimpleNamespace(create=lambda *a, **k: None)  # patch later


# class MockOpenAI:
#     """
#     Drop-in replacement for both `openai.Client` and `openai.AsyncClient`
#     that records calls but never hits the network.
#     """
#     def __init__(self, *a, **k):
#         self.calls: list[tuple[tuple, dict]] = []
#         self.chat = _MockChat()

#     # AsyncClient is identical for our purposes
#     AsyncClient = None


# # the async version just re-uses the same implementation
# MockOpenAI.AsyncClient = MockOpenAI  # type: ignore[attr-defined]


# def fake_delta_chunk(content: str = "hi", tool: bool = False):
#     """Return a fake streaming chunk like openai passes."""
#     if tool:
#         func = SimpleNamespace(name="dummy", arguments="{}")
#         tc = SimpleNamespace(index=0, id="id", function=func)
#         delta = SimpleNamespace(tool_calls=[tc])
#     else:
#         delta = SimpleNamespace(content=content, tool_calls=None)
#     choice = SimpleNamespace(delta=delta, message=None, finish_reason=None)
#     return SimpleNamespace(choices=[choice])


# def fake_final_message(content: str = "hello", *, tool_calls=None, finish="stop"):
#     """Non-streaming final message object."""
#     message = SimpleNamespace(content=content, tool_calls=tool_calls)
#     choice = SimpleNamespace(message=message, finish_reason=finish, delta=None)
#     return SimpleNamespace(choices=[choice])


# # ------------------------------------------------------------------
# # fixtures ----------------------------------------------------------
# # ------------------------------------------------------------------


# @pytest.fixture(autouse=True)
# def patch_openai(monkeypatch):
#     """Auto-patch every test with a mocked `openai` package."""
#     mock_pkg = types.ModuleType("openai")
#     mock_pkg.Client = MockOpenAI
#     mock_pkg.AsyncClient = MockOpenAI
#     _patch_module(monkeypatch, "openai", mock_pkg)
#     yield


# @pytest.fixture
# def openai_adapters():
#     """
#     Import the adapter module **after** patching openai so the import succeeds.
#     Returns the imported module for backward compatibility tests.
#     """
#     from importlib import import_module, reload
    
#     mod = import_module("dachi.adapt.xopenai._openai")
#     return reload(mod)


# @pytest.fixture
# def chat_adapter():
#     """OpenAI Chat Completions adapter."""
#     return OpenAIChat()


# @pytest.fixture  
# def resp_adapter():
#     """OpenAI Responses adapter."""
#     return OpenAIResp()


# @pytest.fixture
# def make_tool_def():
#     """
#     Factory to create a minimal ToolDef (real class) with a random model.
#     """
#     from dachi.core import ToolDef

#     def _factory(name: str = "dummy", description: str = "desc"):
#         class _Args(pydantic.BaseModel):
#             x: int

#         def dummy_fn(x: int) -> str:
#             return f"Result: {x}"

#         return ToolDef(
#             name=name, description=description, fn=dummy_fn, input_model=_Args
#         )

#     return _factory


# # ------------------------------------------------------------------
# # 1  to_openai_tool -------------------------------------------------
# # ------------------------------------------------------------------


# # ------------------------------------------------------------------
# # 1  OpenAI Adapters -----------------------------------------------
# # ------------------------------------------------------------------


# class TestOpenAIChat:
#     """Test OpenAI Chat Completions adapter."""

#     def test_to_input_basic_message(self, chat_adapter):
#         msg = Msg(role="user", text="Hello")
#         result = chat_adapter.to_input(msg, temperature=0.7)
        
#         assert "messages" in result
#         assert result["temperature"] == 0.7
#         assert len(result["messages"]) == 1
#         assert result["messages"][0]["role"] == "user"
#         assert result["messages"][0]["content"] == "Hello"

#     def test_to_input_with_attachments(self, chat_adapter):
#         from dachi.core import Attachment
#         msg = Msg(
#             role="user",
#             text="What's in this image?", 
#             attachments=[
#                 Attachment(kind="image", data="base64data", mime="image/png")
#             ]
#         )
#         result = chat_adapter.to_input(msg)
        
#         message = result["messages"][0]
#         assert isinstance(message["content"], list)
#         assert message["content"][0]["type"] == "text"
#         assert message["content"][1]["type"] == "image_url"
#         assert "data:image/png;base64," in message["content"][1]["image_url"]["url"]

#     def test_from_output(self, chat_adapter):
#         openai_response = {
#             "id": "resp_123",
#             "model": "gpt-4o", 
#             "choices": [{
#                 "message": {"role": "assistant", "content": "Hello there!"},
#                 "finish_reason": "stop"
#             }],
#             "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5}
#         }
        
#         resp = chat_adapter.from_output(openai_response)
        
#         assert isinstance(resp, Resp)
#         assert resp.text == "Hello there!"
#         assert resp.response_id == "resp_123"
#         assert resp.model == "gpt-4o"
#         assert resp.finish_reason == "stop"
#         assert resp.usage["total_tokens"] == 10

#     def test_from_streamed(self, chat_adapter):
#         chunk = {
#             "choices": [{
#                 "delta": {"content": "Hello"},
#                 "finish_reason": None
#             }]
#         }
        
#         resp = chat_adapter.from_streamed(chunk)
        
#         assert isinstance(resp, Resp)
#         assert resp.delta.text == "Hello"
#         assert resp.delta.finish_reason is None


# class TestOpenAIResp:
#     """Test OpenAI Responses adapter."""

#     def test_to_input_single_user_message(self, resp_adapter):
#         msg = Msg(role="user", text="Hello")
#         result = resp_adapter.to_input(msg)
        
#         # Should use 'input' field for single user message
#         assert "input" in result
#         assert result["input"] == "Hello"
#         assert "messages" not in result

#     def test_to_input_with_instructions(self, resp_adapter):
#         msg = Msg(role="user", text="Hello")
#         result = resp_adapter.to_input(msg, instructions="Be helpful")
        
#         # Should use 'messages' when instructions provided
#         assert "messages" in result
#         assert result["instructions"] == "Be helpful"

#     def test_from_output_with_reasoning(self, resp_adapter):
#         openai_response = {
#             "id": "resp_123",
#             "model": "gpt-4o",
#             "reasoning": "The user is greeting me...",
#             "choices": [{
#                 "message": {"role": "assistant", "content": "Hello there!"},
#                 "finish_reason": "stop"
#             }],
#             "usage": {"total_tokens": 15}
#         }
        
#         resp = resp_adapter.from_output(openai_response)
        
#         assert resp.thinking == "The user is greeting me..."
#         assert resp.text == "Hello there!"


# class TestChatCompletionWithAdapter:
#     """Test ChatCompletion integration with AIAdapt."""
    
#     def test_forward_uses_adapter(self, openai_adapters, monkeypatch):
#         """Test that forward method properly uses adapter."""
#         # Mock the adapter methods
#         mock_adapter = OpenAIChat()
#         original_to_input = mock_adapter.to_input
#         original_from_output = mock_adapter.from_output
        
#         # Track calls to adapter methods
#         to_input_calls = []
#         from_output_calls = []
        
#         def mock_to_input(inp, **kwargs):
#             to_input_calls.append((inp, kwargs))
#             return original_to_input(inp, **kwargs)
            
#         def mock_from_output(output):
#             from_output_calls.append(output)
#             return original_from_output(output)
            
#         mock_adapter.to_input = mock_to_input
#         mock_adapter.from_output = mock_from_output
        
#         # Create ChatCompletion with custom adapter
#         chat = openai_adapters.ChatCompletion(adapt=mock_adapter)
        
#         # Mock the OpenAI client
#         fake_response = {
#             "id": "test_123",
#             "model": "gpt-4o", 
#             "choices": [{
#                 "message": {"role": "assistant", "content": "Test response"},
#                 "finish_reason": "stop"
#             }]
#         }
        
#         def mock_create(*args, **kwargs):
#             return fake_response
            
#         chat._client.chat.completions.create = mock_create
        
#         # Test forward call
#         msg = Msg(role="user", text="Hello")
#         result = chat.forward(msg, temperature=0.7)
        
#         # Verify adapter methods were called
#         assert len(to_input_calls) == 1
#         assert len(from_output_calls) == 1
#         assert to_input_calls[0][0] == msg
#         assert "temperature" in to_input_calls[0][1]
#         assert from_output_calls[0] == fake_response
        
#         # Verify result
#         from dachi.core import Resp
#         assert isinstance(result, Resp)
#         assert result.text == "Test response"
        
        
# class TestToOpenAITool:
#     """Unit-tests for the helper that converts ToolDef → OpenAI schema."""

#     def test_single(self, openai_adapters, make_tool_def):
#         tool = make_tool_def()
#         out = openai_adapters.to_openai_tool(tool)
#         assert isinstance(out, list) and len(out) == 1
#         fn = out[0]["function"]
#         assert fn["name"] == tool.name
#         assert fn["description"] == tool.description
#         # pydantic v1 & v2 produce 'properties' key in schema
#         assert "properties" in fn["parameters"]

#     def test_multiple(self, openai_adapters, make_tool_def):
#         out = openai_adapters.to_openai_tool([make_tool_def("a"), make_tool_def("b")])
#         assert [t["function"]["name"] for t in out] == ["a", "b"]

#     def test_wrong_type(self, openai_adapters):
#         # to_openai_tool should work with ToolDef or list of ToolDef
#         # Let's test with an invalid input
#         import dachi.core
#         tool_def = dachi.core.ToolDef
#         # This should fail if we pass something that's not a ToolDef
#         with pytest.raises((TypeError, AttributeError)):
#             openai_adapters.to_openai_tool(42)  # type: ignore[arg-type]


# # ------------------------------------------------------------------
# # 2  OpenAI-Specific Tool Processing -------------------------------
# # ------------------------------------------------------------------


# # ------------------------------------------------------------------
# # 4  ToolConv -------------------------------------------------------
# # ------------------------------------------------------------------


# # ToolConv tests moved to tests/proc/test_resp.py since it's now a unified processor


# # ------------------------------------------------------------------
# # 5  LLM & ChatCompletion ------------------------------------------
# # ------------------------------------------------------------------


# class TestLLMInit:
#     """Ensure converter pipeline assembled correctly."""

#     def test_text_only(self, openai_adapters):
#         llm = openai_adapters.LLM()
#         names = [type(c).__name__ for c in llm.convs]
#         assert "TextConv" in names and "ToolConv" not in names

#     def test_json_model_adds_parsed(self, openai_adapters):
#         from dachi.proc._resp import ParsedConv
#         class Foo(pydantic.BaseModel):
#             y: int
#         llm = openai_adapters.LLM(json_output=Foo)
#         assert any(isinstance(c, ParsedConv) for c in llm.convs)
        
#     def test_llm_has_default_adapter(self, openai_adapters):
#         """Test that LLM initializes with default OpenAIChat adapter."""
#         llm = openai_adapters.LLM()
#         assert llm.adapt is not None
#         assert type(llm.adapt).__name__ == "OpenAIChat"
        
#     def test_llm_custom_adapter(self, openai_adapters):
#         """Test that LLM can use custom adapter."""
#         custom_adapter = OpenAIResp()
#         llm = openai_adapters.LLM(adapt=custom_adapter)
#         assert llm.adapt is custom_adapter
#         assert type(llm.adapt).__name__ == "OpenAIResp"


# # class TestChatCompletion:
# #     """End-to-end through forward / streaming paths."""
# # 
# #     @staticmethod
# #     def _patch_response(monkeypatch, adapters, *, stream=False, chunks=None, final=None):
# #         """Install a `.create` that returns given response."""
# #         def make_async_return(val):
# #             async def coro(*a, **k): return val
# #             return coro
# # 
# #         cl: MockOpenAI = adapters.openai.Client()
# #         if stream:
# #             iterator = iter(chunks)  # type: ignore[arg-type]
# #             cl.chat.completions.create = lambda *a, **k: iterator
# #         else:
# #             cl.chat.completions.create = lambda *a, **k: final
# # 
# #         # async variant mirrors sync
# #         acl: MockOpenAI = adapters.openai.AsyncClient()
# #         if stream:
# #             async def async_iter(*a, **k):  # noqa: D401
# #                 for c in chunks:
# #                     yield c
# #             acl.chat.completions.create = async_iter
# #         else:
# #             acl.chat.completions.create = make_async_return(final)
# # 
# #         # monkeypatch ChatCompletion to use our instantiated clients
# #         monkeypatch.setattr(adapters.ChatCompletion, "_client", cl)
# #         monkeypatch.setattr(adapters.ChatCompletion, "_aclient", acl)
# # 
# #     def test_forward_text(self, adapters, monkeypatch):
# #         final = fake_final_message("hello")
# #         self._patch_response(monkeypatch, adapters, final=final)
# #         chat = adapters.ChatCompletion()
# #         msg = adapters.Msg(role="user", content="hi")  # type: ignore[call-arg]
# #         out_msg, _ = chat.forward(msg)
# #         assert out_msg.content == "hello"
# # 
# #     @pytest.mark.asyncio
# #     async def test_aforward(self, adapters, monkeypatch):
# #         final = fake_final_message("yo")
# #         self._patch_response(monkeypatch, adapters, final=final)
# #         chat = adapters.ChatCompletion()
# #         msg = adapters.Msg(role="user", content="x")  # type: ignore[call-arg]
# #         m, _ = await chat.aforward(msg)
# #         assert m.content == "yo"
# # 
# #     def test_stream_generator(self, adapters, monkeypatch):
# #         chunks = [fake_delta_chunk("h"), fake_delta_chunk("i"), adapters.END_TOK]
# #         self._patch_response(monkeypatch, adapters, stream=True, chunks=chunks)
# #         chat = adapters.ChatCompletion()
# #         msg = adapters.Msg(role="u", content="z")  # type: ignore[call-arg]
# #         resps = list(chat.stream(msg))
# #         # last yielded pair contains full text
# #         assert resps[-1][0].content == "hi"
# # 
# #     @pytest.mark.asyncio
# #     async def test_astream(self, adapters, monkeypatch):
# #         chunks = [fake_delta_chunk("o"), fake_delta_chunk("k"), adapters.END_TOK]
# #         self._patch_response(monkeypatch, adapters, stream=True, chunks=chunks)
# #         chat = adapters.ChatCompletion()
# #         msg = adapters.Msg(role="u", content="k")  # type: ignore[call-arg]
# #         out = []
# #         async for m, _ in chat.astream(msg):
# #             out.append(m.content or "")
# #         assert "".join(out).endswith("ok")  # progressive build
# # 
# # 
# # # ------------------------------------------------------------------
# # # 6  ToolExecConv ---------------------------------------------------
# # # ------------------------------------------------------------------
# # 
# # 
# # Note: TestToolExecConv moved to tests/proc/test_resp.py as it tests core functionality
