from dachi.core._msg import Msg, Resp
from dachi.core import END_TOK, ModuleList, BaseDialog
from dachi.proc import _ai
from dachi.proc._resp import ToOut, TextOut
import typing as t
from dachi.proc.openai import OpenAIChat, OpenAIResp
from dachi import utils
import typing
import types
import sys
import pytest
from types import SimpleNamespace


class DummyAIModel(_ai.LLM):
    """Dummy LLM adapter for testing purposes.
    
    Simulates LLM responses by returning a fixed target string.
    Supports both complete and streaming responses.
    """

    target: str = 'Great!'

    def forward(
        self, 
        inp: Msg | BaseDialog, 
        out=None, 
        **kwargs
    ) -> Resp:
        """Run a standard query to the API

        Args:
            data : Data to pass to the API

        Returns:
            typing.Dict: The result of the API call
        """
        resp = Resp(
            msg=Msg(
            role='assistant', 
            text=self.target
        ))
        resp.data['response'] = self.target
        resp.data['content'] = self.target
        return resp
    
    def stream(
        self, 
        inp: Msg | BaseDialog, 
        out=None, 
        tools=None, 
        **kwargs
    ) -> t.Iterator[Resp]:

        cur_out = ''
        resp = Resp()
        
        for i, c in enumerate(self.target):
            cur_out += c
            is_last = i == len(self.target) - 1
        
            resp = resp.spawn(
                msg=Msg(
                role='assistant', text=cur_out
            ))
            resp.data['response'] = c
            resp.data['content'] = c
            resp.delta.text = c  # Set the individual character as delta
            yield resp
    
        resp = resp.spawn(msg=Msg(
            role='assistant', text=self.target
        ))
        resp.data['response'] = END_TOK
        resp.data['content'] = ''
        resp.delta.text = None  # Explicitly set delta.text to None for END_TOK
        
        yield resp
        
    async def aforward(self, dialog, **kwarg_overrides):
        return self.delta(dialog, **kwarg_overrides)

    async def astream(self, dialog, **kwarg_overrides):
        
        for msg in self.stream(dialog, **kwarg_overrides):

            yield msg


def forward(msg: str) -> typing.Dict:
    """Use to test forward
    """
    return {'content': f'Hi! {msg}'}


async def aforward(msg: str) -> typing.Dict:
    """Use to test aforward
    """

    return {'content': f'Hi! {msg}'}


def stream(msg: str) -> typing.Iterator[typing.Dict]:
    """Use to test stream
    """

    response = f'Hi! {msg}'
    for c in response:
        yield {'content': c}


async def astream(
    msg: str
) -> typing.AsyncIterator[typing.Dict]:
    """Use to test astream
    """

    response = f'Hi! {msg}'
    for c in response:
        yield {'content': c}



class DeltaOut(ToOut):

    def render(self, data: typing.Any) -> str:
        return str(data)
    
    def template(self) -> str:
        return "{content}"
    
    def example(self) -> str:
        return "example text"
    
    def forward(self, resp: str | None) -> typing.Any:
        # For the new API, we just return the text content directly
        return resp if resp is not None else ''

    def delta(
        self, 
        resp: str | None, 
        delta_store: typing.Dict, 
        is_last: bool=True
    ) -> typing.Any: 
        # For the new API, resp is already the delta text content
        # Return empty string when resp is None (indicating END_TOK)
        if resp is None:
            return ''
        return resp


class TestLLM:

    def test_llm_executes_forward(self):
        res = _ai.llm_forward(forward, 'Jack')
        assert res.data['response'] == {'content': 'Hi! Jack'}

    def test_llm_executes_forward_with_processor(self):
        res = _ai.llm_forward(
            forward, 'Jack', 
            out=TextOut()
        )
        assert res.data['response'] == {'content': 'Hi! Jack'}
        assert res.out == 'Hi! Jack'

    def test_llm_executes_stream_with_processor(self):
        responses = []
        contents = []
        for r in _ai.llm_stream(
            stream, 'Jack', out=TextOut()
        ):
            responses.append(r.data['response'])
            contents.append(r.out)
        assert contents[0] == 'H'
        assert contents[-1] == ''

    def test_llm_executes_stream_with_two_processors(self):
        responses = []
        contents = []
        deltas = []
        for r in _ai.llm_stream(
            stream, 'Jack', out=(TextOut(), DeltaOut())
        ):
            print('R: ', type(r))
            responses.append(r)
            contents.append(r.msg.text)
            deltas.append(r.out[1])
        assert contents[0] == 'H'
        assert contents[-1] == 'Hi! Jack'
        assert deltas[-1] is ''
        assert responses[-1].msg.text == 'Hi! Jack'


class MockOpenAI:
    def __init__(self, *args, **kwargs):
        self.calls: list[tuple[tuple, dict]] = []
        self.chat = SimpleNamespace()
        self.chat.completions = SimpleNamespace(create=lambda *args, **kwargs: None)
        self.responses = SimpleNamespace(create=lambda *args, **kwargs: None)


class TestOpenAIChat:

    def test_to_input_basic_message(self):
        adapter = OpenAIChat()
        msg = Msg(role="user", text="Hello")
        result = adapter.to_input(msg, temperature=0.7)
        
        assert "messages" in result
        assert result["temperature"] == 0.7
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello"

    def test_from_output_captures_all_fields(self):
        adapter = OpenAIChat()
        openai_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o",
            "system_fingerprint": "fp_44709d6fcb",
            "service_tier": "default",
            "choices": [{
                "message": {"role": "assistant", "content": "Hello there!"},
                "finish_reason": "stop",
                "index": 0,
                "logprobs": {
                    "content": [{
                        "token": "Hello",
                        "logprob": -0.0001,
                        "bytes": [72, 101, 108, 108, 111]
                    }]
                }
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        }
        
        resp = adapter.to_output(openai_response)
        
        assert isinstance(resp, Resp)
        assert resp.msg.text == "Hello there!"
        assert resp.response_id == "chatcmpl-123"
        assert resp.model == "gpt-4o"
        assert resp.finish_reason == "stop"
        assert resp.usage["total_tokens"] == 21
        assert resp.usage["prompt_tokens"] == 9
        assert resp.usage["completion_tokens"] == 12
        assert "object" in resp.meta
        assert "created" in resp.meta
        assert "system_fingerprint" in resp.meta
        assert resp.meta["service_tier"] == "default"
        assert resp.logprobs is not None
        assert resp.logprobs["content"][0]["token"] == "Hello"

    def test_from_streamed_captures_deltas(self):
        adapter = OpenAIChat()
        chunk = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4o",
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [{
                "delta": {"content": "Hello"},
                "finish_reason": None,
                "index": 0
            }]
        }
        
        resp = adapter.from_streamed(chunk)
        
        assert isinstance(resp, Resp)
        assert resp.delta.text == "Hello"
        assert resp.delta.finish_reason is None


class TestOpenAIResp:

    def test_to_input_single_user_message(self):
        adapter = OpenAIResp()
        msg = Msg(role="user", text="Hello")
        result = adapter.to_input(msg)
        
        assert "input" in result
        assert result["input"] == "Hello"
        assert "messages" not in result

    def test_to_input_with_instructions(self):
        adapter = OpenAIResp()
        msg = Msg(role="user", text="Hello")
        result = adapter.to_input(msg, instructions="Be helpful")
        
        assert "input" in result
        assert result["instructions"] == "Be helpful"
        # When instructions are provided, input should be an array of messages
        assert isinstance(result["input"], list)

    def test_from_output_captures_all_fields_including_reasoning(self):
        adapter = OpenAIResp()
        openai_response = {
            "id": "resp_123",
            "object": "response",
            "created": 1677652288,
            "model": "gpt-4o",
            "system_fingerprint": "fp_44709d6fcb",
            "service_tier": "default",
            "reasoning": "The user is greeting me, so I should respond politely.",
            "choices": [{
                "message": {"role": "assistant", "content": "Hello there!"},
                "finish_reason": "stop",
                "index": 0,
                "logprobs": {
                    "content": [{
                        "token": "Hello",
                        "logprob": -0.0002
                    }]
                }
            }],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 10,
                "total_tokens": 25
            }
        }
        
        resp = adapter.to_output(openai_response)
        
        assert resp.thinking == "The user is greeting me, so I should respond politely."
        assert resp.msg.text == "Hello there!"
        assert resp.response_id == "resp_123"
        assert resp.model == "gpt-4o"
        assert resp.finish_reason == "stop"
        assert resp.usage["total_tokens"] == 25
        assert resp.usage["prompt_tokens"] == 15
        assert resp.usage["completion_tokens"] == 10
        assert "object" in resp.meta
        assert "created" in resp.meta
        assert "system_fingerprint" in resp.meta
        assert resp.meta["service_tier"] == "default"
        assert resp.logprobs is not None
        assert resp.logprobs["content"][0]["token"] == "Hello"
        assert resp.msg.id == "resp_123"

    def test_from_streamed_with_reasoning_deltas(self):
        adapter = OpenAIResp()
        chunk = {
            "id": "resp_123",
            "object": "response.delta",
            "created": 1677652288,
            "model": "gpt-4o",
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [{
                "delta": {
                    "content": "Hello",
                    "reasoning": "The user greeted me"
                },
                "finish_reason": None,
                "index": 0
            }]
        }
        
        resp = adapter.from_streamed(chunk)
        
        assert resp.delta.text == "Hello"
        assert resp.delta.thinking == "The user greeted me"
        assert resp.msg.id == "resp_123"




class TestStreamingUsageStats:
    """Test per-chunk usage statistics in streaming responses."""

    def test_chat_streaming_with_usage_stats(self):
        adapter = OpenAIChat()
        chunk = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "choices": [{
                "delta": {"content": "Hello"},
                "finish_reason": None
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "total_tokens": 11
            }
        }
        
        resp = adapter.from_streamed(chunk)
        
        assert resp.delta.text == "Hello"
        assert resp.delta.usage is not None
        assert resp.delta.usage["prompt_tokens"] == 10
        assert resp.delta.usage["completion_tokens"] == 1
        assert resp.delta.usage["total_tokens"] == 11

    def test_resp_streaming_with_usage_stats(self):
        adapter = OpenAIResp()
        chunk = {
            "id": "resp_123",
            "object": "response.delta",
            "choices": [{
                "delta": {
                    "content": "Hello",
                    "reasoning": "User greeting"
                },
                "finish_reason": None
            }],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 2,
                "total_tokens": 17
            }
        }
        
        resp = adapter.from_streamed(chunk)
        
        assert resp.delta.text == "Hello"
        assert resp.delta.thinking == "User greeting"
        assert resp.delta.usage is not None
        assert resp.delta.usage["total_tokens"] == 17


class TestMultipleCompletions:
    """Test multiple completions support (n > 1)."""

    def test_chat_multiple_completions_captures_choices(self):
        adapter = OpenAIChat()
        openai_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "First completion"},
                    "finish_reason": "stop",
                    "logprobs": {"content": [{"token": "First", "logprob": -0.1}]}
                },
                {
                    "index": 1,
                    "message": {"role": "assistant", "content": "Second completion"},
                    "finish_reason": "stop",
                    "logprobs": {"content": [{"token": "Second", "logprob": -0.2}]}
                }
            ],
            "usage": {"total_tokens": 20}
        }
        
        resp = adapter.to_output(openai_response)
        
        # Main response uses first choice by default
        assert resp.msg.text == "First completion"
        assert resp.finish_reason == "stop"
        
        # All choices metadata captured
        assert resp.choices is not None
        assert len(resp.choices) == 2
        assert resp.choices[0]["index"] == 0
        assert resp.choices[0]["finish_reason"] == "stop"
        assert resp.choices[0]["logprobs"]["content"][0]["token"] == "First"
        assert resp.choices[1]["index"] == 1
        assert resp.choices[1]["finish_reason"] == "stop"
        assert resp.choices[1]["logprobs"]["content"][0]["token"] == "Second"

    def test_resp_multiple_completions_captures_choices(self):
        adapter = OpenAIResp()
        openai_response = {
            "id": "resp_123",
            "object": "response",
            "model": "gpt-4o",
            "reasoning": "Multiple ways to respond",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Option A"},
                    "finish_reason": "stop"
                },
                {
                    "index": 1,
                    "message": {"role": "assistant", "content": "Option B"},
                    "finish_reason": "length"
                }
            ],
            "usage": {"total_tokens": 15}
        }
        
        resp = adapter.to_output(openai_response)
        
        # Main response uses first choice
        assert resp.msg.text == "Option A"
        assert resp.finish_reason == "stop"
        assert resp.thinking == "Multiple ways to respond"
        
        # All choices captured
        assert resp.choices is not None
        assert len(resp.choices) == 2
        assert resp.choices[0]["index"] == 0
        assert resp.choices[0]["finish_reason"] == "stop"
        assert resp.choices[1]["index"] == 1
        assert resp.choices[1]["finish_reason"] == "length"

    def test_single_completion_still_has_choices_array(self):
        adapter = OpenAIChat()
        openai_response = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Single response"},
                "finish_reason": "stop"
            }],
            "usage": {"total_tokens": 10}
        }
        
        resp = adapter.to_output(openai_response)
        
        assert resp.msg.text == "Single response"
        assert resp.choices is not None
        assert len(resp.choices) == 1
        assert resp.choices[0]["index"] == 0
        assert resp.choices[0]["finish_reason"] == "stop"
