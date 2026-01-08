import pytest
import typing as t
from dachi.proc import (
    Call,
    AsyncCall,
    StreamCall,
    AsyncStreamCall,
    BaseToolCall,
    ToolUse,
    LangModel,
    Process,
    AsyncProcess,
    StreamProcess,
    AsyncStreamProcess,
)
from dachi.core import Inp


class TestCall:
    def test_call_forward_executes_function(self):
        def add(a: int, b: int) -> int:
            return a + b

        call = Call(func=add, args={"a": 1, "b": 2})
        result = call.forward()

        assert result == 3

    def test_call_inherits_from_base_tool_call(self):
        def dummy():
            pass

        call = Call(func=dummy, args={})

        assert isinstance(call, BaseToolCall)
        assert isinstance(call, Process)

    def test_call_with_empty_args(self):
        def no_args():
            return "success"

        call = Call(func=no_args)
        result = call.forward()

        assert result == "success"

    def test_call_with_complex_args(self):
        def process(data: dict, multiplier: int) -> int:
            return data["value"] * multiplier

        call = Call(func=process, args={"data": {"value": 5}, "multiplier": 3})
        result = call.forward()

        assert result == 15


class TestAsyncCall:
    @pytest.mark.asyncio
    async def test_async_call_aforward_executes_async_function(self):
        async def async_add(a: int, b: int) -> int:
            return a + b

        call = AsyncCall(func=async_add, args={"a": 1, "b": 2})
        result = await call.aforward()

        assert result == 3

    @pytest.mark.asyncio
    async def test_async_call_inherits_from_base_tool_call(self):
        async def dummy():
            pass

        call = AsyncCall(func=dummy, args={})

        assert isinstance(call, BaseToolCall)
        assert isinstance(call, AsyncProcess)

    @pytest.mark.asyncio
    async def test_async_call_with_empty_args(self):
        async def no_args():
            return "async success"

        call = AsyncCall(func=no_args)
        result = await call.aforward()

        assert result == "async success"


class TestStreamCall:
    def test_stream_call_stream_yields_values(self):
        def count_gen(n: int):
            for i in range(n):
                yield i

        call = StreamCall(func=count_gen, args={"n": 3})
        result = list(call.stream())

        assert result == [0, 1, 2]

    def test_stream_call_inherits_from_base_tool_call(self):
        def dummy_gen():
            yield 1

        call = StreamCall(func=dummy_gen, args={})

        assert isinstance(call, BaseToolCall)
        assert isinstance(call, StreamProcess)

    def test_stream_call_with_multiple_yields(self):
        def generate_sequence(start: int, end: int):
            for i in range(start, end):
                yield i * 2

        call = StreamCall(func=generate_sequence, args={"start": 1, "end": 4})
        result = list(call.stream())

        assert result == [2, 4, 6]


class TestAsyncStreamCall:
    @pytest.mark.asyncio
    async def test_async_stream_call_astream_yields_values(self):
        async def async_count_gen(n: int):
            for i in range(n):
                yield i

        call = AsyncStreamCall(func=async_count_gen, args={"n": 3})
        result = []
        async for item in call.astream():
            result.append(item)

        assert result == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_async_stream_call_inherits_from_base_tool_call(self):
        async def dummy_async_gen():
            yield 1

        call = AsyncStreamCall(func=dummy_async_gen, args={})

        assert isinstance(call, BaseToolCall)
        assert isinstance(call, AsyncStreamProcess)

    @pytest.mark.asyncio
    async def test_async_stream_call_with_multiple_yields(self):
        async def async_generate_sequence(start: int, end: int):
            for i in range(start, end):
                yield i * 2

        call = AsyncStreamCall(func=async_generate_sequence, args={"start": 1, "end": 4})
        result = []
        async for item in call.astream():
            result.append(item)

        assert result == [2, 4, 6]


class TestToolUse:
    def test_tool_use_empty_initialization(self):
        tool_use = ToolUse()

        assert tool_use.text == ""
        assert tool_use.calls == []

    def test_tool_use_with_text_and_calls(self):
        def add(a: int, b: int) -> int:
            return a + b

        call = Call(func=add, args={"a": 1, "b": 2})
        tool_use = ToolUse(text="Let me calculate that", calls=[call])

        assert tool_use.text == "Let me calculate that"
        assert len(tool_use.calls) == 1
        assert isinstance(tool_use.calls[0], BaseToolCall)

    def test_tool_use_contains_multiple_calls(self):
        def add(a: int, b: int) -> int:
            return a + b

        async def async_multiply(a: int, b: int) -> int:
            return a * b

        call1 = Call(func=add, args={"a": 1, "b": 2})
        call2 = AsyncCall(func=async_multiply, args={"a": 3, "b": 4})

        tool_use = ToolUse(text="Processing", calls=[call1, call2])

        assert len(tool_use.calls) == 2
        assert all(isinstance(c, BaseToolCall) for c in tool_use.calls)
        assert isinstance(tool_use.calls[0], Call)
        assert isinstance(tool_use.calls[1], AsyncCall)

    def test_tool_use_execution_pattern(self):
        def add(a: int, b: int) -> int:
            return a + b

        call = Call(func=add, args={"a": 5, "b": 3})
        tool_use = ToolUse(text="Calculating sum", calls=[call])

        for call_item in tool_use.calls:
            if isinstance(call_item, Call):
                result = call_item.forward()
                assert result == 8


class TestLangModelIntegration:
    def test_mock_lang_model_returns_tool_use(self):
        class MockLangModel(LangModel):
            def forward(self, prompt, tools=None, **kwargs):
                add_fn = tools.get("add")
                call = Call(func=add_fn, args={"a": 1, "b": 2})
                return ToolUse(text="Calculating", calls=[call]), [], None

            async def aforward(self, prompt, tools=None, **kwargs):
                pass

            def stream(self, prompt, tools=None, **kwargs):
                pass

            async def astream(self, prompt, tools=None, **kwargs):
                pass

        def add(a: int, b: int) -> int:
            return a + b

        model = MockLangModel()
        result, msgs, raw = model.forward("What is 1+2?", tools={"add": add})

        assert isinstance(result, ToolUse)
        assert result.text == "Calculating"
        assert len(result.calls) == 1

        output = result.calls[0].forward()
        assert output == 3

    def test_mock_lang_model_returns_string(self):
        class MockLangModel(LangModel):
            def forward(self, prompt, tools=None, **kwargs):
                return "This is a text response", [], None

            async def aforward(self, prompt, tools=None, **kwargs):
                pass

            def stream(self, prompt, tools=None, **kwargs):
                pass

            async def astream(self, prompt, tools=None, **kwargs):
                pass

        model = MockLangModel()
        result, msgs, raw = model.forward("Tell me a joke")

        assert isinstance(result, str)
        assert result == "This is a text response"

    @pytest.mark.asyncio
    async def test_mock_lang_model_async_returns_tool_use(self):
        class MockLangModel(LangModel):
            def forward(self, prompt, tools=None, **kwargs):
                pass

            async def aforward(self, prompt, tools=None, **kwargs):
                add_fn = tools.get("add")
                call = AsyncCall(func=add_fn, args={"a": 1, "b": 2})
                return ToolUse(text="Calculating async", calls=[call]), [], None

            def stream(self, prompt, tools=None, **kwargs):
                pass

            async def astream(self, prompt, tools=None, **kwargs):
                pass

        async def add(a: int, b: int) -> int:
            return a + b

        model = MockLangModel()
        result, msgs, raw = await model.aforward("What is 1+2?", tools={"add": add})

        assert isinstance(result, ToolUse)
        assert result.text == "Calculating async"

        output = await result.calls[0].aforward()
        assert output == 3
