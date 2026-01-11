import pytest
from dachi.proc._lang import (
    Call,
    AsyncCall,
    StreamCall,
    AsyncStreamCall,
    ToolUse,
    LangModel,
    LangEngine,
    LangOp,
    ToolUser,
    lang_forward,
    lang_aforward,
    lang_stream,
    lang_astream,
)
from dachi.core._base import ToolResult, ToolMsg


class TestToolUseForward:
    def test_forward_executes_sync_calls(self):
        def add(a: int, b: int) -> int:
            return a + b

        def multiply(a: int, b: int) -> int:
            return a * b

        call1 = Call(func=add, args={"a": 1, "b": 2})
        call2 = Call(func=multiply, args={"a": 3, "b": 4})

        tool_use = ToolUse(text="Calculating", calls=[call1, call2])
        results = tool_use.forward()

        assert len(results) == 2
        assert all(isinstance(r, ToolResult) for r in results)

    def test_forward_executes_stream_calls(self):
        def count_gen(n: int):
            for i in range(n):
                yield str(i)

        call = StreamCall(func=count_gen, args={"n": 3})
        tool_use = ToolUse(calls=[call])

        results = tool_use.forward()

        assert len(results) == 1
        assert isinstance(results[0], ToolResult)
        assert results[0].output == "012"

    def test_forward_raises_on_async_call(self):
        async def async_add(a: int, b: int) -> int:
            return a + b

        call = AsyncCall(func=async_add, args={"a": 1, "b": 2})
        tool_use = ToolUse(calls=[call])

        with pytest.raises(ValueError, match="Cannot execute async call"):
            tool_use.forward()

    def test_forward_raises_on_async_stream_call(self):
        async def async_gen(n: int):
            for i in range(n):
                yield i

        call = AsyncStreamCall(func=async_gen, args={"n": 3})
        tool_use = ToolUse(calls=[call])

        with pytest.raises(ValueError, match="Cannot execute async streaming call"):
            tool_use.forward()


class TestToolUseAforward:
    @pytest.mark.asyncio
    async def test_aforward_executes_sync_calls(self):
        def add(a: int, b: int) -> int:
            return a + b

        call = Call(func=add, args={"a": 1, "b": 2})
        tool_use = ToolUse(calls=[call])

        results = await tool_use.aforward()

        assert len(results) == 1
        assert isinstance(results[0], ToolResult)

    @pytest.mark.asyncio
    async def test_aforward_executes_async_calls(self):
        async def async_add(a: int, b: int) -> int:
            return a + b

        call = AsyncCall(func=async_add, args={"a": 1, "b": 2})
        tool_use = ToolUse(calls=[call])

        results = await tool_use.aforward()

        assert len(results) == 1
        assert isinstance(results[0], ToolResult)

    @pytest.mark.asyncio
    async def test_aforward_executes_stream_calls(self):
        def count_gen(n: int):
            for i in range(n):
                yield str(i)

        call = StreamCall(func=count_gen, args={"n": 3})
        tool_use = ToolUse(calls=[call])

        results = await tool_use.aforward()

        assert len(results) == 1
        assert isinstance(results[0], ToolResult)
        assert results[0].output == "012"

    @pytest.mark.asyncio
    async def test_aforward_executes_async_stream_calls(self):
        async def async_count_gen(n: int):
            for i in range(n):
                yield str(i)

        call = AsyncStreamCall(func=async_count_gen, args={"n": 3})
        tool_use = ToolUse(calls=[call])

        results = await tool_use.aforward()

        assert len(results) == 1
        assert isinstance(results[0], ToolResult)
        assert results[0].output == "012"

    @pytest.mark.asyncio
    async def test_aforward_executes_mixed_call_types(self):
        def add(a: int, b: int) -> int:
            return a + b

        async def async_multiply(a: int, b: int) -> int:
            return a * b

        def count_gen(n: int):
            for i in range(n):
                yield str(i)

        call1 = Call(func=add, args={"a": 1, "b": 2})
        call2 = AsyncCall(func=async_multiply, args={"a": 3, "b": 4})
        call3 = StreamCall(func=count_gen, args={"n": 2})

        tool_use = ToolUse(calls=[call1, call2, call3])
        results = await tool_use.aforward()

        assert len(results) == 3
        assert all(isinstance(r, ToolResult) for r in results)


class TestLangForward:
    def test_lang_forward_calls_model(self):
        class MockModel(LangModel):
            def forward(self, prompt, **kwargs):
                return "test response", [], None

            async def aforward(self, prompt, **kwargs):
                pass

            def stream(self, prompt, **kwargs):
                pass

            async def astream(self, prompt, **kwargs):
                pass

        model = MockModel()
        result = lang_forward("test prompt", _model=model)

        assert result == ("test response", [], None)

    def test_lang_forward_raises_without_model(self):
        with pytest.raises(ValueError, match="Model must be provided"):
            lang_forward("test prompt", _model=None)

    def test_lang_forward_passes_structure_and_tools(self):
        class MockModel(LangModel):
            def forward(self, prompt, structure=None, tools=None, **kwargs):
                return f"structure={structure}, tools={tools}", [], None

            async def aforward(self, prompt, **kwargs):
                pass

            def stream(self, prompt, **kwargs):
                pass

            async def astream(self, prompt, **kwargs):
                pass

        model = MockModel()
        result = lang_forward(
            "test",
            structure={"type": "object"},
            tools={"add": lambda x: x},
            _model=model
        )

        assert "structure=" in result[0]
        assert "tools=" in result[0]


class TestLangAforward:
    @pytest.mark.asyncio
    async def test_lang_aforward_calls_model(self):
        class MockModel(LangModel):
            def forward(self, prompt, **kwargs):
                pass

            async def aforward(self, prompt, **kwargs):
                return "async response", [], None

            def stream(self, prompt, **kwargs):
                pass

            async def astream(self, prompt, **kwargs):
                pass

        model = MockModel()
        result = await lang_aforward("test prompt", _model=model)

        assert result == ("async response", [], None)

    @pytest.mark.asyncio
    async def test_lang_aforward_raises_without_model(self):
        with pytest.raises(ValueError, match="Model must be provided"):
            await lang_aforward("test prompt", _model=None)


class TestLangStream:
    def test_lang_stream_yields_responses(self):
        class MockModel(LangModel):
            def forward(self, prompt, **kwargs):
                pass

            async def aforward(self, prompt, **kwargs):
                pass

            def stream(self, prompt, **kwargs):
                for i in range(3):
                    yield f"chunk {i}", [], None

            async def astream(self, prompt, **kwargs):
                pass

        model = MockModel()
        results = list(lang_stream("test prompt", _model=model))

        assert len(results) >= 3

    def test_lang_stream_raises_without_model(self):
        with pytest.raises(ValueError, match="Model must be provided"):
            list(lang_stream("test prompt", _model=None))


class TestLangAstream:
    @pytest.mark.asyncio
    async def test_lang_astream_yields_responses(self):
        class MockModel(LangModel):
            def forward(self, prompt, **kwargs):
                pass

            async def aforward(self, prompt, **kwargs):
                pass

            def stream(self, prompt, **kwargs):
                pass

            async def astream(self, prompt, **kwargs):
                for i in range(3):
                    yield f"chunk {i}", [], None

        model = MockModel()
        results = []
        async for res in lang_astream("test prompt", _model=model):
            results.append(res)

        assert len(results) >= 3

    @pytest.mark.asyncio
    async def test_lang_astream_raises_without_model(self):
        with pytest.raises(ValueError, match="Model must be provided"):
            async for _ in lang_astream("test prompt", _model=None):
                pass


class TestLangEngine:
    def test_model_property_getter_and_setter(self):
        engine = LangEngine()

        assert engine.model is None

        class MockModel(LangModel):
            def forward(self, prompt, **kwargs):
                return "test", [], None

            async def aforward(self, prompt, **kwargs):
                pass

            def stream(self, prompt, **kwargs):
                pass

            async def astream(self, prompt, **kwargs):
                pass

        model = MockModel()
        engine.model = model

        assert engine.model == model

    def test_get_model_raises_when_not_set(self):
        engine = LangEngine()

        with pytest.raises(ValueError, match="Model is not set"):
            engine.get_model()

    def test_get_model_uses_override(self):
        engine = LangEngine()

        class MockModel(LangModel):
            def forward(self, prompt, **kwargs):
                return "test", [], None

            async def aforward(self, prompt, **kwargs):
                pass

            def stream(self, prompt, **kwargs):
                pass

            async def astream(self, prompt, **kwargs):
                pass

        model = MockModel()
        result = engine.get_model(override=model)

        assert result == model

    def test_forward_uses_model(self):
        class MockModel(LangModel):
            def forward(self, prompt, **kwargs):
                return "response", [], None

            async def aforward(self, prompt, **kwargs):
                pass

            def stream(self, prompt, **kwargs):
                pass

            async def astream(self, prompt, **kwargs):
                pass

        engine = LangEngine()
        model = MockModel()

        result = engine.forward("test prompt", _model=model)

        assert result == ("response", [], None)

    def test_forward_raises_without_model(self):
        engine = LangEngine()

        with pytest.raises(ValueError, match="Model is not set"):
            engine.forward("test prompt")

    @pytest.mark.asyncio
    async def test_aforward_uses_model(self):
        class MockModel(LangModel):
            def forward(self, prompt, **kwargs):
                pass

            async def aforward(self, prompt, **kwargs):
                return "async response", [], None

            def stream(self, prompt, **kwargs):
                pass

            async def astream(self, prompt, **kwargs):
                pass

        engine = LangEngine()
        model = MockModel()

        result = await engine.aforward("test prompt", _model=model)

        assert result == ("async response", [], None)


class TestLangOp:
    def test_forward_calls_model(self):
        class MockModel(LangModel):
            def forward(self, prompt, **kwargs):
                return "response", [], None

            async def aforward(self, prompt, **kwargs):
                pass

            def stream(self, prompt, **kwargs):
                pass

            async def astream(self, prompt, **kwargs):
                pass

        op = LangOp()
        model = MockModel()

        result = op.forward("test prompt", _model=model)

        assert result == "response"

    def test_forward_raises_without_model(self):
        op = LangOp()

        with pytest.raises(ValueError, match="Model must be provided"):
            op.forward("test prompt")

    @pytest.mark.asyncio
    async def test_aforward_calls_model(self):
        class MockModel(LangModel):
            def forward(self, prompt, **kwargs):
                pass

            async def aforward(self, prompt, **kwargs):
                return "async response", [], None

            def stream(self, prompt, **kwargs):
                pass

            async def astream(self, prompt, **kwargs):
                pass

        op = LangOp()
        model = MockModel()

        result = await op.aforward("test prompt", _model=model)

        assert result == "async response"

    def test_stream_yields_responses(self):
        class MockModel(LangModel):
            def forward(self, prompt, **kwargs):
                pass

            async def aforward(self, prompt, **kwargs):
                pass

            def stream(self, prompt, **kwargs):
                for i in range(3):
                    yield f"chunk {i}", [], None

            async def astream(self, prompt, **kwargs):
                pass

        op = LangOp()
        model = MockModel()

        results = list(op.stream("test prompt", _model=model))

        assert len(results) >= 3

    @pytest.mark.asyncio
    async def test_astream_yields_responses(self):
        class MockModel(LangModel):
            def forward(self, prompt, **kwargs):
                pass

            async def aforward(self, prompt, **kwargs):
                pass

            def stream(self, prompt, **kwargs):
                pass

            async def astream(self, prompt, **kwargs):
                for i in range(3):
                    yield f"chunk {i}", [], None

        op = LangOp()
        model = MockModel()

        results = []
        async for res in op.astream("test prompt", _model=model):
            results.append(res)

        assert len(results) >= 3


class TestToolUserForward:
    def test_forward_executes_tools_and_loops(self):
        def add(a: int, b: int) -> int:
            return a + b

        call_count = {"count": 0}

        class MockModel(LangModel):
            def forward(self, prompt, tools=None, **kwargs):
                call_count["count"] += 1
                if call_count["count"] == 1:
                    add_fn = tools.get("add")
                    call = Call(func=add_fn, args={"a": 1, "b": 2})
                    return ToolUse(text="Calculating", calls=[call]), [], None
                else:
                    return "Result is 3", [], None

            async def aforward(self, prompt, **kwargs):
                pass

            def stream(self, prompt, **kwargs):
                pass

            async def astream(self, prompt, **kwargs):
                pass

        tool_user = ToolUser()
        model = MockModel()

        result, msgs, raw = tool_user.forward("test prompt", tools={"add": add}, _model=model)

        assert result == "Result is 3"
        assert call_count["count"] == 2

    def test_forward_stops_when_no_tool_use(self):
        class MockModel(LangModel):
            def forward(self, prompt, **kwargs):
                return "Direct response", [], None

            async def aforward(self, prompt, **kwargs):
                pass

            def stream(self, prompt, **kwargs):
                pass

            async def astream(self, prompt, **kwargs):
                pass

        tool_user = ToolUser()
        model = MockModel()

        result, msgs, raw = tool_user.forward("test prompt", _model=model)

        assert result == "Direct response"

    def test_forward_raises_on_max_iterations(self):
        def add(a: int, b: int) -> int:
            return a + b

        class MockModel(LangModel):
            def forward(self, prompt, tools=None, **kwargs):
                add_fn = tools.get("add")
                call = Call(func=add_fn, args={"a": 1, "b": 2})
                return ToolUse(text="Looping", calls=[call]), [], None

            async def aforward(self, prompt, **kwargs):
                pass

            def stream(self, prompt, **kwargs):
                pass

            async def astream(self, prompt, **kwargs):
                pass

        tool_user = ToolUser()
        model = MockModel()

        with pytest.raises(RuntimeError, match="Max tool use iterations"):
            tool_user.forward("test prompt", tools={"add": add}, _model=model)

    def test_forward_invokes_callback(self):
        def add(a: int, b: int) -> int:
            return a + b

        callback_invocations = []
        call_count = {"count": 0}

        def callback(res, tool_results, iteration, msgs):
            callback_invocations.append({
                "res": res,
                "tool_results": tool_results,
                "iteration": iteration,
                "msgs": msgs
            })

        class MockModel(LangModel):
            def forward(self, prompt, tools=None, **kwargs):
                call_count["count"] += 1
                if call_count["count"] == 1:
                    add_fn = tools.get("add")
                    call = Call(func=add_fn, args={"a": 1, "b": 2})
                    return ToolUse(text="Calculating", calls=[call]), [], None
                else:
                    return "Done", [], None

            async def aforward(self, prompt, **kwargs):
                pass

            def stream(self, prompt, **kwargs):
                pass

            async def astream(self, prompt, **kwargs):
                pass

        tool_user = ToolUser()
        model = MockModel()

        tool_user.forward("test prompt", tools={"add": add}, _model=model, _callback=callback)

        assert len(callback_invocations) == 1
        assert callback_invocations[0]["iteration"] == 0
        assert isinstance(callback_invocations[0]["res"], ToolUse)
        assert len(callback_invocations[0]["tool_results"]) == 1


class TestToolUserAforward:
    @pytest.mark.asyncio
    async def test_aforward_executes_tools_and_loops(self):
        async def async_add(a: int, b: int) -> int:
            return a + b

        call_count = {"count": 0}

        class MockModel(LangModel):
            def forward(self, prompt, **kwargs):
                pass

            async def aforward(self, prompt, tools=None, **kwargs):
                call_count["count"] += 1
                if call_count["count"] == 1:
                    add_fn = tools.get("async_add")
                    call = AsyncCall(func=add_fn, args={"a": 1, "b": 2})
                    return ToolUse(text="Calculating", calls=[call]), [], None
                else:
                    return "Result is 3", [], None

            def stream(self, prompt, **kwargs):
                pass

            async def astream(self, prompt, **kwargs):
                pass

        tool_user = ToolUser()
        model = MockModel()

        result, msgs, raw = await tool_user.aforward("test prompt", tools={"async_add": async_add}, _model=model)

        assert result == "Result is 3"
        assert call_count["count"] == 2

    @pytest.mark.asyncio
    async def test_aforward_raises_on_max_iterations(self):
        async def async_add(a: int, b: int) -> int:
            return a + b

        class MockModel(LangModel):
            def forward(self, prompt, **kwargs):
                pass

            async def aforward(self, prompt, tools=None, **kwargs):
                add_fn = tools.get("async_add")
                call = AsyncCall(func=add_fn, args={"a": 1, "b": 2})
                return ToolUse(text="Looping", calls=[call]), [], None

            def stream(self, prompt, **kwargs):
                pass

            async def astream(self, prompt, **kwargs):
                pass

        tool_user = ToolUser()
        model = MockModel()

        with pytest.raises(RuntimeError, match="Max tool use iterations"):
            await tool_user.aforward("test prompt", tools={"async_add": async_add}, _model=model)


class TestToolUserStream:
    def test_stream_yields_chunks_and_loops(self):
        def add(a: int, b: int) -> int:
            return a + b

        call_count = {"count": 0}

        class MockModel(LangModel):
            def forward(self, prompt, **kwargs):
                pass

            async def aforward(self, prompt, **kwargs):
                pass

            def stream(self, prompt, tools=None, **kwargs):
                call_count["count"] += 1
                if call_count["count"] == 1:
                    add_fn = tools.get("add")
                    call = Call(func=add_fn, args={"a": 1, "b": 2})
                    yield "chunk1", [], None
                    yield "chunk2", [], None
                    yield ToolUse(text="Calculating", calls=[call]), [], None
                else:
                    yield "final", [], None

            async def astream(self, prompt, **kwargs):
                pass

        tool_user = ToolUser()
        model = MockModel()

        results = list(tool_user.stream("test prompt", tools={"add": add}, _model=model))

        assert call_count["count"] == 2
        assert results[0][0] == "chunk1"
        assert results[1][0] == "chunk2"
        assert isinstance(results[2][0], ToolUse)
        assert results[3][0] == "final"

    def test_stream_raises_on_max_iterations(self):
        def add(a: int, b: int) -> int:
            return a + b

        class MockModel(LangModel):
            def forward(self, prompt, **kwargs):
                pass

            async def aforward(self, prompt, **kwargs):
                pass

            def stream(self, prompt, tools=None, **kwargs):
                add_fn = tools.get("add")
                call = Call(func=add_fn, args={"a": 1, "b": 2})
                yield ToolUse(text="Looping", calls=[call]), [], None

            async def astream(self, prompt, **kwargs):
                pass

        tool_user = ToolUser()
        model = MockModel()

        with pytest.raises(RuntimeError, match="Max tool use iterations"):
            list(tool_user.stream("test prompt", tools={"add": add}, _model=model))


class TestToolUserAstream:
    @pytest.mark.asyncio
    async def test_astream_yields_chunks_and_loops(self):
        async def async_add(a: int, b: int) -> int:
            return a + b

        call_count = {"count": 0}

        class MockModel(LangModel):
            def forward(self, prompt, **kwargs):
                pass

            async def aforward(self, prompt, **kwargs):
                pass

            def stream(self, prompt, **kwargs):
                pass

            async def astream(self, prompt, tools=None, **kwargs):
                call_count["count"] += 1
                if call_count["count"] == 1:
                    add_fn = tools.get("async_add")
                    call = AsyncCall(func=add_fn, args={"a": 1, "b": 2})
                    yield "chunk1", [], None
                    yield "chunk2", [], None
                    yield ToolUse(text="Calculating", calls=[call]), [], None
                else:
                    yield "final", [], None

        tool_user = ToolUser()
        model = MockModel()

        results = []
        async for res in tool_user.astream("test prompt", tools={"async_add": async_add}, _model=model):
            results.append(res)

        assert call_count["count"] == 2
        assert results[0][0] == "chunk1"
        assert results[1][0] == "chunk2"
        assert isinstance(results[2][0], ToolUse)
        assert results[3][0] == "final"

    @pytest.mark.asyncio
    async def test_astream_raises_on_max_iterations(self):
        async def async_add(a: int, b: int) -> int:
            return a + b

        class MockModel(LangModel):
            def forward(self, prompt, **kwargs):
                pass

            async def aforward(self, prompt, **kwargs):
                pass

            def stream(self, prompt, **kwargs):
                pass

            async def astream(self, prompt, tools=None, **kwargs):
                add_fn = tools.get("async_add")
                call = AsyncCall(func=add_fn, args={"a": 1, "b": 2})
                yield ToolUse(text="Looping", calls=[call]), [], None

        tool_user = ToolUser()
        model = MockModel()

        with pytest.raises(RuntimeError, match="Max tool use iterations"):
            async for _ in tool_user.astream("test prompt", tools={"async_add": async_add}, _model=model):
                pass
