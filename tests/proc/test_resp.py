from dachi.core import Msg, Resp, ToolDef, ToolCall, ToolBuilder
from dachi.proc import _resp as _resp, TextConv, StructConv, ParsedConv, ToolExecConv
from dachi.proc._resp import ToolConv, StructStreamConv
from dachi import utils
import json
import pytest
import pydantic
from types import SimpleNamespace


class EchoProc(_resp.RespProc):
    """Pass-through; copies 'response' into 'echo'."""

    def __init__(self):
        self.name = "echo"
        self.from_ = "response"
        self.__post_init__()  # sets _single

    def delta(self, resp, delta_store, is_streamed=False, is_last=True):
        return resp


class SumProc(_resp.RespProc):
    """Adds numeric fields 'a' and 'b'."""

    def __init__(self):
        self.name = "sum"
        self.from_ = ["a", "b"]
        self.__post_init__()

    def delta(
        self, 
        resp, 
        delta_store, 
        is_streamed=False, 
        is_last=True
    ):
        return sum(resp)


class ConcatProc(_resp.RespProc):
    """Concatenates chunks across streaming."""

    def __init__(self):
        self.name = "concat"
        self.from_ = "chunk"
        self.__post_init__()

    def delta(self, chunk, delta_store, is_streamed=False, is_last=True):
        buf = delta_store.get("buf", "") + chunk
        if is_last:
            return buf
        delta_store["buf"] = buf
        return utils.UNDEFINED


class UpperConv(_resp.RespProc):
    """Upper-cases the assistant 'response'."""

    def __init__(self):
        self.name = "upper"
        self.from_ = "response"
        self.__post_init__()

    def delta(self, txt, delta_store, is_streamed=False, is_last=True):
        return txt.upper()


# RespProc basic forwarding
class TestRespProcForward:
    def test_single_source_pass_through(self):
        resp = Resp(msg=Msg(role="assistant"))
        resp.data["response"] = "hello"

        out = EchoProc()(resp)
        assert out.data["echo"] == "hello"

    def test_multi_source_sum(self):
        resp = Resp(msg=Msg(role="assistant"))
        resp.data.update(a=1, b=2)

        out = SumProc()(resp)
        assert out.data["sum"] == 3

    def test_returns_undefined_if_all_inputs_undefined(self):
        resp = Resp(msg=Msg(role="assistant"))
        resp.data.update(a=utils.UNDEFINED, b=utils.UNDEFINED)

        out = SumProc()(resp)
        assert out is utils.UNDEFINED


# Streaming behaviour with delta_store
class TestRespProcStreaming:

    def test_concat_stream(self):
        proc = ConcatProc()
        ds = {}

        # 1st chunk
        r1 = Resp(msg=Msg(role="assistant"))
        r1.data["chunk"] = "Hel"
        res1 = proc.forward(r1, is_streamed=True, is_last=False)
        assert res1.data["concat"] is utils.UNDEFINED

        # 2nd / final chunk
        r2 = r1.spawn(
            msg=Msg(role="assistant")
        )
        # r2 = Resp(msg=Msg(role="assistant"))
        r2.data["chunk"] = "lo"
        res2 = proc.forward(r2, is_streamed=True, is_last=True)
        assert res2.data["concat"] == "Hello"


# RespProc.run helper
class TestRespProcRun:
    def test_run_with_list(self):
        resp = Resp(msg=Msg(role="assistant"))
        resp.data["response"] = "abc"
        resp.data.update(a=1, b=2)

        out = _resp.RespProc.run(resp, [EchoProc(), SumProc()])
        assert out.data["echo"] == "abc"
        assert out.data["sum"] == 3

    def test_run_with_single_proc(self):
        resp = Resp(msg=Msg(role="assistant"))
        resp.data["response"] = "xyz"

        out = _resp.RespProc.run(resp, EchoProc())
        assert out.data["echo"] == "xyz"


# RespConv behaviour
class TestRespConv:
    def test_upper_conv(self):
        resp = Resp(msg=Msg(role="assistant"))
        resp.data["response"] = "hello"

        out = UpperConv()(resp)
        assert out.data["upper"] == "HELLO"


# FromResp utility
class TestFromResp:

    def test_fromresp_tuple(self):
        resp = Resp(msg=Msg(role="assistant"))
        resp.out.update(
            {"a": "A", "b": "B"}
        )

        fr = _resp.FromResp(keys=["a", "b"], as_dict=False)
        assert fr(resp) == ("A", "B")

    def test_fromresp_dict(self):
        resp = Resp(msg=Msg(role="assistant"))
        resp.out.update(
            {"a": "A", "b": "B"}
        )

        fr = _resp.FromResp(keys=["a", "b"], as_dict=True)
        assert fr(resp) == {"a": "A", "b": "B"}


# ------------------------------------------------------------------
# Unified Response Processors -------------------------------------
# ------------------------------------------------------------------


class TestTextConv:
    """
    Validate delta & post logic for TextConv with unified response format.
    """

    def test_non_stream(self):
        conv = TextConv()
        delta_store = {}
        resp_text = "hello"
        
        result = conv.delta(resp_text, delta_store, streamed=False, is_last=True)
        
        # Create mock response and test post processing
        resp = Resp(msg=Msg(role="assistant"))
        conv.post(resp, result, delta_store, streamed=False, is_last=True)
        
        assert delta_store["all_content"] == "hello"
        assert resp.msg.text == "hello"

    def test_stream_chunks_accumulate(self):
        conv = TextConv()
        delta_store = {}
        
        # Process streaming chunks
        conv.delta("he", delta_store, streamed=True, is_last=False)
        conv.delta("llo", delta_store, streamed=True, is_last=False)
        
        # Create mock response and test post processing
        resp = Resp(msg=Msg(role="assistant"))
        conv.post(resp, None, delta_store, streamed=True, is_last=True)
        
        assert delta_store["all_content"] == "hello"
        assert resp.msg.text == "hello"

    @pytest.mark.parametrize("payload", [None, ""])
    def test_none_empty(self, payload):
        conv = TextConv()
        delta_store = {}
        
        result = conv.delta(payload, delta_store, streamed=False, is_last=True)
        
        resp = Resp(msg=Msg(role="assistant"))
        conv.post(resp, result, delta_store, streamed=False, is_last=True)
        
        assert delta_store["all_content"] == ""
        assert resp.msg.text == ""


class TestStructConv:
    """Exercise StructConv in all supported modes with unified format."""

    def test_default_json_object(self):
        conv = StructConv()
        assert conv.prep() == {"response_format": "json_object"}

    def test_custom_schema_dict(self):
        schema = {"type": "object", "properties": {"foo": {"type": "string"}}}
        conv = StructConv(struct=schema)
        out = conv.prep()
        assert out["response_format"]["json_schema"] == schema

    def test_custom_pydantic_model(self):
        class Foo(pydantic.BaseModel):
            bar: int

        conv = StructConv(struct=Foo)
        fmt = conv.prep()
        assert fmt["response_format"] == Foo

    def test_json_parsing(self):
        conv = StructConv()
        delta_store = {}
        json_text = '{"name": "test", "value": 42}'
        
        result = conv.delta(json_text, delta_store, streamed=False, is_last=True)
        
        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 42


class TestParsedConv:
    """ParsedConv parses final JSON into model with unified format."""

    def test_valid_json(self):
        class Foo(pydantic.BaseModel):
            bar: int

        conv = ParsedConv(struct=Foo)
        delta_store = {}
        json_text = json.dumps({"bar": 3})
        
        result = conv.delta(json_text, delta_store, streamed=False, is_last=True)
        
        assert isinstance(result, Foo) and result.bar == 3

    def test_bad_json_returns_undefined(self):
        class Foo(pydantic.BaseModel):
            bar: int

        conv = ParsedConv(struct=Foo)
        delta_store = {}
        
        result = conv.delta("not-json", delta_store, streamed=False, is_last=True)
        
        # Should return UNDEFINED instead of raising
        assert result is utils.UNDEFINED

    def test_not_last_returns_undefined(self):
        class Foo(pydantic.BaseModel):
            bar: int

        conv = ParsedConv(struct=Foo)
        delta_store = {}
        
        result = conv.delta('{"bar": 3}', delta_store, streamed=True, is_last=False)
        
        assert result is utils.UNDEFINED


class TestToolExecConv:
    """Validate deferred execution helper with unified format."""

    def test_single_call_executed(self):
        # create a dummy ToolCall that returns value 7
        class MockTool:
            def __call__(self):
                return 7
                
        tc = MockTool()
        
        conv = ToolExecConv()
        result = conv.delta(tc, {}, False, True)
        assert result == 7

    def test_list_executes(self):
        class MockTool1:
            def __call__(self):
                return 1
                
        class MockTool2:
            def __call__(self):
                return 2
                
        tc1 = MockTool1()
        tc2 = MockTool2()
        conv = ToolExecConv()
        assert conv.delta([tc1, tc2], {}, False, True) == [1, 2]

    def test_undefined_pass_through(self):
        conv = ToolExecConv()
        assert conv.delta(utils.UNDEFINED, {}, False, True) is utils.UNDEFINED

    def test_none_pass_through(self):
        conv = ToolExecConv()
        assert conv.delta(None, {}, False, True) is utils.UNDEFINED


class TestToolConv:
    """Test unified ToolConv for processing tool calls."""

    def make_tool_def(self, name="dummy", description="desc"):
        """Factory to create a minimal ToolDef."""
        class Args(pydantic.BaseModel):
            x: int

        def dummy_fn(x: int) -> str:
            return f"Result: {x}"

        return ToolDef(
            name=name, description=description, fn=dummy_fn, input_model=Args
        )

    def test_prep(self):
        tools = [self.make_tool_def("t")]
        conv = ToolConv(tools=tools)
        prep = conv.prep()
        assert "tools" in prep and prep["tools"][0]["function"]["name"] == "t"

    def test_non_stream(self):
        tools = [self.make_tool_def()]
        conv = ToolConv(tools=tools, run_call=False)
        delta_store = {}
        
        # Create mock tool call response
        func = SimpleNamespace(name=tools[0].name, arguments=json.dumps({"x": 1}))
        tc = SimpleNamespace(id="id", function=func)
        resp = SimpleNamespace(tool_calls=[tc])
        
        out = conv.delta(resp, delta_store, is_streamed=False)
        assert isinstance(out[0], ToolCall)
        assert out[0].inputs.x == 1

    def test_stream_returns_undefined_when_no_tools(self):
        tools = [self.make_tool_def()]
        conv = ToolConv(tools=tools, run_call=False)
        ds = {}
        
        # Mock response without tool calls
        resp = SimpleNamespace(tool_calls=None)
        out = conv.delta(resp, ds, is_streamed=True, is_last=False)
        assert out == []


class TestStructStreamConv:
    """Test unified StructStreamConv for streaming structured data."""

    def test_prep(self):
        conv = StructStreamConv()
        prep = conv.prep()
        assert prep["response_format"] == "json_object"
        assert prep["stream"] is True

    def test_delta_chunk(self):
        conv = StructStreamConv()
        delta_store = {}
        
        # Mock streaming chunk
        chunk = SimpleNamespace(type="content.delta", parsed={"key": "value"})
        result = conv.delta(chunk, delta_store, is_streamed=True, is_last=False)
        
        assert result == {"key": "value"}

    def test_delta_done(self):
        conv = StructStreamConv()
        delta_store = {}
        
        # Mock done signal
        done = SimpleNamespace(type="content.done")
        result = conv.delta(done, delta_store, is_streamed=True, is_last=True)
        
        assert result is None

    def test_delta_error(self):
        conv = StructStreamConv()
        delta_store = {}
        
        # Mock error
        error = SimpleNamespace(type="error", error="Something went wrong")
        
        with pytest.raises(RuntimeError, match="Something went wrong"):
            conv.delta(error, delta_store, is_streamed=True, is_last=False)

    def test_non_streamed_returns_undefined(self):
        conv = StructStreamConv()
        delta_store = {}
        
        result = conv.delta("some text", delta_store, is_streamed=False, is_last=True)
        
        assert result is utils.UNDEFINED
