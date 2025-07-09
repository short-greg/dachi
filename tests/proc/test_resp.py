from dachi.core import Msg, Resp
from dachi.proc import _resp as _resp
from dachi import utils


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
