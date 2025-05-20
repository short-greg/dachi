import pytest
from dachi.proc import _process as p
from dachi.utils import WAITING, UNDEFINED
from dachi.proc import _graph as g
from .test_process import (
    Append, WriteOut # process_test_func, # 
)


class MyProcess:

    def forward(self, x, y):
        return x + y


class TestT:

    def test_src_returns_src(self):
        p = MyProcess()
        src = g.ModSrc(p, (0, 1))
        t = g.T(src=src)
        assert t.src == src

    def test_g_returns_src(self):
        p = MyProcess()
        src = g.ModSrc(p, (0, 1))
        t = g.T(src=src)
        assert t.val == UNDEFINED

    def test_label_updates_the_label(self):
        p = MyProcess()
        src = g.ModSrc(p, (0, 1))
        t = g.T(src=src)
        t = t.label(name='x')
        assert t.name == 'x'

    def test_is_undefined_returns_true(self):
        p = MyProcess()
        src = g.ModSrc(p, (0, 1))
        t = g.T(src=src)
        t = t.label(name='x')
        assert t.is_undefined()

    def test_get_item_returns_first_item(self):
        p = MyProcess()
        src = g.ModSrc(p, (0, 1))
        t = g.T([1, 2], src=src)
        t = t.label(name='x')
        assert t[0].val == 1

    def test_get_item_returns_first_index_source(self):
        p = MyProcess()
        src = g.ModSrc(p, (0, 1))
        t = g.T(UNDEFINED, src=src)
        t = t.label(name='x')
        assert t.val is UNDEFINED

    def test_detach_has_no_source(self):
        p = MyProcess()
        src = g.ModSrc(p, (0, 1))
        t = g.T([1, 2], src=src)
        t = t.detach()
        assert t.src is None
    
    def test_async_module_works_with_async(self):
        module = p.AsyncParallel(
            Append('x')
        )
        src = g.ModSrc.create(module, p.B(['hi']))
        res = src()
        assert res == ['hix']
    
    def test_async_module_works_with_multiple_async(self):
        module = p.AsyncParallel([
            Append('x'),
            Append('y')
        ])
        src = g.ModSrc.create(module, p.B(['hi', 'hi']))
        res = src()
        assert res == ['hix', 'hiy']

    def test_t_has_correct_annotation(self):
        p = MyProcess()
        src = g.ModSrc(p, (0, 1))
        annotation = 'A simple module.'
        t = g.T(src=src, annotation=annotation)
        assert t.annotation == annotation


class TestVar:

    def test_var_returns_value(self):
        var = g.Var(1)
        assert var() == 1

    def test_var_returns_value_with_factory(self):
        var = g.Var(default_factory=lambda: 3)
        assert var() == 3

    def test_var_has_no_incoming(self):
        var = g.Var(1)
        assert len(list(var.incoming())) == 0

    def test_var_raises_error_if_no_default_or_factory(self):
        with pytest.raises(RuntimeError, match="Either the default value or default factory must be defined"):
            g.Var()

    def test_var_with_default_overrides_factory(self):
        var = g.Var(default=5, default_factory=lambda: 10)
        assert var() == 5

    def test_var_with_none_default_and_factory(self):
        var = g.Var(default=None, default_factory=lambda: "fallback")
        assert var() == None  # Explicitly testing None as a valid default

    def test_var_with_complex_factory(self):
        var = g.Var(default_factory=lambda: [i for i in range(5)])
        assert var() == [0, 1, 2, 3, 4]

    def test_var_with_callable_default(self):
        var = g.Var(default_factory=lambda: "callable_default")
        assert callable(var.default_factory)
        assert var() == "callable_default"

    def test_var_with_mutable_default(self):
        default_list = [1, 2, 3]
        var = g.Var(default=default_list)
        assert var() == default_list
        default_list.append(4)
        assert var() == [1, 2, 3, 4]  # Ensure mutable default is reflected

    def test_var_with_factory_returning_mutable(self):
        var = g.Var(default_factory=lambda: {"key": "value"})
        result = var()
        assert result == {"key": "value"}
        result["key"] = "new_value"
        assert var() == {"key": "value"}  # Factory should return a new instance each time

    def test_var_forward_with_empty_by(self):
        var = g.Var(default=42)
        assert var.forward(by={}) == 42

    def test_var_forward_ignores_by(self):
        var = g.Var(default=42)
        assert var.forward(by={"irrelevant": "data"}) == 42

    def test_var_with_large_default_value(self):
        large_value = "x" * 10**6
        var = g.Var(default=large_value)
        assert var() == large_value

    def test_var_with_nested_factory(self):
        var = g.Var(default_factory=lambda: {"nested": [1, 2, {"key": "value"}]})
        assert var() == {"nested": [1, 2, {"key": "value"}]}

    def test_var_with_default_as_falsey_value(self):
        var = g.Var(default=0)
        assert var() == 0
        var = g.Var(default="")
        assert var() == ""
        var = g.Var(default=False)
        assert var() == False

    def test_var_returns_value(self):

        var = g.Var(1)
        assert var() == 1

    def test_var_returns_value_with_factory(self):

        var = g.Var(default_factory=lambda: 3)
        assert var() == 3

    def test_var_has_no_incoming(self):

        var = g.Var(1)
        assert len(list(var.incoming())) == 0


class TestIdxSrc:

    def test_idx_returns_value(self):
        src = g.Var([0, 1])
        t = g.T(src=src)
        idx = g.IdxSrc(t, 0)
        
        assert idx() == 0

    def test_idx_incoming_returns_t(self):
        src = g.Var([0, 1])
        t = g.T(src=src)
        idx = g.IdxSrc(t, 0)
        
        incoming = list(idx.incoming())[0]
        assert incoming is t

    def test_probe_incoming_returns_val(self):
        src = g.Var([0, 1])
        idx = g.T(src=src)[0]
        val = idx.probe()
        assert val == 0



class TestWaitSrc:

    def test_returns_waiting(self):
        writer = WriteOut('hi')
        t = g.stream_link(writer, g.T('bye'))
        src = g.WaitSrc(t)
        assert src() == WAITING

    def test_returns_value_if_finished(self):
        writer = WriteOut('hi')
        t = g.stream_link(writer, g.T(''))
        src = g.WaitSrc(t, lambda x: ''.join(x))
        src()
        src()
        assert src() == 'hi'

    def test_returns_waiting_for_partial_incomplete(self):
        writer = WriteOut('hi')
        t = g.stream_link(writer, g.T('xyz'))
        partial = t.val()
        src = g.WaitSrc(t)
        assert src() == WAITING

    def test_aggregates_partial_when_complete(self):
        writer = WriteOut('hi')
        t = g.stream_link(writer, g.T('xyz'))
        # partial = t.val()
        # partial.complete = True
        # partial.full = 'xyz_full'
        src = g.WaitSrc(t, lambda x: ''.join(
            xi.upper() for xi in x
        ))
        src()
        src()
        src()
        src()
        src()
        res = src()
    
        assert res == 'XYZHI'

    # def test_returns_waiting_for_streamer_incomplete(self):
    #     k = g.Var('k')
    #     src = g.StreamSrc(Append('s'), g.TArgs(k))
    #     streamer = src()
    #     wait_src = g.WaitSrc(streamer)
    #     assert wait_src() == WAITING

    # def test_aggregates_streamer_when_complete(self):
    #     k = g.Var('k')
    #     src = g.StreamSrc(Append('s'), g.TArgs(k))
    #     streamer = src()
    #     # streamer.complete = True
    #     # streamer.output = g.Partial(full='ks_full')
    #     wait_src = g.WaitSrc(streamer, lambda x: x[::-1])
    #     assert wait_src() == 'lluf_sk'

    def test_handles_non_partial_non_streamer_values(self):
        t = g.T('static_value')
        src = g.WaitSrc(t)
        assert src() == 'static_value'

    def test_raises_error_for_invalid_incoming_type(self):
        with pytest.raises(AttributeError):
            invalid_incoming = 123  # Not a valid transmission
            src = g.WaitSrc(invalid_incoming)
            src()

    def test_handles_empty_aggregation_function(self):
        writer = WriteOut('hi')
        t = g.stream_link(writer, g.T('xyz'))
        # partial = t.val()
        # partial.complete = True
        # partial.full = 'xyz_full'
        src = g.WaitSrc(t, lambda x: ''.join(x))
        src()
        src()
        src()
        src()
        src()
        assert src() == 'xyzhi'

    def test_handles_none_as_incoming(self):
        with pytest.raises(AttributeError):
            src = g.WaitSrc(None)
            src()

    def test_forward_with_empty_by(self):
        writer = WriteOut('hi')
        t = g.stream_link(writer, g.T('xyz'))
        src = g.WaitSrc(t)
        assert src.forward(by={}) == WAITING

    def test_forward_with_valid_by(self):
        writer = WriteOut('hi')
        t = g.stream_link(writer, g.T())
        src = g.WaitSrc(t)
        assert src.forward(by={t: 'xyz'}) == "xyz"

    def test_returns_waiting(self):

        writer = WriteOut('hi')
        t = g.stream_link(writer, g.T('bye'))
        src = g.WaitSrc(t)
        assert src() == WAITING

    def test_returns_value_if_finished(self):

        writer = WriteOut('hi')
        t = g.stream_link(writer, g.T(''))
        src = g.WaitSrc(t, lambda x: ''.join(x))
        src()
        src()
        assert src() == 'hi'


class TestLink:

    def test_module_outputs_t_with_correct_value(self):

        append = Append('_t')
        t = g.link(append, ('x'))
        assert t.val == 'x_t'
        assert t.src.mod is append

    def test_chaining_appends_produces_correct_value(self):

        append = Append('_t')
        t = g.link(append, 'x')
        t = g.link(append, t)
        assert t.val == 'x_t_t'

    def test_it_is_undefined_if_val_not_defined(self):

        append = Append('_t')
        t = g.T()
        t = g.link(append, t)
        t = g.link(append, t)
        assert t.is_undefined() is True

    def test_it_probes_the_input(self):

        append = Append('_t')
        t1 = g.T()
        t = g.link(append, t1)
        t = g.link(append, t)
        res = t.probe(by={t1: 'x'})
        assert res == 'x_t_t'

    def test_t_probes_UNDEFINED_if_not_defined(self):

        append = Append('_t')
        t1 = g.T()
        t = g.link(append, t1)
        t = g.link(append, t)
        with pytest.raises(RuntimeError):
            t.probe({})


class TestStreamSrc:

    def test_stream_src_returns_streamer_instance(self):
        """Test that StreamSrc returns a Streamer instance."""
        k = g.Var('k')
        src = g.StreamSrc(Append('s'), g.NodeArgs(k))
        val = src()
        assert isinstance(val, g.Streamer)

    def test_stream_src_with_empty_args(self):
        """Test StreamSrc with empty arguments."""
        k = g.Var([])
        src = g.StreamSrc(Append('s'), g.NodeArgs(k))
        val = src()
        assert isinstance(val, g.Streamer)

    def test_stream_src_with_invalid_module(self):
        """Test StreamSrc with an invalid module."""
        k = g.Var('k')
        with pytest.raises(AttributeError):
            src = g.StreamSrc(None, g.NodeArgs(k))
            src()

    def test_stream_src_with_none_args(self):
        """Test StreamSrc with None as arguments."""
        with pytest.raises(TypeError):
            src = g.StreamSrc(Append('s'), None)
            src()

    def test_stream_src_with_multiple_args(self):
        """Test StreamSrc with multiple arguments."""
        k1 = g.Var('k1')
        src = g.StreamSrc(Append('s'), g.NodeArgs(k1))
        val = src()
        assert isinstance(val, g.Streamer)

    def test_stream_src_forward_with_empty_by(self):
        """Test forward method with an empty 'by' dictionary."""
        k = g.Var('k')
        src = g.StreamSrc(Append('s'), g.NodeArgs(k))
        streamer = src.forward(by={})
        assert isinstance(streamer, g.Streamer)

    def test_stream_src_forward_with_valid_by(self):
        """Test forward method with a valid 'by' dictionary."""
        k = g.Var('k')
        src = g.StreamSrc(Append('s'), g.NodeArgs(k))
        streamer = src.forward(by={k: 'test'})
        assert isinstance(streamer, g.Streamer)

    def test_stream_src_forward_with_invalid_by(self):
        """Test forward method with an invalid 'by' dictionary."""
        k = g.Var('k')
        src = g.StreamSrc(Append('s'), g.NodeArgs(k))
        with pytest.raises(TypeError):
            src.forward(by="invalid")

    def test_stream_src_call_with_empty_by(self):
        """Test __call__ method with an empty 'by' dictionary."""
        k = g.Var('k')
        src = g.StreamSrc(Append('s'), g.NodeArgs(k))
        streamer = src(by={})
        assert isinstance(streamer, g.Streamer)

    def test_stream_src_call_with_valid_by(self):
        """Test __call__ method with a valid 'by' dictionary."""
        k = g.Var('k')
        src = g.StreamSrc(Append('s'), g.NodeArgs(k))
        streamer = src(by={k: 'test'})
        assert isinstance(streamer, g.Streamer)

    def test_stream_src_incoming_yields_correct_values(self):
        """Test that incoming method yields correct values."""
        k = g.Var('k')
        src = g.StreamSrc(Append('s'), g.NodeArgs(k))
        incoming = list(src.incoming())
        assert len(incoming) == 1
        assert incoming[0] is k

    def test_stream_src_with_large_input(self):
        """Test StreamSrc with a large input."""
        large_input = g.Var('x' * 10**6)
        src = g.StreamSrc(Append('s'), g.NodeArgs(large_input))
        val = src()
        assert isinstance(val, g.Streamer)

    def test_stream_src_with_callable_args(self):
        """Test StreamSrc with callable arguments."""
        k = g.Var(lambda: 'dynamic_value')
        src = g.StreamSrc(Append('s'), g.NodeArgs(k))
        val = src()
        assert isinstance(val, g.Streamer)

    def test_stream_src_with_mutable_args(self):
        """Test StreamSrc with mutable arguments."""
        mutable_arg = g.Var([1, 2, 3])
        src = g.StreamSrc(Append('s'), g.NodeArgs(mutable_arg))
        val = src()
        assert isinstance(val, g.Streamer)
        mutable_arg().append(4)
        assert mutable_arg() == [1, 2, 3, 4]

    def test_stream_src_with_no_module_stream_method(self):
        """Test StreamSrc with a module that lacks a 'stream' method."""
        class InvalidModule:
            pass

        k = g.Var('k')
        with pytest.raises(AttributeError):
            src = g.StreamSrc(InvalidModule(), g.NodeArgs(k))
            src()

    def test_stream_src_returns_a_streamer(self):
        k = g.Var('k')
        src = g.StreamSrc(
            Append('s'), g.NodeArgs(k)
        )
        val = src()
        
        assert isinstance(val, g.Streamer)

    def test_streamer_returns_value(self):
        k = g.Var('k')
        src = g.StreamSrc(
            Append('s'), g.NodeArgs(k)
        )
        val = src()
        res = val()
        
        assert res.dx == 'ks'


class TestStreamLink:

    def test_call_returns_t_with_streamer(self):

        writer = WriteOut('hi')
        t = g.stream_link(writer, g.T('xyz'))
        partial = t.val()
        assert partial.dx == 'x'

    def test_call_returns_undefined_if_t_undefined(self):

        writer = WriteOut('hi')
        t = g.stream_link(writer, g.T())
        assert t.is_undefined()

# # #     # TODO* make this return partial
    def test_chained_after_stream_appends(self):

        writer = WriteOut('hi')
        append = Append('_t')
        t = g.stream_link(writer, g.T('xyz'))
        t = g.link(append, t)
        assert t.val.dx == 'x_t'

    def test_stream_completes_the_stream(self):
        writer = WriteOut('hi')
        append = Append('_t')

        for t in g.stream(writer, g.T('xyz')):
            t = g.link(append, t)
        
        assert t.val.dx == 'i_t'


class TestWait:

    def test_wait_results_in_waiting(self):

        writer = WriteOut('hi')
        t = g.stream_link(writer, g.T('xyz'))
        t = g.wait(t)

        assert t.val is WAITING

    def test_wait(self):

        append = Append('_t')
        t = g.link(append, g.T('xyz'))
        t = g.wait(t)

        assert t.val == 'xyz_t'


class TestParallelLink:

    def test_module_outputs_t_with_correct_value(self):

        append = Append('_t')
        t = g.link(append, ('x'))
        assert t.val == 'x_t'
        assert t.src.mod is append

    def test_chaining_appends_produces_correct_value(self):

        append = Append('_t')
        t = g.link(append, 'x')
        t = g.link(append, t)
        assert t.val == 'x_t_t'
