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

    def test_is_undefined_returns_true(self):
        p = MyProcess()
        src = g.ModSrc(p, (0, 1))
        t = g.T(1, src=src)
        t = t.label(name='x')
        assert not t.is_undefined()

    def test_get_item_returns_first_item(self):
        p = MyProcess()
        src = g.ModSrc(p, (0, 1))
        t = g.T([1, 2], src=src)
        t = t.label(name='x')
        assert t[0].val == 1

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


class TestIdxSrc:

    def test_var_returns_value(self):

        var = g.Var(1)
        assert var() == 1

    def test_var_returns_value_with_factory(self):

        var = g.Var(default_factory=lambda: 3)
        assert var() == 3

    def test_var_has_no_incoming(self):

        var = g.Var(1)
        assert len(list(var.incoming())) == 0


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


class TestStreamLink:

    def test_call_returns_t_with_streamer(self):

        writer = WriteOut('hi')
        t = g.stream_link(writer, g.T('xyz'))
        partial = t.val()
        assert partial.dx == 'x'

#     def test_call_returns_undefined_if_t_undefined(self):

#         writer = WriteOut('hi')
#         t = g.stream_link(writer, g.T())
#         assert t.is_undefined()

# # #     # TODO* make this return partial
#     def test_chained_after_stream_appends(self):

#         writer = WriteOut('hi')
#         append = Append('_t')
#         t = g.stream_link(writer, g.T('xyz'))
#         t = g.link(append, t)
#         assert t.val.cur == 'x_t'

#     def test_stream_completes_the_stream(self):

#         writer = WriteOut('hi')
#         append = Append('_t')

#         for t in g.stream(writer, g.T('xyz')):
#             t = g.link(append, t)
        
#         assert t.val.cur == 'xyzhi_t'


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
