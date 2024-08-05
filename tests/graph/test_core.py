from typing import Any, Iterator, Tuple
import pytest
from dachi._core import _process as p
from dachi.graph import _core as g
from ..core.test_process import Append, WriteOut, MyProcess, process_test_func


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
        t1 = p.T()
        t = g.link(append, t1)
        t = g.link(append, t)
        res = t.probe(by={t1: 'x'})
        assert res == 'x_t_t'

    def test_t_probes_UNDEFINED_if_not_defined(self):

        append = Append('_t')
        t1 = p.T()
        t = g.link(append, t1)
        t = g.link(append, t)
        with pytest.raises(RuntimeError):
            t.probe({})


class TestStreamLink:

    def test_call_returns_t_with_streamer(self):

        writer = WriteOut()
        t = g.link(writer, p.T('xyz'))
        partial = t.val()
        assert partial.cur == 'x'
        assert partial.dx == 'x'

    def test_call_returns_undefined_if_t_undefined(self):

        writer = WriteOut()
        t = g.link(writer, p.T())
        assert t.is_undefined()

#     # TODO* make this return partial
    def test_chained_after_stream_appends(self):

        writer = WriteOut()
        append = Append('_t')
        t = g.link(writer, p.T('xyz'))
        t = g.link(append, t)
        assert t.val.cur == 'x_t'

    def test_stream_completes_the_stream(self):

        writer = WriteOut()
        append = Append('_t')

        for t in p.stream(writer, p.T('xyz')):
            t = append.link(t)
        
        assert t.val.cur == 'xyz_t'


class TestWait:

    def test_wait_results_in_waiting(self):

        writer = WriteOut()
        t = writer.link(p.T('xyz'))
        t = p.wait(t)

        assert t.val is p.WAITING

    def test_wait(self):

        append = Append('_t')
        t = append.link(p.T('xyz'))
        t = p.wait(t)

        assert t.val == 'xyz_t'


class TestDecorator:

    def test_process_decorator_with_method_link(self):

        process = MyProcess()
        t1 = p.T(val=2)
        result = g.link(process.process_test_method, t1, 3)
        assert result.val == 5

    def test_process_decorator_with_function_after_two(self):

        t1 = p.T(val=2)
        result = g.link(process_test_func, t1, 3)
        result = g.link(process_test_func, t1, 3)
        assert result.val == 5

# TODO: Add more tests
