from typing import Any, Iterator, Tuple
import pytest
from dachi._core import _process as p


class Append(p.Module):

    def __init__(self, append: str):
        super().__init__()
        self._append = append

    def forward(self, name: str='') -> Any:
        return name + self._append
    

class WriteOut(p.StreamableModule):

    def stream_iter(self, x: str) -> Iterator[Tuple[Any, Any]]:
        
        out = ''
        for c in x:
            out = out + c
            yield out, c


class TestModule:

    def test_module_forward_outputs_correct_value(self):

        append = Append('_t')
        assert append.forward('x') == 'x_t'

    def test_module_outputs_t_with_correct_value(self):

        append = Append('_t')
        t = append.link('x')
        assert t.val == 'x_t'
        assert t.src.mod is append

    def test_chaining_appends_produces_correct_value(self):

        append = Append('_t')
        t = append.link('x')
        t = append.link(t)
        assert t.val == 'x_t_t'

    def test_it_is_undefined_if_val_not_defined(self):

        append = Append('_t')
        t = p.T()
        t = append.link(t)
        t = append.link(t)
        assert t.is_undefined() is True

    def test_it_probes_the_input(self):

        append = Append('_t')
        t1 = p.T()
        t = append.link(t1)
        t = append.link(t)
        res = t.probe(by={t1: 'x'})
        assert res == 'x_t_t'

    def test_t_probes_UNDEFINED_if_not_defined(self):

        append = Append('_t')
        t1 = p.T()
        t = append.link(t1)
        t = append.link(t)
        with pytest.raises(RuntimeError):
            t.probe({})


class TestStreamable:

    def test_streamable_streams_characters(self):

        writer = WriteOut()
        streamer = writer.forward('xyz')
        partial = streamer()
        assert partial.cur == 'x'
        assert partial.dx == 'x'

    def test_streamable_streams_characters_to_end(self):

        writer = WriteOut()
        streamer = writer.forward('xyz')
        partial = streamer()
        partial = streamer()
        partial = streamer()
        assert partial.cur == 'xyz'
        assert partial.dx == 'z'

    def test_call_returns_t_with_streamer(self):

        writer = WriteOut()
        t = writer.link(p.T('xyz'))
        partial = t.val()
        assert partial.cur == 'x'
        assert partial.dx == 'x'

    def test_call_returns_undefined_if_t_undefined(self):

        writer = WriteOut()
        t = writer.link(p.T())
        assert t.is_undefined()

#     # TODO* make this return partial
    def test_chained_after_stream_appends(self):

        writer = WriteOut()
        append = Append('_t')
        t = writer.link(p.T('xyz'))
        t = append.link(t)
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
        t = p.wait(
            t
        )

        assert t.val is p.WAITING

    def test_wait(self):

        append = Append('_t')
        t = append.link(p.T('xyz'))
        t = p.wait(
            t
        )

        assert t.val == 'xyz_t'


class MyProcess:

    @p.processf
    def process_test_method(self, x, y):
        return x + y

@p.processf
def process_test_func(x, y):
    return x + y


class TestProcessDecorator:

    def test_process_decorator_with_method(self):

        process = MyProcess()
        result = process.process_test_method(2, 3)
        assert result == 5

    def test_process_decorator_with_function(self):

        result = process_test_func(2, 3)
        assert result == 5

    def test_process_decorator_with_function_after_two(self):

        result = process_test_func(2, 3)
        result = process_test_func(2, 3)
        assert result == 5

    def test_process_decorator_with_method_after_two(self):

        process = MyProcess()
        result = process.process_test_method(2, 3)
        result = process.process_test_method(2, 3)
        assert result == 5

    def test_process_decorator_with_method_link(self):

        process = MyProcess()
        t1 = p.T(val=2)
        result = process.process_test_method.link(t1, 3)
        assert result.val == 5

    def test_process_decorator_with_function_after_two(self):

        t1 = p.T(val=2)
        result = process_test_func.link(t1, 3)
        result = process_test_func.link(t1, 3)
        assert result.val == 5
