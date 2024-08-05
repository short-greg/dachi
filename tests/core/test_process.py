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
