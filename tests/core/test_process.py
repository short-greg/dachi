from typing import Any, Iterator, Tuple
import pytest
from dachi._core import _process as p
from dachi._core._core import Module
from .test_core import WriteOut

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


class TestStreamer:

    def test_streamable_streams_characters(self):

        writer = WriteOut()
        streamer = writer.streamer('xyz')
        partial = streamer()
        assert partial.cur == 'x'
        assert partial.dx == 'x'

    def test_streamable_streams_characters_to_end(self):

        writer = WriteOut()
        streamer = writer.streamer('xyz')
        partial = streamer()
        partial = streamer()
        partial = streamer()
        assert partial.cur == 'xyz'
        assert partial.dx == 'z'

    def test_streamer_gets_next_item(self):

        streamer = _core.Streamer(
            iter([(1, 0), (2, 2), (3, 2)])
        )
        partial = streamer()
        assert partial.cur == 1
        assert partial.complete is False

    def test_streamer_gets_final_item(self):

        streamer = _core.Streamer(
            iter([(1, 0), (2, 2), (3, 2)])
        )
        partial = streamer()
        partial = streamer()
        partial = streamer()
        partial = streamer()
        assert partial.cur == 3
        assert partial.complete is True


class TestGet:
    pass


class TestSet:
    pass


class TestMulti:
    pass




