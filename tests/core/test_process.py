from typing import Any, Iterator, Tuple
import pytest
from dachi._core import _process as p
from dachi._core._core import Module


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
