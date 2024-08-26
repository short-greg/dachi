from typing import Any, Iterator, Tuple
import pytest
from dachi._core import _process as p
from dachi._core._core import Module
from .test_core import WriteOut
from typing import Any
from dachi._core import Args
from dachi._core import _process as core
from dachi._core import _process
from .test_core import SimpleStruct        
import numpy as np


class Append(core.Module):

    def __init__(self, append: str):
        super().__init__()
        self._append = append

    def forward(self, name: str='') -> Any:
        return name + self._append


class Append2(core.Module):

    def __init__(self, append: str):
        super().__init__()
        self._append = append

    def forward(self, val1: str, val2: str) -> Any:
        return val1 + val2 + self._append


# use for "reducing"
class RefinerAppender(core.Module):

    def __init__(self, append: str):
        super().__init__()
        self._append = append

    def forward(self, cur: str, val: str) -> Any:
        if cur is None:
            return val + self._append
        return cur + val + self._append


class WaitAppend(core.Module):

    def __init__(self, append: str):
        super().__init__()
        self._append = append

    def forward(self, name: str='') -> Any:
        return name + self._append


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


def _s(x):
    cur = ''
    for x_i in x:
        cur += x_i
        yield cur, x_i


class TestStreamer:

    def test_streamable_streams_characters(self):

        streamer = _process.Streamer(_s('xyz'))
        partial = streamer()
        assert partial.cur == 'x'
        assert partial.dx == 'x'

    def test_streamable_streams_characters_to_end(self):

        streamer = _process.Streamer(_s('xyz'))
        partial = streamer()
        partial = streamer()
        partial = streamer()
        assert partial.cur == 'xyz'
        assert partial.dx == 'z'

    def test_streamer_gets_next_item(self):

        streamer = _process.Streamer(
            iter([(1, 0), (2, 2), (3, 2)])
        )
        partial = streamer()
        assert partial.cur == 1
        assert partial.complete is False

    def test_streamer_gets_final_item(self):

        streamer = _process.Streamer(
            iter([(1, 0), (2, 2), (3, 2)])
        )
        partial = streamer()
        partial = streamer()
        partial = streamer()
        partial = streamer()
        assert partial.cur == 3
        assert partial.complete is True


class TestSequential:

    def test_sequential_does_nothing_if_zero(self):

        sequential = _process.Sequential()
        res = sequential('x')
        assert res == 'x'

    def test_sequential_appends(self):

        sequential = _process.Sequential(Append('z'))
        res = sequential('x')
        assert res == 'xz'

    def test_sequential_appends2(self):

        sequential = _process.Sequential(Append('z'), Append('y'))
        res = sequential('x')
        assert res == 'xzy'

    def test_sequential_works_with_two_inputs(self):

        sequential = _process.Sequential(Append2('x'))
        res = sequential('x', 'y')
        assert res == 'xyx'

    def test_len_returns_correct_len(self):

        sequential = _process.Sequential(Append('z'), Append('y'))
        return len(sequential) == 2


class TestModuleList(object):

    def test_module_list_has1(self):

        module_list = _process.ModuleList([Append('z')])
        assert len(module_list) == 1

    def test_module_list_has2(self):

        module_list = _process.ModuleList([Append('z'), Append('z')])
        assert len(module_list) == 2

    def test_module_list_two_children(self):

        module_list = _process.ModuleList([Append('z'), Append('z')])
        assert len(list(module_list.children())) == 2


class TestBatched:

    def test_batched_len_is_correct(self):
        
        batched = _process.Batched([1,2,3,4,5,6], size=3)
        assert len(batched) == 2

    def test_loop_over_batches_returns_correct_values(self):
        
        batched = _process.Batched([1,2,3,4,5,6], size=3)
        batch_list = list(batched)
        assert batch_list[0] == [1, 2, 3]
        assert batch_list[1] == [4, 5, 6]

    def test_loop_over_batches_returns_correct_values_with_two(self):
        
        batched = _process.Batched([1,2,3,4,5,6], [0, 1,2,3,4,5], size=3)
        batch_list = list(batched)
        print(batch_list)
        assert batch_list[0][0] == [1, 2, 3]
        assert batch_list[0][1] == [0, 1, 2]

    def test_shuffle_changes_the_order(self):
        
        batched = _process.Batched([1,2,3,4,5,6], [0, 1,2,3,4,5], size=3)

        np.random.seed(0)
        batched = batched.shuffle()
        batch_list = list(batched)
        assert batch_list[0][0] != [1, 2, 3]


class TestReduce:

    def test_reduce_reduces_with_init(self):

        r = Append('x')
        b = RefinerAppender('y')
        res = _process.reduce(
            b, _process.P('xy'), init=r
        )
        assert res == 'xxyy'

    def test_reduce_reduces_with_three_values(self):

        r = Append('x')
        b = RefinerAppender('y')
        res = _process.reduce(
            b, _process.P('xyz'), init=r
        )
        assert res == 'xxyyzy'

    def test_reduce_reduces_without_init(self):

        b = RefinerAppender('y')
        res = _process.reduce(
            b, _process.P('xy')
        )
        assert res == 'xyyy'

    # def test_map_(self):
        
    #     batched = _process.Batched(['xyz', 'abc'], size=1)
    #     append = Append('x')
    #     res = batched.map(append)

    #     assert batch_list[0][0] == [1, 2, 3]
    #     assert batch_list[0][1] == [0, 1, 2]


class TestMulti:
    
    def test_that_multi_loops_over_the_modules(self):

        module = _process.MultiModule(
            [Append('x'), Append('y')]
        )
        res = module('hi')
        assert res[0] == 'hix'
        assert res[1] == 'hiy'

    def test_that_multi_loops_over_the_modules_with_a_batch(self):

        x = _process.P(['hi', 'bye'])
        module = _process.MultiModule(
            Append('x')
        )
        res = module(x)
        assert res[0] == 'hix'
        assert res[1] == 'byex'


class TestAsync:
    
    def test_that_async_loops_over_the_modules(self):

        module = _process.AsyncModule(
            [Append('x'), Append('y')]
        )
        res = module('hi')
        assert res[0] == 'hix'
        assert res[1] == 'hiy'

    def test_that_async_loops_over_the_modules_with_a_batch(self):

        x = _process.P(['hi', 'bye'])
        module = _process.AsyncModule(
            Append('x')
        )
        res = module(x)
        assert res[0] == 'hix'
        assert res[1] == 'byex'

    def test_that_async_loops_over_the_modules_with_a_batch_with_two(self):

        x = _process.P(['hi', 'bye'])
        module = _process.AsyncModule(
            Append2('x')
        )
        res = module(x, 'z')
        assert res[0] == 'hizx'
        assert res[1] == 'byezx'
