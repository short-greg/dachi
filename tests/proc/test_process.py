import asyncio
import typing
from typing import Any
from dachi.proc import _process as core
from dachi.proc import Param, Module
from dachi.proc import _process
import numpy as np
# TODO: remove
from dachi.msg import Cue
import pytest


class Append(
    core.Module, core.AsyncModule, 
    core.StreamModule, core.AsyncStreamModule
):

    def __init__(self, append: str):
        super().__init__()
        self._append = append

    def forward(self, name: str='') -> Any:
        return name + self._append
    
    async def aforward(self, name: str=''):
        return self.forward(name)

    def stream(self, name: str=''):
        yield self.forward(name)
    
    async def astream(self, name: str=''):
        for v in self.stream(name):
            yield v


class Append2(
    core.Module, core.AsyncModule, 
    core.StreamModule, core.AsyncStreamModule
):

    def __init__(self, append: str):
        super().__init__()
        self._append = append

    def forward(self, val1: str, val2: str) -> Any:
        return val1 + val2 + self._append

    async def aforward(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def stream(self, *args, **kwargs):
        yield self.forward(*args, **kwargs)
    
    async def astream(self, *args, **kwargs):
        for v in self.stream(*args, **kwargs):
            yield v


class RefinerAppender(
    core.Module, core.AsyncModule, 
    core.StreamModule, core.AsyncStreamModule
):

    def __init__(self, append: str):
        super().__init__()
        self._append = append

    def forward(self, cur: str, val: str) -> Any:
        if cur is None:
            return val + self._append
        return cur + val + self._append

    async def aforward(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def stream(self, *args, **kwargs):
        yield self.forward(*args, **kwargs)
    
    async def astream(self, *args, **kwargs):
        for v in self.stream(*args, **kwargs):
            yield v


class WriteOut(core.Module):

    def __init__(self, append: str):
        super().__init__()
        self._append = append

    def forward(self, val: str) -> Any:
        return val + self._append

    def stream(self, val: str) -> Any:
        for v in val:
            yield v
        for v in self._append:
            yield v

    async def aforward(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    async def astream(self, *args, **kwargs):
        for v in self.stream(*args, **kwargs):
            yield v


class WaitAppend(core.Module):

    def __init__(self, append: str):
        super().__init__()
        self._append = append

    def forward(self, name: str='') -> Any:
        return name + self._append

    def stream(self, *args, **kwargs):
        yield self.forward(*args, **kwargs)

    async def aforward(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    async def astream(self, *args, **kwargs):
        for v in self.stream(*args, **kwargs):
            yield v

def _s(x):
    cur = ''
    for x_i in x:
        yield x_i


class TestStreamer:

    def test_streamable_streams_characters(self):

        streamer = _process.Streamer(_s('xyz'))
        partial = streamer()
        assert partial.dx == 'x'

    def test_streamable_streams_characters_to_end(self):

        streamer = _process.Streamer(_s('xyz'))
        partial = streamer()
        partial = streamer()
        partial = streamer()
        assert partial.dx == 'z'

    def test_streamer_gets_next_item(self):

        streamer = _process.Streamer(
            iter([(1, 0), (2, 2), (3, 2)])
        )
        partial = streamer()
        assert partial.dx == (1, 0)
        assert partial.complete is False

    def test_streamer_gets_final_item(self):

        streamer = _process.Streamer(
            iter([(1, 0), (2, 2), (3, 2)])
        )
        partial = streamer()
        partial = streamer()
        partial = streamer()
        partial = streamer()
        assert partial.dx == (3, 2)
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


class TestForwardStream:

    def test_forward_executes_module(self):

        append = Append('s')
        res = _process.forward(append, 't')
        assert res == 'ts'

    @pytest.mark.asyncio
    async def test_aforward_executes_module(self):

        append = Append('s')
        res = await _process.aforward(append, 't')
        assert res == 'ts'

    def test_stream_executes_module(self):

        append = Append('s')
        for r in _process.stream(append, 't'):
            pass
        assert r == 'ts'


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
            b, _process.B('xy'), init_mod=r
        )
        assert res == 'xxyy'

    def test_reduce_reduces_with_three_values(self):

        r = Append('x')
        b = RefinerAppender('y')
        res = _process.reduce(
            b, _process.B('xyz'), init_mod=r
        )
        assert res == 'xxyyzy'

    def test_reduce_reduces_without_init(self):

        b = RefinerAppender('y')
        res = _process.reduce(
            b, _process.B('xy')
        )
        assert res == 'xyyy'


class TestMulti:
    
    def test_that_multi_loops_over_the_modules(self):

        module = _process.MultiParallel(
            [Append('x'), Append('y')]
        )
        res = module('hi')
        assert res[0] == 'hix'
        assert res[1] == 'hiy'

    def test_that_multi_loops_over_the_modules_with_a_batch(self):

        x = _process.B(['hi', 'bye'])
        module = _process.MultiParallel(
            Append('x')
        )
        res = module(x)
        assert res[0] == 'hix'
        assert res[1] == 'byex'


class TestAsync:
    
    def test_that_async_loops_over_the_modules(self):

        module = _process.AsyncParallel(
            [Append('x'), Append('y')]
        )
        res = module('hi')
        assert res[0] == 'hix'
        assert res[1] == 'hiy'

    def test_that_async_loops_over_the_modules_with_a_batch(self):

        x = _process.B(['hi', 'bye'])
        module = _process.AsyncParallel(
            Append('x')
        )
        res = module(x)
        assert res[0] == 'hix'
        assert res[1] == 'byex'

    def test_that_async_loops_over_the_modules_with_a_batch_with_two(self):

        x = _process.B(['hi', 'bye'])
        module = _process.AsyncParallel(
            Append2('x')
        )
        res = module(x, 'z')
        assert res[0] == 'hizx'
        assert res[1] == 'byezx'


class NestedModule(Module):

    def __init__(self, child: Module):
        super().__init__()
        self.child = child
        self.p = Param(
            name='p',
            data=Cue(text='Do this')
        )

    def forward(self) -> Any:
        return None


class TestParallelLoop:

    def test_parallel_loop_with_two_modules(self):

        modules = _process.ModuleList([Append('s'), Append('t')])
        ress = []
        for m, r, kwargs in _process.parallel_loop(
            modules, 'r'
        ):
            ress.append(m(*r, **kwargs))
        assert ress[0] == 'rs'
        assert ress[1] == 'rt'

    def test_parallel_loop_with_two_inputs(self):

        # modules = _process.ModuleList([Append('s'), Append('t')])
        ress = []
        for m, r, kwargs in _process.parallel_loop(
            Append('s'), _process.B(['r', 'v'])
        ):
            ress.append(m(*r, **kwargs))
        assert ress[0] == 'rs'
        assert ress[1] == 'vs'

    def test_parallel_loop_with_two_inputs_and_two_modles(self):

        modules = _process.ModuleList([Append('s'), Append('t')])
        ress = []
        for m, r, kwargs in _process.parallel_loop(
            modules, _process.B(['r', 'v'])
        ):
            ress.append(m(*r, **kwargs))
        assert ress[0] == 'rs'
        assert ress[1] == 'vt'


class TestModule:

    def test_module_forward_outputs_correct_value(self):

        append = Append('_t')
        assert append.forward('x') == 'x_t'

    def test_async_runs_the_model_asynchronously(self):
        
        module = Append('t')

        async def run_model(data: typing.List[str]):

            tasks = []
            async with asyncio.TaskGroup() as tg:
                for data_i in data:
                    tasks.append(
                        tg.create_task(module.aforward(data_i))
                    )

            return list(task.result() for task in tasks)

        with asyncio.Runner() as runner:
            results = runner.run(run_model(['hi', 'bye']))
        
        assert results[0] == 'hit'
        assert results[1] == 'byet'

    def test_stream_forward_returns_the_results(self):
        
        module = Append('t')

        res = ''
        for dx in module.stream('xyz'):
            res += dx
        assert res == 'xyzt'

    def test_children_returns_no_children(self):
        
        module = Append('t')

        children = list(module.children())
        assert len(children) == 0

    def test_children_returns_two_children_with_nested(self):
        
        module = NestedModule(NestedModule(Append('a')))

        children = list(module.children())
        assert len(children) == 2

    def test_parameters_returns_all_parameters(self):
        
        module = NestedModule(NestedModule(Append('a')))

        children = list(module.parameters())
        assert len(children) == 2

    def test_streamable_streams_characters(self):

        writer = WriteOut('')
        results = []
        for dx in writer.stream('xyz'):
            results.append(dx)
        assert results == list('xyz')

