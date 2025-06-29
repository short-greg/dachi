"""Updated test‑suite for _process.py.

Each test class targets a single public surface.  Tests are small, black‑box, and
nit‑picky; each verifies *one* behaviour (except where a function returns
multiple values that must be asserted together).

Google‑style docstrings are used throughout.
"""

import asyncio
from typing import Any, Iterator, List, Tuple

import pytest

from dachi import process as P


async def _async_identity(x: int) -> int:  # noqa: D401
    return x


def _gen(x: int) -> Iterator[int]:  # noqa: D401
    yield x
    yield x + 1


class _EchoProcess(P.Process):
    def __init__(self):
        self.calls = []

    def forward(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return args[0] if args else kwargs.get("x")


class _AEchoProcess(P.AsyncProcess):
    def __init__(self):
        self.calls = []

    async def aforward(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return args[0] if args else kwargs.get("x")


class _SEchoProcess(P.StreamProcess):
    def __init__(self, n: int):
        self.n = n

    def stream(self, x):
        for _ in range(self.n):
            yield x


class _ASEchoProcess(P.AsyncStreamProcess):
    def __init__(self, n: int):
        self.n = n

    async def astream(self, x):
        for _ in range(self.n):
            yield x


class TestProcess:
    """`Process.__call__` forwards to `forward` with positional and keyword args."""

    def test_call_identity(self):
        proc = _EchoProcess()
        assert proc("abc") == "abc"

    def test_call_kwargs(self):
        proc = _EchoProcess()
        assert proc(x=42) == 42

    def test_forward_returns_none(self):
        class _N(P.Process):
            def forward(self, *_: Any, **__: Any) -> None:
                return None
        assert _N()() is None

    def test_base_instantiation_fails(self):
        with pytest.raises(TypeError):
            P.Process()  # type: ignore[abstract]

    def test_multiple_calls_recorded(self):
        proc = _EchoProcess()
        proc(1)
        proc(2, y=3)
        assert len(proc.calls) == 2
        assert proc.calls[0][0] == (1,)
        assert proc.calls[1][1] == {"y": 3}


@pytest.mark.asyncio
class TestAsyncProcess:
    """`AsyncProcess.aforward` handles args and awaits correctly."""

    async def test_async_identity(self):
        proc = _AEchoProcess()
        result = await proc.aforward("xyz")
        assert result == "xyz"

    async def test_multiple_async_calls(self):
        proc = _AEchoProcess()
        await proc.aforward(1)
        await proc.aforward(x=9)
        assert len(proc.calls) == 2
        assert proc.calls[0][0] == (1,)
        assert proc.calls[1][1] == {"x": 9}

    async def test_base_instantiation_fails(self):
        with pytest.raises(TypeError):
            P.AsyncProcess()  # type: ignore[abstract]


class TestStreamProcess:
    """`StreamProcess.stream` yields the correct number and values."""

    def test_stream_repetition(self):
        proc = _SEchoProcess(n=2)
        assert list(proc.stream("hi")) == ["hi", "hi"]

    def test_stream_zero_repeats(self):
        proc = _SEchoProcess(n=0)
        assert list(proc.stream("test")) == []


@pytest.mark.asyncio
class TestAsyncStreamProcess:
    """`AsyncStreamProcess.astream` yields correctly in async context."""

    async def test_astream_repetition(self):
        proc = _ASEchoProcess(n=2)
        out: List[str] = []
        async for item in proc.astream("hi"):
            out.append(item)
        assert out == ["hi", "hi"]

    async def test_astream_zero(self):
        proc = _ASEchoProcess(n=0)
        out: List[str] = []
        async for item in proc.astream("skip"):
            out.append(item)
        assert out == []

    async def test_base_instantiation_fails(self):
        with pytest.raises(TypeError):
            P.AsyncStreamProcess()  # type: ignore[abstract]



# class _EchoProcess(P.Process):
#     def forward(self, x):
#         return x

# class _AEchoProcess(P.AsyncProcess):
#     async def aforward(self, x):
#         return x

# class _SEchoProcess(P.StreamProcess):
#     def __init__(self, n: int):
#         self.n = n
#     def stream(self, x):
#         for _ in range(self.n):
#             yield x


class TestForwardHelper:
    """Branch coverage for :func:`_process.forward`."""

    def test_with_process(self):
        assert P.forward(_EchoProcess(), 1) == 1

    def test_with_plain_func(self):
        assert P.forward(lambda x: x * 2, 3) == 6

    def test_with_generator(self):
        assert P.forward(_gen, 1) == [1, 2]

    def test_async_func_raises(self):
        with pytest.raises(NotImplementedError):
            P.forward(_async_identity, 7)  # type: ignore[arg-type]

    def test_forward_with_zero_args(self):
        def unit():
            return 42
        assert P.forward(unit) == 42

    def test_forward_with_kwargs(self):
        def add(x, y):
            return x + y
        assert P.forward(add, x=1, y=2) == 3


@pytest.mark.asyncio
class TestAForwardHelper:
    """`aforward` handles both sync and async callables uniformly."""

    async def test_async_lambda(self):
        out = await P.aforward(lambda x: x + 1, 2)
        assert out == 3

    async def test_with_async_process(self):
        out = await P.aforward(_AEchoProcess(), 5)
        assert out == 5

    async def test_with_plain_sync_func(self):
        def double(x):
            return x * 2
        out = await P.aforward(double, 3)
        assert out == 6

    async def test_with_plain_async_func(self):
        async def add(x, y):
            return x + y
        out = await P.aforward(add, 2, 3)
        assert out == 5

    async def test_invalid_type_raises(self):
        with pytest.raises(RuntimeError):
            await P.aforward(object())  # type: ignore[arg-type]


class TestStreamHelper:
    """`stream` collapses scalars into single-item iterators and preserves generators."""

    def test_stream_with_proc(self):
        proc = _SEchoProcess(n=3)
        assert list(P.stream(proc, "a")) == ["a", "a", "a"]

    def test_stream_plain_func(self):
        out_iter = P.stream(lambda x: x * 2, 2)
        assert list(out_iter) == [4]

    def test_stream_plain_generator(self):
        def yield_one(x):
            yield x
        assert list(P.stream(yield_one, 9)) == [9]

    def test_async_gen_not_supported(self):
        async def _agen(x):
            yield x
        with pytest.raises(NotImplementedError):
            list(P.stream(_agen, 1))  # type: ignore[arg-type]


@pytest.mark.asyncio
class TestAStreamHelper:
    """`astream` correctly converts sync generators to async iterators."""

    async def test_sync_gen_promoted(self):
        gen = _gen(1)
        out: List[int] = []
        async for val in P.astream(gen):
            out.append(val)
        assert out == [1, 2]

    async def test_empty_sync_gen(self):
        def _empty():
            if False:
                yield 1
        out: List[int] = []
        async for val in P.astream(_empty):
            out.append(val)
        assert out == []

    async def test_async_gen_direct(self):
        async def agen():
            for i in range(2):
                yield i
        out: List[int] = []
        async for val in P.astream(agen):
            out.append(val)
        assert out == [0, 1]

    async def test_invalid_type_raises(self):
        with pytest.raises(RuntimeError):
            async for _ in P.astream(123):  # type: ignore[arg-type]
                pass



# ---------------------------------------------------------------------------
# 4.  Batch helpers `I` & `B`
# ---------------------------------------------------------------------------


class TestRecur:
    """Iterator `I` yields the same object *n* times."""

    def test_positive(self) -> None:
        obj = object()
        assert list(P.Recur(obj, n=3)) == [obj, obj, obj]

    def test_zero(self) -> None:
        assert list(P.Recur(1, n=0)) == []

    def test_negative(self) -> None:
        with pytest.raises(ValueError):
            list(P.Recur(1, n=-1))


class TestChunk:
    """`B` batches an iterable preserving order and supports `.m` helper."""

    def test_preserves_order(self) -> None:
        b = P.Chunk([1, 2, 3])
        assert list(b) == [1, 2, 3]
        assert b.n == 3

    def test_truncate(self) -> None:
        b = P.Chunk([1, 2, 3], n=2)
        assert list(b) == [1, 2]
        assert b.n == 2

    def test_classmethod_m(self) -> None:
        b1, b2 = P.Chunk.m([1, 2], [3, 4])
        assert list(b1) == [1, 2]
        assert list(b2) == [3, 4]
        assert b1.n == b2.n == 2

    def test_non_iterable(self) -> None:
        with pytest.raises(TypeError):
            P.Chunk(123)  # type: ignore[arg-type]



class TestSequential:
    """`Sequential` composes heterogeneous modules in order."""

    def test_three_stage_pipeline(self) -> None:
        seq = P.Sequential(
            [lambda x: x + 1, _EchoProcess(), lambda x: x * 10]
        )
        assert seq.forward(2) == 30  # (2+1) -> 3 -> 3*10

    def test_empty_sequential_identity(self) -> None:
        seq = P.Sequential([])
        assert seq.forward(99) == 99

    def test_type_mismatch_raises(self) -> None:
        seq = P.Sequential([lambda *_: (1, 2), lambda x: x])
        with pytest.raises(TypeError):
            seq.forward(0)


@pytest.mark.asyncio
class TestAsyncParallel:
    """`AsyncParallel` preserves order even when tasks finish out‑of‑order."""

    async def test_order_preserved(self) -> None:
        async def slow(x):
            await asyncio.sleep(0.02 if x == 1 else 0)
            return x
        par = P.AsyncParallel([slow, slow])
        assert await par.aforward(1, 2) == (1, 2)

    async def test_single_module(self) -> None:
        par = P.AsyncParallel([lambda x: x + 1])
        assert await par.aforward(3) == (4,)

    async def test_exception_propagates(self) -> None:
        async def bad(x):
            raise ValueError("boom")
        par = P.AsyncParallel([bad])
        with pytest.raises(ValueError):
            await par.aforward(0)


class TestProcessLoop:
    """Extended tests for process_loop batch orchestration."""

    def test_multiple_modules(self):
        b1 = P.Chunk([1, 2, 3], n=3)
        mods = [P.Func(lambda x: x + 1), P.Func(lambda x: x * 2)]
        out = list(P.process_loop(mods, b1))
        assert out == [4, 6, 8]  # ((1+1)*2), ((2+1)*2), ((3+1)*2)

    def test_keyword_arguments_passed(self):
        class AddXY(P.Process):
            def forward(self, x, y=0):
                return x + y

        b1 = P.Chunk([1, 2], n=2)
        b2 = P.Chunk([10, 20], n=2)
        out = list(P.process_loop([AddXY()], b1, b2))
        assert out == [11, 22]

    def test_zero_length_inputs(self):
        b1 = P.Chunk([], n=0)
        out = list(P.process_loop([lambda x: x], b1))
        assert out == []

    def test_empty_modules_list_returns_input(self):
        b = P.Chunk([1, 2, 3], n=3)
        out = list(P.process_loop([], b))
        assert out == [1, 2, 3]


    def test_happy_path(self) -> None:
        b1 = P.B([1, 2])
        b2 = P.B([10, 20])
        out = list(P.process_loop([lambda x, y: x + y], (b1, b2)))
        assert out == [11, 22]

    def test_length_mismatch(self) -> None:
        b1 = P.B([1])
        b2 = P.B([10, 20])
        with pytest.raises(ValueError):
            list(P.process_loop([lambda x, y: x + y], (b1, b2)))


@pytest.mark.asyncio
class TestCreateTaskExtended:
    """Extended tests for `create_task` async task generation."""

    async def test_with_plain_async_func(self):
        async def mul(x):
            return x * 3

        task = P.create_task(mul, 3)
        assert await task == 9

    async def test_with_plain_sync_func(self):
        def square(x):
            return x * x

        task = P.create_task(square, 5)
        assert await task == 25

    async def test_with_func_object(self):
        class Mult:
            def __call__(self, x):
                return x * 4

        task = P.create_task(Mult(), 2)
        assert await task == 8


    async def test_with_async_process(self) -> None:
        task = P.create_task(_AEchoProcess(), 5)
        assert await task == 5

    async def test_with_sync_func(self) -> None:
        task = P.create_task(lambda x: x * 2, 4)
        assert await task == 8



class TestReduceExtended:
    """Extended test coverage for `reduce` logic."""

    def test_reduce_with_init_val(self):
        b = P.Chunk([2, 3], n=2)
        result = P.reduce(lambda a, b: a * b, b, init_val=4)
        assert result == 24  # 4 * 2 * 3

    def test_reduce_empty_with_init_val(self):
        b = P.Chunk([], n=0)
        result = P.reduce(lambda a, b: a + b, b, init_val=100)
        assert result == 100

    def test_reduce_raises_without_init_on_empty(self):
        b = P.Chunk([], n=0)
        with pytest.raises(ValueError):
            P.reduce(lambda a, b: a + b, b)

    def test_reduce_multiple_modules(self):
        b = P.Chunk([1, 2, 3], n=3)
        mods = [P.Func(lambda a, b: a + b), P.Func(lambda a, b: a * b)]
        result = P.reduce(b, *mods)
        assert result == 9  # (1+2)=3, (3*3)=9


    def test_reduce_sum(self) -> None:
        vals = P.B([1, 2, 3])
        total = P.reduce(vals, lambda a, b: a + b)
        assert total == 6





@pytest.mark.asyncio
class TestAsyncReduce:
    """Additional tests for async_reduce to cover edge and error cases."""

    async def test_reduce_product(self) -> None:
        vals = P.B([2, 3, 4])
        async def mul(a, b):
            await asyncio.sleep(0)
            return a * b
        product = await P.async_reduce(vals, mul)
        assert product == 24

    async def test_async_reduce_no_init_val(self):
        chunk = P.Chunk([1, 2, 3], n=3)

        async def add(a, b):
            return a + b

        result = await P.async_reduce(add, chunk)
        assert result == 6

    async def test_async_reduce_empty_with_init_val(self):
        chunk = P.Chunk([], n=0)

        async def add(a, b):
            return a + b

        result = await P.async_reduce(add, chunk, init_val=10)
        assert result == 10

    async def test_async_reduce_raises_on_empty_without_init_val(self):
        chunk = P.Chunk([], n=0)

        async def add(a, b):
            return a + b

        with pytest.raises(ValueError):
            await P.async_reduce(add, chunk)


class TestStreamSequenceExtended:
    """Extended tests for StreamSequence to capture edge cases."""

    def test_stream_sequence_empty_input(self):
        pre = P.Func(lambda xs: (x for x in xs))
        mod = P.Func(lambda gen: (x + 1 for x in gen))
        post = P.Func(list)
        seq = P.StreamSequence(pre, mod, post)
        out = seq.forward([])
        assert out == []

    def test_stream_sequence_none_input(self):
        pre = P.Func(lambda _: (x for x in []))
        mod = P.Func(lambda gen: (x + 1 for x in gen))
        post = P.Func(list)
        seq = P.StreamSequence(pre, mod, post)
        out = seq.forward(None)
        assert out == []


@pytest.mark.asyncio
class TestAsyncStreamSequenceExtended:
    """Extended tests for AsyncStreamSequence edge behaviors."""

    async def test_async_stream_sequence_empty(self):
        async def pre(xs):
            for x in xs:
                if False:
                    yield x

        async def mod(gen):
            async for x in gen:
                yield x + 1

        post = P.Func(list)
        seq = P.AsyncStreamSequence(pre, mod, post)
        out = await seq.astream([])
        assert out == []

    async def test_async_stream_sequence_none(self):
        async def pre(_):
            return
            yield  # needed to make it async generator

        async def mod(gen):
            async for x in gen:
                yield x + 1

        post = P.Func(list)
        seq = P.AsyncStreamSequence(pre, mod, post)
        out = await seq.astream(None)
        assert out == []


# import asyncio
# import typing
# from typing import Any
# from dachi.proc import _process as core
# from dachi.proc import Module
# from dachi.proc import _process
# from dachi.store import Param
# import numpy as np
# # TODO: remove
# from dachi.inst import Cue
# import pytest


# class Append(
#     core.Module, core.AsyncModule, 
#     core.StreamModule, core.AsyncStreamModule
# ):

#     def __init__(self, append: str):
#         super().__init__()
#         self._append = append

#     def forward(self, name: str='') -> Any:
#         return name + self._append
    
#     async def aforward(self, name: str=''):
#         return self.forward(name)

#     def stream(self, name: str=''):
#         yield self.forward(name)
    
#     async def astream(self, name: str=''):
#         for v in self.stream(name):
#             yield v


# class Append2(
#     core.Module, core.AsyncModule, 
#     core.StreamModule, core.AsyncStreamModule
# ):

#     def __init__(self, append: str):
#         super().__init__()
#         self._append = append

#     def forward(self, val1: str, val2: str) -> Any:
#         return val1 + val2 + self._append

#     async def aforward(self, *args, **kwargs):
#         return self.forward(*args, **kwargs)

#     def stream(self, *args, **kwargs):
#         yield self.forward(*args, **kwargs)
    
#     async def astream(self, *args, **kwargs):
#         for v in self.stream(*args, **kwargs):
#             yield v


# class RefinerAppender(
#     core.Module, core.AsyncModule, 
#     core.StreamModule, core.AsyncStreamModule
# ):

#     def __init__(self, append: str):
#         super().__init__()
#         self._append = append

#     def forward(self, cur: str, val: str) -> Any:
#         if cur is None:
#             return val + self._append
#         return cur + val + self._append

#     async def aforward(self, *args, **kwargs):
#         return self.forward(*args, **kwargs)

#     def stream(self, *args, **kwargs):
#         yield self.forward(*args, **kwargs)
    
#     async def astream(self, *args, **kwargs):
#         for v in self.stream(*args, **kwargs):
#             yield v


# class WriteOut(core.Module):

#     def __init__(self, append: str):
#         super().__init__()
#         self._append = append

#     def forward(self, val: str) -> Any:
#         return val + self._append

#     def stream(self, val: str) -> Any:
#         for v in val:
#             yield v
#         for v in self._append:
#             yield v

#     async def aforward(self, *args, **kwargs):
#         return self.forward(*args, **kwargs)
    
#     async def astream(self, *args, **kwargs):
#         for v in self.stream(*args, **kwargs):
#             yield v


# class WaitAppend(core.Module):

#     def __init__(self, append: str):
#         super().__init__()
#         self._append = append

#     def forward(self, name: str='') -> Any:
#         return name + self._append

#     def stream(self, *args, **kwargs):
#         yield self.forward(*args, **kwargs)

#     async def aforward(self, *args, **kwargs):
#         return self.forward(*args, **kwargs)
    
#     async def astream(self, *args, **kwargs):
#         for v in self.stream(*args, **kwargs):
#             yield v

# def _s(x):
#     cur = ''
#     for x_i in x:
#         yield x_i


# class TestStreamer:

#     def test_streamable_streams_characters(self):

#         streamer = _process.Streamer(_s('xyz'))
#         partial = streamer()
#         assert partial.dx == 'x'

#     def test_streamable_streams_characters_to_end(self):

#         streamer = _process.Streamer(_s('xyz'))
#         partial = streamer()
#         partial = streamer()
#         partial = streamer()
#         assert partial.dx == 'z'

#     def test_streamer_gets_next_item(self):

#         streamer = _process.Streamer(
#             iter([(1, 0), (2, 2), (3, 2)])
#         )
#         partial = streamer()
#         assert partial.dx == (1, 0)
#         assert partial.complete is False

#     def test_streamer_gets_final_item(self):

#         streamer = _process.Streamer(
#             iter([(1, 0), (2, 2), (3, 2)])
#         )
#         partial = streamer()
#         partial = streamer()
#         partial = streamer()
#         partial = streamer()
#         assert partial.dx == (3, 2)
#         assert partial.complete is True


# class TestSequential:

#     def test_sequential_does_nothing_if_zero(self):

#         sequential = _process.Sequential()
#         res = sequential('x')
#         assert res == 'x'

#     def test_sequential_appends(self):

#         sequential = _process.Sequential(Append('z'))
#         res = sequential('x')
#         assert res == 'xz'

#     def test_sequential_appends2(self):

#         sequential = _process.Sequential(Append('z'), Append('y'))
#         res = sequential('x')
#         assert res == 'xzy'

#     def test_sequential_works_with_two_inputs(self):

#         sequential = _process.Sequential(Append2('x'))
#         res = sequential('x', 'y')
#         assert res == 'xyx'

#     def test_len_returns_correct_len(self):

#         sequential = _process.Sequential(Append('z'), Append('y'))
#         return len(sequential) == 2


# class TestModuleList(object):

#     def test_module_list_has1(self):

#         module_list = _process.ModuleList([Append('z')])
#         assert len(module_list) == 1

#     def test_module_list_has2(self):

#         module_list = _process.ModuleList([Append('z'), Append('z')])
#         assert len(module_list) == 2

#     def test_module_list_two_children(self):

#         module_list = _process.ModuleList([Append('z'), Append('z')])
#         assert len(list(module_list.children())) == 2


# class TestForwardStream:

#     def test_forward_executes_module(self):

#         append = Append('s')
#         res = _process.forward(append, 't')
#         assert res == 'ts'

#     @pytest.mark.asyncio
#     async def test_aforward_executes_module(self):

#         append = Append('s')
#         res = await _process.aforward(append, 't')
#         assert res == 'ts'

#     def test_stream_executes_module(self):

#         append = Append('s')
#         for r in _process.stream(append, 't'):
#             pass
#         assert r == 'ts'


# class TestBatched:

#     def test_batched_len_is_correct(self):
        
#         batched = _process.Batched([1,2,3,4,5,6], size=3)
#         assert len(batched) == 2

#     def test_loop_over_batches_returns_correct_values(self):
        
#         batched = _process.Batched([1,2,3,4,5,6], size=3)
#         batch_list = list(batched)
#         assert batch_list[0] == [1, 2, 3]
#         assert batch_list[1] == [4, 5, 6]

#     def test_loop_over_batches_returns_correct_values_with_two(self):
        
#         batched = _process.Batched([1,2,3,4,5,6], [0, 1,2,3,4,5], size=3)
#         batch_list = list(batched)
#         assert batch_list[0][0] == [1, 2, 3]
#         assert batch_list[0][1] == [0, 1, 2]

#     def test_shuffle_changes_the_order(self):
        
#         batched = _process.Batched([1,2,3,4,5,6], [0, 1,2,3,4,5], size=3)

#         np.random.seed(0)
#         batched = batched.shuffle()
#         batch_list = list(batched)
#         assert batch_list[0][0] != [1, 2, 3]


# class TestReduce:

#     def test_reduce_reduces_with_init(self):

#         r = Append('x')
#         b = RefinerAppender('y')
#         res = _process.reduce(
#             b, _process.B('xy'), init_mod=r
#         )
#         assert res == 'xxyy'

#     def test_reduce_reduces_with_three_values(self):

#         r = Append('x')
#         b = RefinerAppender('y')
#         res = _process.reduce(
#             b, _process.B('xyz'), init_mod=r
#         )
#         assert res == 'xxyyzy'

#     def test_reduce_reduces_without_init(self):

#         b = RefinerAppender('y')
#         res = _process.reduce(
#             b, _process.B('xy')
#         )
#         assert res == 'xyyy'


# class TestMulti:
    
#     def test_that_multi_loops_over_the_modules(self):

#         module = _process.MultiParallel(
#             [Append('x'), Append('y')]
#         )
#         res = module('hi')
#         assert res[0] == 'hix'
#         assert res[1] == 'hiy'

#     def test_that_multi_loops_over_the_modules_with_a_batch(self):

#         x = _process.B(['hi', 'bye'])
#         module = _process.MultiParallel(
#             Append('x')
#         )
#         res = module(x)
#         assert res[0] == 'hix'
#         assert res[1] == 'byex'


# class TestAsync:
    
#     def test_that_async_loops_over_the_modules(self):

#         module = _process.AsyncParallel(
#             [Append('x'), Append('y')]
#         )
#         res = module('hi')
#         assert res[0] == 'hix'
#         assert res[1] == 'hiy'

#     def test_that_async_loops_over_the_modules_with_a_batch(self):

#         x = _process.B(['hi', 'bye'])
#         module = _process.AsyncParallel(
#             Append('x')
#         )
#         res = module(x)
#         assert res[0] == 'hix'
#         assert res[1] == 'byex'

#     def test_that_async_loops_over_the_modules_with_a_batch_with_two(self):

#         x = _process.B(['hi', 'bye'])
#         module = _process.AsyncParallel(
#             Append2('x')
#         )
#         res = module(x, 'z')
#         assert res[0] == 'hizx'
#         assert res[1] == 'byezx'


# class NestedModule(Module):

#     def __init__(self, child: Module):
#         super().__init__()
#         self.child = child
#         self.p = Param(
#             name='p',
#             data=Cue(text='Do this')
#         )

#     def forward(self) -> Any:
#         return None


# class TestParallelLoop:

#     def test_parallel_loop_with_two_modules(self):

#         modules = _process.ModuleList([Append('s'), Append('t')])
#         ress = []
#         for m, r, kwargs in _process.parallel_loop(
#             modules, 'r'
#         ):
#             ress.append(m(*r, **kwargs))
#         assert ress[0] == 'rs'
#         assert ress[1] == 'rt'

#     def test_parallel_loop_with_two_inputs(self):

#         # modules = _process.ModuleList([Append('s'), Append('t')])
#         ress = []
#         for m, r, kwargs in _process.parallel_loop(
#             Append('s'), _process.B(['r', 'v'])
#         ):
#             ress.append(m(*r, **kwargs))
#         assert ress[0] == 'rs'
#         assert ress[1] == 'vs'

#     def test_parallel_loop_with_two_inputs_and_two_modles(self):

#         modules = _process.ModuleList([Append('s'), Append('t')])
#         ress = []
#         for m, r, kwargs in _process.parallel_loop(
#             modules, _process.B(['r', 'v'])
#         ):
#             ress.append(m(*r, **kwargs))
#         assert ress[0] == 'rs'
#         assert ress[1] == 'vt'


# class TestModule:

#     def test_module_forward_outputs_correct_value(self):

#         append = Append('_t')
#         assert append.forward('x') == 'x_t'

#     def test_async_runs_the_model_asynchronously(self):
        
#         module = Append('t')

#         async def run_model(data: typing.List[str]):

#             tasks = []
#             async with asyncio.TaskGroup() as tg:
#                 for data_i in data:
#                     tasks.append(
#                         tg.create_task(module.aforward(data_i))
#                     )

#             return list(task.result() for task in tasks)

#         with asyncio.Runner() as runner:
#             results = runner.run(run_model(['hi', 'bye']))
        
#         assert results[0] == 'hit'
#         assert results[1] == 'byet'

#     def test_stream_forward_returns_the_results(self):
        
#         module = Append('t')

#         res = ''
#         for dx in module.stream('xyz'):
#             res += dx
#         assert res == 'xyzt'

#     def test_children_returns_no_children(self):
        
#         module = Append('t')

#         children = list(module.children())
#         assert len(children) == 0

#     def test_children_returns_two_children_with_nested(self):
        
#         module = NestedModule(NestedModule(Append('a')))

#         children = list(module.children())
#         assert len(children) == 2

#     def test_parameters_returns_all_parameters(self):
        
#         module = NestedModule(NestedModule(Append('a')))

#         children = list(module.parameters())
#         assert len(children) == 2

#     def test_streamable_streams_characters(self):

#         writer = WriteOut('')
#         results = []
#         for dx in writer.stream('xyz'):
#             results.append(dx)
#         assert results == list('xyz')

