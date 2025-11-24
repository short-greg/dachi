"""Updated test‑suite for _process.py.

Each test class targets a single public surface.  Tests are small, black‑box, and
nit‑picky; each verifies *one* behaviour (except where a function returns
multiple values that must be asserted together).

Google‑style docstrings are used throughout.
"""
import asyncio
from typing import Any, Iterator, List, Tuple
import pytest
import pydantic

from dachi import proc as P


async def _async_identity(x: int) -> int:  # noqa: D401
    return x


def _gen(x: int) -> Iterator[int]:  # noqa: D401
    yield x
    yield x + 1


class _EchoProcess(P.Process):
    calls: List[Tuple] = pydantic.Field(default_factory=list)

    def forward(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return args[0] if args else kwargs.get("x")


class _AEchoProcess(P.AsyncProcess):
    calls: List[Tuple] = pydantic.Field(default_factory=list)

    async def aforward(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return args[0] if args else kwargs.get("x")


class _SEchoProcess(P.StreamProcess):
    n: int

    def stream(self, x):
        for _ in range(self.n):
            yield x


class _ASEchoProcess(P.AsyncStreamProcess):
    n: int

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



# # class _EchoProcess(P.Process):
# #     def forward(self, x):
# #         return x

# # class _AEchoProcess(P.AsyncProcess):
# #     async def aforward(self, x):
# #         return x

# # class _SEchoProcess(P.StreamProcess):
# #     def __init__(self, n: int):
# #         self.n = n
# #     def stream(self, x):
# #         for _ in range(self.n):
# #             yield x


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
class TestAForward:
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
        with pytest.raises(TypeError):
            await P.aforward(object())  # type: ignore[arg-type]


# class TestStream:
#     """`stream` collapses scalars into single-item iterators and preserves generators."""

#     def test_stream_with_proc(self):
#         proc = _SEchoProcess(n=3)
#         assert list(P.stream(proc, "a")) == ["a", "a", "a"]

#     def test_stream_plain_func(self):
#         out_iter = P.stream(lambda x: x * 2, 2)
#         assert list(out_iter) == [4]

#     def test_stream_plain_generator(self):
#         def yield_one(x):
#             yield x
#         assert list(P.stream(yield_one, 9)) == [9]

#     def test_async_gen_not_supported(self):
#         async def _agen(x):
#             yield x
#         with pytest.raises(TypeError):
#             list(P.stream(_agen, 1))  # type: ignore[arg-type]


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
        with pytest.raises(TypeError):
            async for _ in P.astream(123):  # type: ignore[arg-type]
                pass


class TestRecur:
    """Iterator `I` yields the same object *n* times."""

    def test_positive(self) -> None:
        obj = object()
        assert list(P.Recur(data=obj, n=3)) == [obj, obj, obj]

    def test_zero(self) -> None:
        assert list(P.Recur(data=1, n=0)) == []

    def test_negative(self) -> None:
        with pytest.raises(ValueError):
            list(P.Recur(data=1, n=-1))


class TestChunk:
    """`B` batches an iterable preserving order and supports `.m` helper."""

    def test_preserves_order(self) -> None:
        b = P.Chunk(data=[1, 2, 3])
        assert list(b) == [[1], [2], [3]]
        assert b.n is None

    def test_truncate(self) -> None:
        b = P.Chunk(data=[1, 2, 3], n=2)
        assert list(b)[1] == [2, 3]
        assert b.n == 2

    def test_classmethod_m(self) -> None:
        b1, b2 = P.Chunk.m([1, 2], [3, 4])
        assert list(b1) == [[1], [2]]
        assert list(b2) == [[3], [4]]
        assert b1.n == b2.n == None

    def test_non_iterable(self) -> None:
        with pytest.raises(TypeError):
            P.Chunk(123)  # type: ignore[arg-type]



class TestSequential:
    """`Sequential` composes heterogeneous modules in order."""

    def test_three_stage_pipeline(self) -> None:
        seq = P.Sequential(
            vals=[_EchoProcess(), _EchoProcess()]
        )
        assert seq.forward(2) == 2

    def test_empty_sequential_identity(self) -> None:
        seq = P.Sequential(vals=[])
        assert seq.forward(99) == 99

#     # def test_type_mismatch_raises(self) -> None:
#     #     seq = P.Sequential(items=[lambda *_: (1, 2), lambda x: x])
#     #     with pytest.raises(TypeError):
#     #         seq.forward(0)



class TestProcessLoop:
    """Extended tests for process_loop batch orchestration."""

    class SyncP(P.Process):
        """A simple sync process for testing."""
        def forward(self, x: int) -> int:
            return x + 1

    def test_multiple_modules(self):
        b1 = P.Chunk(data=[1, 2, 3], n=None)
        mods = [self.SyncP(), self.SyncP(), self.SyncP()]
        out = list(P.process_loop(mods, b1))
        assert out[0] == (mods[0], (1,), {})
        assert out[1] == (mods[1], (2,), {})
        assert out[2] == (mods[2], (3,), {})

    def test_keyword_arguments_passed(self):
        class AddXY(P.Process):
            def forward(self, x, y=0):
                return x + y

        b1 = P.Chunk(data=[1, 2], n=None)
        b2 = P.Chunk(data=[10, 20], n=None)
        add_xy = AddXY()
        out = list(P.process_loop(add_xy, b1, y=b2))
        assert out[0] == (add_xy, (1,), {'y': 10})
        assert out[1] == (add_xy, (2,), {'y': 20})

    def test_zero_length_inputs(self):
        b1 = P.Chunk(data=[], n=0)
        out = list(P.process_loop([], b1))
        assert out == []

    def test_empty_modules_list_returns_input(self):
        b = P.Chunk(data=[1, 2, 3], n=3)
        with pytest.raises(ValueError):
            list(P.process_loop([], b))

    def test_happy_path(self) -> None:
        b1 = P.Chunk(data=[1, 2])
        b2 = P.Chunk(data=[10, 20])
        p = lambda x, y: x + y
        out = list(P.process_loop(p, b1, b2))
        assert out[0][0] is p

    def test_length_mismatch(self) -> None:
        b1 = P.Chunk(data=[1])
        p = lambda x, y: x + y
        b2 = P.Chunk(data=[10, 20])
        with pytest.raises(ValueError):
            list(P.process_loop([p], b1, b2))


@pytest.mark.asyncio
class TestAsyncParallel:
    """`AsyncParallel` preserves order even when tasks finish out‑of‑order."""

    class AsyncP(P.AsyncProcess):
        """A simple async process for testing."""
        async def aforward(self, x: int) -> int:
            await asyncio.sleep(0.01)
            return x
        
    class SyncP(P.Process):
        """A simple sync process for testing."""
        def forward(self, x: int) -> int:
            return x + 1

    async def test_order_preserved(self) -> None:
        """Ensure that async tasks complete in the order they were started."""
        items = [self.AsyncP(), self.AsyncP()]
        par = P.AsyncParallel(vals=items)
        assert await par.aforward(1) == [1, 1]

    async def test_single_module(self) -> None:
        par = P.AsyncParallel(vals=[self.SyncP()])
        assert await par.aforward(3) == [4]



@pytest.mark.asyncio
class TestCreateTaskExtended:
    """Extended tests for `create_task` async task generation."""

    async def test_with_plain_async_func(self):
        async def mul(x):
            return x * 3
        
        async with asyncio.TaskGroup() as tg:
            res = P.create_proc_task(tg, mul, 3)
        
        assert res.result() == 9

    async def test_with_plain_sync_func(self):
        def square(x):
            return x * x

        async with asyncio.TaskGroup() as tg:
            task = P.create_proc_task(tg, square, 5)
        assert task.result() == 25

    async def test_with_func_object(self):
        class Mult:
            def __call__(self, x):
                return x * 4

        async with asyncio.TaskGroup() as tg:
            task = P.create_proc_task(tg, Mult(), 2)
        assert task.result() == 8


    async def test_with_async_process(self) -> None:

        async with asyncio.TaskGroup() as tg:
            task = P.create_proc_task(tg, _AEchoProcess(), 5)
        assert task.result() == 5

    async def test_with_sync_func(self) -> None:
        async with asyncio.TaskGroup() as tg:
            task = P.create_proc_task(tg, lambda x: x * 2, 4)
        assert task.result() == 8



class TestReduce:
    """Extended test coverage for `reduce` logic."""

    def test_reduce_with_init_val(self):
        b = P.Chunk(data=[2, 3], n=None)
        result = P.reduce(lambda a, b: a * b, b, init_val=4)
        assert result == 24  # 4 * 2 * 3

    def test_reduce_empty_with_init_val(self):
        b = P.Chunk(data=[], n=0)
        result = P.reduce(lambda a, b: a + b, b, init_val=100)
        assert result == 100

    def test_reduce_raises_without_init_on_empty(self):
        b = P.Chunk(data=[], n=0)
        with pytest.raises(ValueError):
            P.reduce(lambda a, b: a + b, b)

    def test_reduce_sum(self) -> None:
        vals = P.Chunk(data=[1, 2, 3])
        total = P.reduce(lambda a, b: a + b, vals, init_val=0)
        assert total == 6

    def test_reduce_sum(self) -> None:
        vals = P.Chunk(data=[1, 2, 3])
        with pytest.raises(ValueError):
            P.reduce(lambda a, b: a + b, vals)


@pytest.mark.asyncio
class TestAsyncReduce:
    """Additional tests for async_reduce to cover edge and error cases."""

    async def test_reduce_product(self) -> None:
        vals = P.chunk([2, 3, 4])
        async def mul(a, b):
            await asyncio.sleep(0)
            return a * b
        product = await P.async_reduce(mul, vals, init_val=1)
        assert product == 24

    async def test_async_reduce_no_init_val(self):
        chunk = P.chunk([1, 2, 3])

        async def add(a, b):
            return a + b

        result = await P.async_reduce(add, chunk, init_val=0)
        assert result == 6

    async def test_async_reduce_no_init_val(self):
        chunk = P.chunk([1, 2, 3])

        async def add(a, b):
            return a + b
        # Test that async_reduce doesn't work without an initial value
        with pytest.raises(ValueError):
            await P.async_reduce(add, chunk)

    async def test_async_reduce_empty_with_init_val(self):
        chunk = P.chunk([])

        async def add(a, b):
            return a + b

        result = await P.async_reduce(add, chunk, init_val=10)
        assert result == 10

    async def test_async_reduce_raises_on_empty_without_init_val(self):
        chunk = P.chunk([])

        async def add(a, b):
            return a + b

        with pytest.raises(ValueError):
            await P.async_reduce(add, chunk)
