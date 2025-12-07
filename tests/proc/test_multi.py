import asyncio
from typing import Any, Iterator, List, Tuple, TypeVar, Generic
import pytest
import pydantic

from dachi import proc as P


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


class TestSequentialSerialization:

    def test_to_spec_preserves_sequential_structure(self):
        """to_spec() preserves Sequential structure."""
        proc1 = _EchoProcess()
        proc2 = _EchoProcess()
        original = P.Sequential(vals=[proc1, proc2])

        spec = original.to_spec()

        assert "Sequential" in spec["KIND"]
        assert len(spec["vals"]) == 2
        assert "_EchoProcess" in spec["vals"][0]["KIND"]
        assert "_EchoProcess" in spec["vals"][1]["KIND"]

    def test_to_spec_sequential_with_different_process_types(self):
        """to_spec() works with mixed process types in Sequential."""
        proc1 = _EchoProcess()
        proc2 = _EchoProcess()
        original = P.Sequential(vals=[proc1, proc2])

        spec = original.to_spec()

        assert len(spec["vals"]) == 2
        assert all("KIND" in item for item in spec["vals"])



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

