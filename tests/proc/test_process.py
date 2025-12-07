
import asyncio
from typing import Any, Iterator, List, Tuple, TypeVar, Generic
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


class TestProcessSerialization:

    def test_spec_roundtrip_preserves_process(self):
        """Spec round-trip via to_spec() and from_spec() works."""
        original = _EchoProcess()
        spec = original.to_spec()

        restored = _EchoProcess.from_spec(spec)

        assert restored("test") == "test"


T = TypeVar('T')

class WrapProcess(P.Process, Generic[T]):
    """Process wrapper for testing generics."""
    inner: T

    def forward(self, *args, **kwargs):
        if hasattr(self.inner, 'forward'):
            return self.inner.forward(*args, **kwargs)
        return None


class TestProcessGenericSerialization:

    def test_to_spec_preserves_generic_type_parameter(self):
        """to_spec() preserves generic type parameter in KIND."""
        inner = _EchoProcess()
        original = WrapProcess[_EchoProcess](inner=inner)

        spec = original.to_spec()

        assert "_EchoProcess" in spec["inner"]["KIND"]

    def test_to_spec_preserves_union_type_parameter(self):
        """to_spec() preserves union type parameter in KIND."""
        inner = _AEchoProcess()
        original = WrapProcess[_EchoProcess | _AEchoProcess](inner=inner)

        spec = original.to_spec()

        assert "_AEchoProcess" in spec["inner"]["KIND"]

    def test_to_spec_with_different_union_members(self):
        """to_spec() works with different union type members."""
        # First with _EchoProcess
        inner1 = _EchoProcess()
        proc1 = WrapProcess[_EchoProcess | _AEchoProcess](inner=inner1)
        spec1 = proc1.to_spec()

        assert "_EchoProcess" in spec1["inner"]["KIND"]

        # Then with _AEchoProcess
        inner2 = _AEchoProcess()
        proc2 = WrapProcess[_EchoProcess | _AEchoProcess](inner=inner2)
        spec2 = proc2.to_spec()

        assert "_AEchoProcess" in spec2["inner"]["KIND"]

    def test_spec_roundtrip_with_union_type_parameter(self):
        """Spec round-trip with union type parameter works."""
        inner = _EchoProcess()
        original = WrapProcess[_EchoProcess | _AEchoProcess](inner=inner)
        spec = original.to_spec()

        restored = WrapProcess[_EchoProcess | _AEchoProcess].from_spec(spec)

        assert restored.forward("test") == "test"
        assert isinstance(restored.inner, _EchoProcess)


