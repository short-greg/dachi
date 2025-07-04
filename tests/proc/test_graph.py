
import asyncio
import pytest
from typing import AsyncIterator, Iterator
from typing import Iterator, AsyncIterator

from dachi.proc import _graph as G
from dachi.core import SerialDict
from dachi.utils import WAITING, UNDEFINED
from dachi.proc import _process as P
from dachi.proc._graph import Stream 
from dachi.utils import UNDEFINED 


class MyProcess:

    def forward(self, x, y):
        return x + y


### Test helpers / stubs
class _DummyNode(G.BaseNode):
    """Concrete subclass of :class:`dachi.proc._graph.BaseNode` for testing.

    Implements the two abstract hooks with minimal behaviour and allows the
    `_val` payload to be injected at construction time so that the black‑box
    tests can manipulate the node state directly without touching internal
    logic.
    """

    #### AsyncProcess hook 
    async def aforward(self, by=None):  # pragma: no cover – trivial stub
        return self.val

    #### Graph hook
    def incoming(self):  # pragma: no cover – not used in these tests
        return iter(())


# @pytest.fixture(autouse=True)
# def _patch_graph_stubs(monkeypatch):
#     """Monkey‑patch :class:`~dachi.proc._graph.T` and :class:`~dachi.proc._graph.Idx`
#     with trivial stand‑ins so that tests for ``__getitem__`` / ``detach`` can
#     focus on *BaseNode* behaviour in isolation without having to satisfy the
#     full constructor signatures of the real classes.
#     """

#     class _StubIdx:
#         def __init__(self, node, idx):
#             self.node = node
#             self.idx = idx

#     class _StubT:
#         def __init__(self, val, src):
#             self.val = val
#             self.src = src

#     monkeypatch.setattr(G, "Idx", _StubIdx, raising=True)
#     monkeypatch.setattr(G, "T", _StubT, raising=True)


class TestBaseNode:
    """Black‑box, behaviour‑level tests for :class:`dachi.proc._graph.BaseNode`.

    Each test checks *exactly one* observable behaviour to maximise coherence
    and debuggability.
    """

    ####  Construction & basic attributes
    def test_default_initialisation_is_undefined(self):
        """Node starts with ``UNDEFINED`` value and reports as undefined."""
        node = _DummyNode()
        assert node.val is G.UNDEFINED
        assert node.is_undefined() is True
        assert node.name is None and node.annotation is None


    #### Label
    def test_label_updates_both_fields_and_is_chainable(self):
        """Calling *label* with *name* **and** *annotation* mutates in‑place and
        returns *self* (enabling chaining)."""
        node = _DummyNode()
        out = node.label(name="foo", annotation="bar")
        assert out is node
        assert node.name == "foo"
        assert node.annotation == "bar"

    def test_label_partial_update_keeps_existing_values(self):
        """Passing *None* for a parameter must leave the existing value intact."""
        node = _DummyNode()
        node.label(name="foo", annotation="bar")
        node.label(annotation="baz")  # only annotation changes
        assert node.name == "foo"
        assert node.annotation == "baz"

    def test_is_undefined_true_for_WAITING(self):
        """``WAITING`` sentinel also counts as undefined."""
        node = _DummyNode(val=G.WAITING)
        assert node.is_undefined() is True

    def test_is_undefined_false_for_regular_value(self):
        """Concrete values should flip *is_undefined* to *False*."""
        node = _DummyNode(val=0)
        assert node.is_undefined() is False

    def test_getitem_returns_wrapped_slice_when_defined(self):
        """When the node holds a sequence, ``node[idx]`` must forward the
        element at *idx* into a new *T* instance whose *src* is an *Idx*
        wrapper pointing back to *node*.
        """
        seq = [10, 20, 30]
        node = _DummyNode(val=seq)
        wrapped = node[1]
        assert isinstance(wrapped, G.T)
        assert wrapped.val == 20
        assert isinstance(wrapped.src, G.Idx)
        # assert wrapped.src.node is node and wrapped.src.idx == 1

    def test_getitem_returns_wrapped_UNDEFINED_when_value_missing(self):
        """If the node value is undefined, the wrapper should carry the same
        sentinel.*"""
        node = _DummyNode()  # UNDEFINED by default
        wrapped = node[5]
        assert wrapped.val is G.UNDEFINED
        assert isinstance(wrapped.src, G.Idx)
        assert wrapped.src.idx == 5

    def test_detach_produces_T_with_null_src(self):
        """*detach* must return a *T* whose ``src`` field is *None* but whose
        ``val`` is identical to the current node value."""
        node = _DummyNode(val=99)
        detached = node.detach()
        assert detached.val == 99
        assert detached.src is None

    @pytest.mark.asyncio
    async def test_aforward_default_noop(self):
        """Base implementation is a *pass* – should simply will use the default value."""
        node = _DummyNode(val=123)
        assert await node.aforward() is 123


# -------------------------------------------------
# Auto‑applied fixtures
# -------------------------------------------------
# @pytest.fixture(autouse=True)
# def _patch_partial(monkeypatch):
#     """Patch :pydata:`dachi.proc._graph.Partial` with a minimalist stub so we can
#     test the *isinstance(..., Partial)* branch without importing the real impl.
#     """
#     class _StubPartial:  # pylint: disable=too-few-public-methods
#         def __init__(self):
#             self.complete = False

#     monkeypatch.setattr(G, "Partial", _StubPartial, raising=False)


# -------------------------------------------------
# Test‑suite for Var
# -------------------------------------------------
class TestVar:
    """Thorough black‑box tests for :class:`dachi.proc._graph.Var`. Each case is
    intentionally focused on a single observable behaviour.
    """
    #### 1. Cached value path
    @pytest.mark.asyncio
    async def test_aforward_returns_cached_val(self):
        """When ``_val`` is already defined (non‑undefined), it must be returned
        immediately without consulting *by* or *_src*."""
        v = G.Var()
        v.val = 123  # simulate previously‑computed value

        assert await v.aforward() == 123

    # "by" dict override path
    @pytest.mark.asyncio
    async def test_aforward_uses_by_mapping(self):
        """If the probe dictionary carries an entry for *self* that is **not** a
        :class:`Partial`, that value should short‑circuit evaluation."""
        v = G.Var()

        by = {v: 456}
        assert await v.aforward(by) == 456

    ### Aforward
    @pytest.mark.asyncio
    async def test_aforward_raises_error(self):
        """If *_src* is present, Var must:
        1. Call *probe* on each upstream node.
        2. Invoke the source callable.
        3. Cache the result in *by[self]* and return it.
        """
        v = G.Var()
        v.val = G.UNDEFINED

        by: dict = {}
        with pytest.raises(RuntimeError):
            # no *by* entry for *v*, so should raise RuntimeError
            await v.aforward(by)

    @pytest.mark.asyncio
    async def test_aforward_retrieves_value_from_var(self):
        """If *_src* is present, Var must:
        1. Call *probe* on each upstream node.
        2. Invoke the source callable.
        3. Cache the result in *by[self]* and return it.
        """
        v = G.Var(val=2)

        by: dict = {}
            # no *by* entry for *v*, so should raise RuntimeError
        res = await v.aforward(by)
        assert res == 2  # value retrieved from Var's _val

    @pytest.mark.asyncio
    async def test_aforward_retrieves_value_from_var_with_by(self):
        """If *_src* is present, Var must:
        1. Call *probe* on each upstream node.
        2. Invoke the source callable.
        3. Cache the result in *by[self]* and return it.
        """
        v = G.Var()

        by: dict = {v: 3}
            # no *by* entry for *v*, so should raise RuntimeError
        res = await v.aforward(by)
        assert res == 3  # value retrieved from Var's _val

    @pytest.mark.asyncio
    async def test_incoming_retrieves_no_notdes(self):
        """If *_src* is present, Var must:
        1. Call *probe* on each upstream node.
        2. Invoke the source callable.
        3. Cache the result in *by[self]* and return it.
        """
        v = G.Var()

        incoming = list(v.incoming())
        assert len(incoming) == 0


from dachi.proc._graph import (
    t,             # sync-task factory
    async_t,       # async-task factory
    Var,           # constant/input node
    UNDEFINED,     # sentinel for “no value yet”
    T
)
class _SyncProc:
    """Callable that mimics a synchronous Process."""
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        # if an upstream kwarg exists, just return it + 1 so we can
        # confirm dependency resolution
        if kwargs:
            return next(iter(kwargs.values())) + 1
        return self.value


class _AsyncProc:
    """Callable that mimics an asynchronous AsyncProcess."""
    def __init__(self, value):
        self.value = value
        self.calls = 0

    async def __call__(self, **kwargs):
        self.calls += 1
        await asyncio.sleep(0)          # prove we’re really async
        return self.value


@pytest.mark.asyncio
class TestT:
    async def test_post_init_defaults(self):
        proc = _SyncProc(value=42)
        node = t(p=proc)
        assert node.is_async is False
        assert node.val is UNDEFINED                    # starts undefined

    async def test_aforward_sync_exec_and_cache(self):
        proc = _SyncProc(10)
        node = t(proc)
        probe = {}

        out1 = await node.aforward(probe)
        assert out1 == 10          
        assert probe[node] == 10

        # second call must *not* re-invoke the underlying process
        calls_before = proc.calls
        out2 = await node.aforward(probe)
        assert out2 == 10
        assert proc.calls == calls_before               # unchanged

    async def test_aforward_probe_dict_short_circuit(self):
        proc = _SyncProc(999)
        node = t(proc)
        probe = {node: 123}                             # override

        result = await node.aforward(probe)
        assert result == 123
        assert proc.calls == 0                          # never executed
        assert node.val is UNDEFINED                    # cache untouched

    async def test_aforward_async_exec(self):
        proc = _AsyncProc(7)
        node = async_t(proc)

        res = await node.aforward()
        assert res == 7
        assert proc.calls == 1

    async def test_aforward_dependency_resolution(self):
        # Var supplies input ‘x’; process returns x + 1 (see _SyncProc logic)
        x_var = Var()
        x_var.val = 3                                   # make it concrete

        proc = _SyncProc(None)
        node = t(proc, x=x_var)                         # node(x) = x + 1

        res = await node.aforward()
        assert res == 4

    async def test_aforward_undefined_result_error(self):
        class _BadProc:
            def __init__(self):
                self.calls = 0
            def __call__(self, **kwargs):
                self.calls += 1
                return UNDEFINED

        node = t(_BadProc())
        with pytest.raises(RuntimeError):
            await node.aforward()

        # value should remain undefined after failure
        assert node.val is UNDEFINED

    async def test_aforward_exception_bubbles_and_no_cache(self):
        class _BoomProc:
            def __init__(self):
                self.calls = 0
            def __call__(self, **kwargs):
                self.calls += 1
                raise ValueError("boom")

        proc = _BoomProc()
        node = t(proc)

        with pytest.raises(ValueError):
            await node.aforward()

        assert node.val is UNDEFINED
        assert proc.calls == 1

    @pytest.mark.asyncio
    async def test_aforward_returns_cached_val(self):
        tnode = G.T(src=lambda: 1, args=SerialDict(data={}), is_async=False)
        tnode.val = 42
        out = await tnode.aforward()
        assert out == 42

#     # ------------- fast‑return: probe‑dict ----------------------------
    @pytest.mark.asyncio
    async def test_aforward_uses_probe_dict(self):
        tnode = G.T(src=lambda: 0, args=SerialDict(data={}), is_async=False)
        probe = {tnode: 99}
        out = await tnode.aforward(probe)
        assert out == 99

#     # ------------- sync execution ------------------------------------
    @pytest.mark.asyncio
    async def test_aforward_executes_sync_src_once_and_caches(self):
        counter = _CallCounter(async_=False, ret=7)
        tnode = G.T(src=counter.fn, args=SerialDict(data={}), is_async=False)
        probe: dict = {}
        result1 = await tnode.aforward(probe)
        result2 = await tnode.aforward(probe)
        assert result1 == result2 == 7
        assert counter.calls == 1  # memoisation worked
        assert probe[tnode] == 7

#     # ------------- async execution -----------------------------------
    @pytest.mark.asyncio
    async def test_aforward_executes_async_src(self):
        counter = _CallCounter(async_=True, ret=13)
        tnode = G.T(src=counter.fn, args=SerialDict(data={}), is_async=True)
        res = await tnode.aforward()
        assert res == 13 and counter.calls == 1

    @pytest.mark.asyncio
    async def test_aforward_raises_when_src_returns_undefined(self):
        bad_src = _CallCounter(async_=False, ret=G.UNDEFINED)
        tnode = G.T(src=bad_src.fn, args=SerialDict(data={}), is_async=False)
        with pytest.raises(RuntimeError):
            await tnode.aforward()


class _CallCounter:
    def __init__(self, *, async_=False, ret):
        self.calls = 0
        self.ret = ret
        self.async_ = async_
        if async_:
            async def _af(*args, **kwargs):
                self.calls += 1
                await asyncio.sleep(0)
                return self.ret
            self.fn = _af
        else:
            def _f(*args, **kwargs):
                self.calls += 1
                return self.ret
            self.fn = _f



class TestIdx:
    """
    Unit tests for Idx.index – direct indexing of a value.
    Each test isolates exactly one contract of the method.
    """

    # --- positive scalar indices ------------------------------------------
    def test_scalar_positive(self):
        idx = G.Idx(idx=1)
        assert idx.index(['a', 'b', 'c']) == 'b'

    def test_scalar_negative(self):
        idx = G.Idx(idx=-1)
        assert idx.index([10, 20, 30]) == 30

    # --- positive list indices --------------------------------------------
    def test_list_positive(self):
        idx = G.Idx(idx=[0, 2])
        assert idx.index(['x', 'y', 'z']) == ['x', 'z']

    def test_list_mixed_sign(self):
        idx = G.Idx(idx=[-1, 0])
        assert idx.index([7, 8, 9]) == [9, 7]

    def test_list_with_duplicates(self):
        idx = G.Idx(idx=[0, 0, 2])
        assert idx.index([1, 2, 3]) == [1, 1, 3]

    # --- error paths -------------------------------------------------------
    def test_scalar_out_of_range(self):
        idx = G.Idx(idx=3)
        with pytest.raises(IndexError):
            idx.index([1, 2])

    def test_list_contains_out_of_range(self):
        idx = G.Idx(idx=[0, 5])
        with pytest.raises(IndexError):
            idx.index([1, 2])

    def test_non_indexable_value(self):
        idx = G.Idx(idx=0)
        with pytest.raises(TypeError):
            idx.index(42)

    """
    Unit tests for Idx.forward – should behave identically to index().
    """

    def test_forward_scalar(self):
        idx = G.Idx(idx=1)
        data = ['alpha', 'beta', 'gamma']
        assert idx.forward(data) == idx.index(data)

    def test_forward_list(self):
        idx = G.Idx(idx=[0, 2])
        data = ['foo', 'bar', 'baz']
        assert idx.forward(data) == idx.index(data)



class _RangeStream(P.StreamProcess):
    """Yield the numbers 0‥(stop-1)."""

    def __init__(self, stop: int):
        self._stop = stop

    def stream(self) -> Iterator[int]:                    # noqa: D401
        for i in range(self._stop):
            yield i


class _AsyncRangeStream(P.AsyncStreamProcess):
    """Async variant of `_RangeStream`."""

    def __init__(self, stop: int):
        self._stop = stop

    async def astream(self) -> AsyncIterator[int]:        # noqa: D401
        for i in range(self._stop):
            yield i
            await asyncio.sleep(0)                        # co-operative step


class _BadStream(P.StreamProcess):
    """Returns a scalar instead of an iterator – should break."""

    def stream(self) -> int:                              # type: ignore[override]
        return 123                                        # not iterable


@pytest.mark.asyncio
class TestStream:
    """
    Sync `Stream.aforward` yields `Partial` objects with correct flags and
    memoises results.
    """

    async def test_first_item_partial_fields(self):
        stream = Stream(
            src=_RangeStream(2),
            args=SerialDict(data={}),
            is_async=False
        )
        part = await stream.aforward()
        assert part.dx == 0
        assert part.complete is False
        assert part.prev is None
        assert part.full == [0]

    async def test_second_call_returns_cached_partial(self):
        stream = Stream(
            src=_RangeStream(2),
            args=SerialDict(data={}),
            is_async=False
        )
        first = await stream.aforward()
        second = await stream.aforward()
        assert first == second                                  # cached
        assert second.full == [0]                               # unchanged

    async def test_zero_length_stream_is_immediately_complete(self):
        stream = Stream(
            src=_RangeStream(0),
            args=SerialDict(data={}),
            is_async=False
        )
        part = await stream.aforward()
        assert part.complete is True
        assert part.dx is None
        assert part.prev is None
        assert part.full == []

    async def test_bad_stream_raises_type_error(self):
        stream = Stream(
            src=_BadStream(),
            args=SerialDict(data={}),
            is_async=False
        )
        with pytest.raises(TypeError):
            await stream.aforward()


@pytest.mark.asyncio
class TestAsyncStream:
    """
    Async variant behaves identically when driven by an `AsyncStreamProcess`.
    """

    async def test_async_first_item(self):
        stream = Stream(
            src=_AsyncRangeStream(3),
            args=SerialDict(data={}),
            is_async=True
        )
        part = await stream.aforward()
        assert part.dx == 0
        assert part.complete is False

    async def test_async_immediate_complete(self):
        stream = Stream(
            src=_AsyncRangeStream(0),
            args=SerialDict(data={}),
            is_async=True
        )
        part = await stream.aforward()
        assert part.complete is True
        assert part.full == []

    async def test_async_cached_partial(self):
        stream = Stream(
            src=_AsyncRangeStream(1),
            args=SerialDict(data={}),
            is_async=True
        )
        first = await stream.aforward()
        second = await stream.aforward()
        assert first == second


@pytest.mark.asyncio
class TestProcNodeHelpers:
    """Covers has_partial(), eval_args() and get_incoming()."""

    async def test_has_partial_flags_incomplete(self):
        part = G.Partial(dx=1, complete=False)
        node = G.T(src=lambda **_: 0,
                   args=SerialDict(data={"p": part}),
                   is_async=False)
        assert node.has_partial() is True

    async def test_has_partial_ignores_completed(self):
        done = G.Partial(dx=1, complete=True, full=[1])
        node = G.T(src=lambda **_: 0,
                   args=SerialDict(data={"p": done}),
                   is_async=False)
        assert node.has_partial() is False

    async def test_eval_args_returns_concrete_values(self):
        v = G.Var(); v.val = 5
        part = G.Partial(dx=7, complete=False)
        node = G.T(src=lambda **_: 0,
                   args=SerialDict(data={"v": v, "p": part, "k": 9}),
                   is_async=False)
        out = node.eval_args()
        assert dict(out.items()) == {"v": 5, "p": 7, "k": 9}

    async def test_get_incoming_mixes_cached_and_probe(self):
        v1 = G.Var(); v1.val = 1          # cached
        v2 = G.Var()                      # supplied via probe-dict
        node = G.T(src=lambda **_: 0,
                   args=SerialDict(data={"a": v1, "b": v2}),
                   is_async=False)

        incoming = await node.get_incoming({v2: 42})
        assert dict(incoming.items()) == {"a": 1, "b": 42}


class TestWaitProcess:
    """incoming() and forward() semantics."""

    def _partial(self, complete: bool):
        return G.Partial(dx='x', complete=complete, full=['a', 'b'])

    def test_incoming_yields_single_node(self):
        t = G.Var(); wp = G.WaitProcess(t)
        assert list(wp.incoming()) == [t]

    def test_forward_waiting_on_incomplete(self):
        wp = G.WaitProcess(G.Var())
        assert wp.forward(self._partial(False)) is WAITING

    def test_forward_returns_full_on_complete(self):
        wp = G.WaitProcess(G.Var(), agg=list)
        out = wp.forward(self._partial(True))
        assert out == ['a', 'b']

    def test_forward_with_custom_agg(self):
        wp = G.WaitProcess(G.Var(), agg="".join)
        out = wp.forward(self._partial(True))
        assert out == "ab"

# # ---------------------------------------------------------------------------
# # Streamer hash behaviour
# # ---------------------------------------------------------------------------

class TestStreamerHash:
    """Streamer must be usable as a dict key."""

    def test_same_iter_same_hash(self):
        it = iter([1])
        s1, s2 = G.Streamer(stream=it), G.Streamer(stream=it)
        assert hash(s1) == hash(s2)

    def test_different_iter_diff_hash(self):
        assert hash(G.Streamer(stream=iter([1]))) != hash(G.Streamer(stream=iter([1])))

# # ---------------------------------------------------------------------------
# # Stream.__post_init__ async flag
# # ---------------------------------------------------------------------------

class TestStreamPostInit:
    def test_sync_src_sets_flag_false(self):
        st = G.Stream(src=_RangeStream(1),
                      args=SerialDict(data={}),
                      is_async=False)
        assert st._is_async is False

    def test_async_src_sets_flag_true(self):
        st = G.Stream(src=_AsyncRangeStream(1),
                      args=SerialDict(data={}),
                      is_async=True)
        assert st._is_async is True

# # ---------------------------------------------------------------------------
# # Factory helpers (t / async_t / stream / async_stream)
# # ---------------------------------------------------------------------------

async def _async_fn(**kwargs):  # simple async callable
    await asyncio.sleep(0)
    return kwargs.get("x", 0)

def _sync_fn(**kwargs):         # simple sync callable
    return kwargs.get("x", 0)

class TestFactoryHelpers:
    def test_t_creates_sync_T(self):
        node = G.t(_sync_fn, x=1)
        assert isinstance(node, G.T) and node.is_async is False

    def test_async_t_creates_async_T(self):
        node = G.async_t(_async_fn, x=1)
        assert isinstance(node, G.T) and node.is_async is True

    def test_stream_creates_sync_stream(self):
        st = G.stream                               # factory under test
        node = st(_RangeStream(0))
        assert isinstance(node, G.Stream) and node.is_async is False

    def test_async_stream_creates_async_stream(self):
        ast = G.async_stream
        node = ast(_AsyncRangeStream(0))
        assert isinstance(node, G.Stream) and node.is_async is True
