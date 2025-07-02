import pytest
from dachi.proc import _process as p
from dachi.utils import WAITING, UNDEFINED
from dachi.proc import _graph as g
from dachi.proc import _graph as G


import asyncio
import types
import pytest

from dachi.proc import _graph as G
from dachi.proc._process import Partial  # use real Partial for realism
from dachi.core import SerialDict

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
        assert wrapped.src.node is node and wrapped.src.idx == 1

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


import pytest

from dachi.proc import _graph as G

# -------------------------------------------------
# Local helper stubs
# -------------------------------------------------
class _StubIncoming:
    """Mimics an upstream node whose *probe* method will be invoked by Var."""
    def __init__(self):
        self.probed = False

    def probe(self, by: dict):  # signature is irrelevant, only side‑effect matters
        self.probed = True


class _StubSrc:
    """Callable object that acts as the *source* for a Var instance.

    It returns a predetermined *ret* value and exposes an *incoming()* iterator
    over arbitrary stub nodes so that we can verify *probe* propagation.
    """

    def __init__(self, ret, incoming_nodes=None):
        self._ret = ret
        self._incoming_nodes = incoming_nodes or []

    # ------------------------------------------------------------------
    # Public protocol expected by Var.aforward
    # ------------------------------------------------------------------
    def incoming(self):
        for n in self._incoming_nodes:
            yield n

    def __call__(self, by):
        return self._ret


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




# tests/proc/test_t.py
import asyncio
import pytest

from dachi.proc._graph import (
    t,             # sync-task factory
    async_t,       # async-task factory
    Var,           # constant/input node
    UNDEFINED,     # sentinel for “no value yet”
    T
)

# ---------------------------------------------------------------------------
# Minimal dummy “processes” to drive T
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Tests for the concrete task-node `T`
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestT:
    async def test_post_init_defaults(self):
        proc = _SyncProc(value=42)
        node = t(p=proc)
        assert node.is_async is False
        assert node.val is UNDEFINED                    # starts undefined

    # async def test_aforward_sync_exec_and_cache(self):
    #     proc = _SyncProc(10)
    #     node = t(proc)
    #     probe = {}

    #     out1 = await node.aforward(probe)
    #     assert out1 == 10
    #     assert node.val == 10                           # cached
    #     assert probe[node] == 10

    #     # second call must *not* re-invoke the underlying process
    #     calls_before = proc.calls
    #     out2 = await node.aforward(probe)
    #     assert out2 == 10
    #     assert proc.calls == calls_before               # unchanged

    # async def test_aforward_probe_dict_short_circuit(self):
    #     proc = _SyncProc(999)
    #     node = t(proc)
    #     probe = {node: 123}                             # override

    #     result = await node.aforward(probe)
    #     assert result == 123
    #     assert proc.calls == 0                          # never executed
    #     assert node.val is UNDEFINED                    # cache untouched

    # async def test_aforward_async_exec(self):
    #     proc = _AsyncProc(7)
    #     node = async_t(proc)

    #     res = await node.aforward()
    #     assert res == 7
    #     assert node.val == 7
    #     assert proc.calls == 1

    # async def test_aforward_dependency_resolution(self):
    #     # Var supplies input ‘x’; process returns x + 1 (see _SyncProc logic)
    #     x_var = Var()
    #     x_var.val = 3                                   # make it concrete

    #     proc = _SyncProc(None)
    #     node = t(proc, x=x_var)                         # node(x) = x + 1

    #     res = await node.aforward()
    #     assert res == 4

    # async def test_aforward_undefined_result_error(self):
    #     class _BadProc:
    #         def __init__(self):
    #             self.calls = 0
    #         def __call__(self, **kwargs):
    #             self.calls += 1
    #             return UNDEFINED

    #     node = t(_BadProc())
    #     with pytest.raises(RuntimeError):
    #         await node.aforward()

    #     # value should remain undefined after failure
    #     assert node.val is UNDEFINED

    # async def test_aforward_exception_bubbles_and_no_cache(self):
    #     class _BoomProc:
    #         def __init__(self):
    #             self.calls = 0
    #         def __call__(self, **kwargs):
    #             self.calls += 1
    #             raise ValueError("boom")

    #     proc = _BoomProc()
    #     node = t(proc)

    #     with pytest.raises(ValueError):
    #         await node.aforward()

    #     assert node.val is UNDEFINED
    #     assert proc.calls == 1

# # -----------------------------------------------------------------------------
# # Helper for T tests – call counters
# # -----------------------------------------------------------------------------
# class _CallCounter:
#     def __init__(self, *, async_=False, ret):
#         self.calls = 0
#         self.ret = ret
#         self.async_ = async_
#         if async_:
#             async def _af(*args, **kwargs):
#                 self.calls += 1
#                 await asyncio.sleep(0)
#                 return self.ret
#             self.fn = _af
#         else:
#             def _f(*args, **kwargs):
#                 self.calls += 1
#                 return self.ret
#             self.fn = _f


# # -----------------------------------------------------------------------------
# # T tests
# # -----------------------------------------------------------------------------
# class TestT:
#     """Exhaustive tests for the task‑node wrapper *T*."""

#     # ------------- fast‑return: cached --------------------------------
#     @pytest.mark.asyncio
#     async def test_aforward_returns_cached_val(self):
#         tnode = G.T(src=lambda: 1, args=SerialDict({}), is_async=False)
#         tnode.val = 42
#         out = await tnode.aforward()
#         assert out == 42

#     # ------------- fast‑return: probe‑dict ----------------------------
#     @pytest.mark.asyncio
#     async def test_aforward_uses_probe_dict(self):
#         tnode = G.T(src=lambda: 0, args=SerialDict({}), is_async=False)
#         probe = {tnode: 99}
#         out = await tnode.aforward(probe)
#         assert out == 99

#     # ------------- sync execution ------------------------------------
#     @pytest.mark.asyncio
#     async def test_aforward_executes_sync_src_once_and_caches(self):
#         counter = _CallCounter(async_=False, ret=7)
#         tnode = G.T(src=counter.fn, args=SerialDict({}), is_async=False)
#         probe: dict = {}
#         result1 = await tnode.aforward(probe)
#         result2 = await tnode.aforward(probe)
#         assert result1 == result2 == 7
#         assert counter.calls == 1  # memoisation worked
#         assert probe[tnode] == 7 and tnode.val == 7

#     # ------------- async execution -----------------------------------
#     @pytest.mark.asyncio
#     async def test_aforward_executes_async_src(self):
#         counter = _CallCounter(async_=True, ret=13)
#         tnode = G.T(src=counter.fn, args=SerialDict({}), is_async=True)
#         res = await tnode.aforward()
#         assert res == 13 and counter.calls == 1

#     # ------------- UNDEFINED result error ----------------------------
#     @pytest.mark.asyncio
#     async def test_aforward_raises_when_src_returns_undefined(self):
#         bad_src = _CallCounter(async_=False, ret=G.UNDEFINED)
#         tnode = G.T(src=bad_src.fn, args=SerialDict({}), is_async=False)
#         with pytest.raises(RuntimeError):
#             await tnode.aforward()




# class TestT:

#     def test_src_returns_src(self):
#         p = MyProcess()
#         src = g.ModSrc(p, (0, 1))
#         t = g.T(src=src)
#         assert t.src == src

#     def test_g_returns_src(self):
#         p = MyProcess()
#         src = g.ModSrc(p, (0, 1))
#         t = g.T(src=src)
#         assert t.val == UNDEFINED

#     def test_label_updates_the_label(self):
#         p = MyProcess()
#         src = g.ModSrc(p, (0, 1))
#         t = g.T(src=src)
#         t = t.label(name='x')
#         assert t.name == 'x'

#     def test_is_undefined_returns_true(self):
#         p = MyProcess()
#         src = g.ModSrc(p, (0, 1))
#         t = g.T(src=src)
#         t = t.label(name='x')
#         assert t.is_undefined()

#     def test_get_item_returns_first_item(self):
#         p = MyProcess()
#         src = g.ModSrc(p, (0, 1))
#         t = g.T([1, 2], src=src)
#         t = t.label(name='x')
#         assert t[0].val == 1

#     def test_get_item_returns_first_index_source(self):
#         p = MyProcess()
#         src = g.ModSrc(p, (0, 1))
#         t = g.T(UNDEFINED, src=src)
#         t = t.label(name='x')
#         assert t.val is UNDEFINED

#     def test_detach_has_no_source(self):
#         p = MyProcess()
#         src = g.ModSrc(p, (0, 1))
#         t = g.T([1, 2], src=src)
#         t = t.detach()
#         assert t.src is None
    
#     def test_async_module_works_with_async(self):
#         module = p.AsyncParallel(
#             Append('x')
#         )
#         src = g.ModSrc.create(module, p.Chunk(['hi']))
#         res = src()
#         assert res == ['hix']
    
#     def test_async_module_works_with_multiple_async(self):
#         module = p.AsyncParallel([
#             Append('x'),
#             Append('y')
#         ])
#         src = g.ModSrc.create(module, p.Chunk(['hi', 'hi']))
#         res = src()
#         assert res == ['hix', 'hiy']

#     def test_t_has_correct_annotation(self):
#         p = MyProcess()
#         src = g.ModSrc(p, (0, 1))
#         annotation = 'A simple module.'
#         t = g.T(src=src, annotation=annotation)
#         assert t.annotation == annotation


# class TestVar:

#     def test_var_returns_value(self):
#         var = g.Var(1)
#         assert var() == 1

#     def test_var_returns_value_with_factory(self):
#         var = g.Var(default_factory=lambda: 3)
#         assert var() == 3

#     def test_var_has_no_incoming(self):
#         var = g.Var(1)
#         assert len(list(var.incoming())) == 0

#     def test_var_raises_error_if_no_default_or_factory(self):
#         with pytest.raises(RuntimeError, match="Either the default value or default factory must be defined"):
#             g.Var()

#     def test_var_with_default_overrides_factory(self):
#         var = g.Var(default=5, default_factory=lambda: 10)
#         assert var() == 5

#     def test_var_with_none_default_and_factory(self):
#         var = g.Var(default=None, default_factory=lambda: "fallback")
#         assert var() == None  # Explicitly testing None as a valid default

#     def test_var_with_complex_factory(self):
#         var = g.Var(default_factory=lambda: [i for i in range(5)])
#         assert var() == [0, 1, 2, 3, 4]

#     def test_var_with_callable_default(self):
#         var = g.Var(default_factory=lambda: "callable_default")
#         assert callable(var.default_factory)
#         assert var() == "callable_default"

#     def test_var_with_mutable_default(self):
#         default_list = [1, 2, 3]
#         var = g.Var(default=default_list)
#         assert var() == default_list
#         default_list.append(4)
#         assert var() == [1, 2, 3, 4]  # Ensure mutable default is reflected

#     def test_var_with_factory_returning_mutable(self):
#         var = g.Var(default_factory=lambda: {"key": "value"})
#         result = var()
#         assert result == {"key": "value"}
#         result["key"] = "new_value"
#         assert var() == {"key": "value"}  # Factory should return a new instance each time

#     def test_var_forward_with_empty_by(self):
#         var = g.Var(default=42)
#         assert var.forward(by={}) == 42

#     def test_var_forward_ignores_by(self):
#         var = g.Var(default=42)
#         assert var.forward(by={"irrelevant": "data"}) == 42

#     def test_var_with_large_default_value(self):
#         large_value = "x" * 10**6
#         var = g.Var(default=large_value)
#         assert var() == large_value

#     def test_var_with_nested_factory(self):
#         var = g.Var(default_factory=lambda: {"nested": [1, 2, {"key": "value"}]})
#         assert var() == {"nested": [1, 2, {"key": "value"}]}

#     def test_var_with_default_as_falsey_value(self):
#         var = g.Var(default=0)
#         assert var() == 0
#         var = g.Var(default="")
#         assert var() == ""
#         var = g.Var(default=False)
#         assert var() == False

#     def test_var_returns_value(self):

#         var = g.Var(1)
#         assert var() == 1

#     def test_var_returns_value_with_factory(self):

#         var = g.Var(default_factory=lambda: 3)
#         assert var() == 3

#     def test_var_has_no_incoming(self):

#         var = g.Var(1)
#         assert len(list(var.incoming())) == 0


# class TestIdxSrc:

#     def test_idx_returns_value(self):
#         src = g.Var([0, 1])
#         t = g.T(src=src)
#         idx = g.IdxSrc(t, 0)
        
#         assert idx() == 0

#     def test_idx_incoming_returns_t(self):
#         src = g.Var([0, 1])
#         t = g.T(src=src)
#         idx = g.IdxSrc(t, 0)
        
#         incoming = list(idx.incoming())[0]
#         assert incoming is t

#     def test_probe_incoming_returns_val(self):
#         src = g.Var([0, 1])
#         idx = g.T(src=src)[0]
#         val = idx.probe()
#         assert val == 0



# class TestWaitSrc:

#     def test_returns_waiting(self):
#         writer = WriteOut('hi')
#         t = g.stream_link(writer, g.T('bye'))
#         src = g.WaitSrc(t)
#         assert src() == WAITING

#     def test_returns_value_if_finished(self):
#         writer = WriteOut('hi')
#         t = g.stream_link(writer, g.T(''))
#         src = g.WaitSrc(t, lambda x: ''.join(x))
#         src()
#         src()
#         assert src() == 'hi'

#     def test_returns_waiting_for_partial_incomplete(self):
#         writer = WriteOut('hi')
#         t = g.stream_link(writer, g.T('xyz'))
#         partial = t.val()
#         src = g.WaitSrc(t)
#         assert src() == WAITING

#     def test_aggregates_partial_when_complete(self):
#         writer = WriteOut('hi')
#         t = g.stream_link(writer, g.T('xyz'))
#         # partial = t.val()
#         # partial.complete = True
#         # partial.full = 'xyz_full'
#         src = g.WaitSrc(t, lambda x: ''.join(
#             xi.upper() for xi in x
#         ))
#         src()
#         src()
#         src()
#         src()
#         src()
#         res = src()
    
#         assert res == 'XYZHI'

#     # def test_returns_waiting_for_streamer_incomplete(self):
#     #     k = g.Var('k')
#     #     src = g.StreamSrc(Append('s'), g.TArgs(k))
#     #     streamer = src()
#     #     wait_src = g.WaitSrc(streamer)
#     #     assert wait_src() == WAITING

#     # def test_aggregates_streamer_when_complete(self):
#     #     k = g.Var('k')
#     #     src = g.StreamSrc(Append('s'), g.TArgs(k))
#     #     streamer = src()
#     #     # streamer.complete = True
#     #     # streamer.output = g.Partial(full='ks_full')
#     #     wait_src = g.WaitSrc(streamer, lambda x: x[::-1])
#     #     assert wait_src() == 'lluf_sk'

#     def test_handles_non_partial_non_streamer_values(self):
#         t = g.T('static_value')
#         src = g.WaitSrc(t)
#         assert src() == 'static_value'

#     def test_raises_error_for_invalid_incoming_type(self):
#         with pytest.raises(AttributeError):
#             invalid_incoming = 123  # Not a valid transmission
#             src = g.WaitSrc(invalid_incoming)
#             src()

#     def test_handles_empty_aggregation_function(self):
#         writer = WriteOut('hi')
#         t = g.stream_link(writer, g.T('xyz'))
#         # partial = t.val()
#         # partial.complete = True
#         # partial.full = 'xyz_full'
#         src = g.WaitSrc(t, lambda x: ''.join(x))
#         src()
#         src()
#         src()
#         src()
#         src()
#         assert src() == 'xyzhi'

#     def test_handles_none_as_incoming(self):
#         with pytest.raises(AttributeError):
#             src = g.WaitSrc(None)
#             src()

#     def test_forward_with_empty_by(self):
#         writer = WriteOut('hi')
#         t = g.stream_link(writer, g.T('xyz'))
#         src = g.WaitSrc(t)
#         assert src.forward(by={}) == WAITING

#     def test_forward_with_valid_by(self):
#         writer = WriteOut('hi')
#         t = g.stream_link(writer, g.T())
#         src = g.WaitSrc(t)
#         assert src.forward(by={t: 'xyz'}) == "xyz"

#     def test_returns_waiting(self):

#         writer = WriteOut('hi')
#         t = g.stream_link(writer, g.T('bye'))
#         src = g.WaitSrc(t)
#         assert src() == WAITING

#     def test_returns_value_if_finished(self):

#         writer = WriteOut('hi')
#         t = g.stream_link(writer, g.T(''))
#         src = g.WaitSrc(t, lambda x: ''.join(x))
#         src()
#         src()
#         assert src() == 'hi'


# class TestLink:

#     def test_module_outputs_t_with_correct_value(self):

#         append = Append('_t')
#         t = g.link(append, ('x'))
#         assert t.val == 'x_t'
#         assert t.src.mod is append

#     def test_chaining_appends_produces_correct_value(self):

#         append = Append('_t')
#         t = g.link(append, 'x')
#         t = g.link(append, t)
#         assert t.val == 'x_t_t'

#     def test_it_is_undefined_if_val_not_defined(self):

#         append = Append('_t')
#         t = g.T()
#         t = g.link(append, t)
#         t = g.link(append, t)
#         assert t.is_undefined() is True

#     def test_it_probes_the_input(self):

#         append = Append('_t')
#         t1 = g.T()
#         t = g.link(append, t1)
#         t = g.link(append, t)
#         res = t.probe(by={t1: 'x'})
#         assert res == 'x_t_t'

#     def test_t_probes_UNDEFINED_if_not_defined(self):

#         append = Append('_t')
#         t1 = g.T()
#         t = g.link(append, t1)
#         t = g.link(append, t)
#         with pytest.raises(RuntimeError):
#             t.probe({})


# class TestStreamSrc:

#     def test_stream_src_returns_streamer_instance(self):
#         """Test that StreamSrc returns a Streamer instance."""
#         k = g.Var('k')
#         src = g.StreamSrc(Append('s'), g.NodeArgs(k))
#         val = src()
#         assert isinstance(val, g.Streamer)

#     def test_stream_src_with_empty_args(self):
#         """Test StreamSrc with empty arguments."""
#         k = g.Var([])
#         src = g.StreamSrc(Append('s'), g.NodeArgs(k))
#         val = src()
#         assert isinstance(val, g.Streamer)

#     def test_stream_src_with_invalid_module(self):
#         """Test StreamSrc with an invalid module."""
#         k = g.Var('k')
#         with pytest.raises(AttributeError):
#             src = g.StreamSrc(None, g.NodeArgs(k))
#             src()

#     def test_stream_src_with_none_args(self):
#         """Test StreamSrc with None as arguments."""
#         with pytest.raises(TypeError):
#             src = g.StreamSrc(Append('s'), None)
#             src()

#     def test_stream_src_with_multiple_args(self):
#         """Test StreamSrc with multiple arguments."""
#         k1 = g.Var('k1')
#         src = g.StreamSrc(Append('s'), g.NodeArgs(k1))
#         val = src()
#         assert isinstance(val, g.Streamer)

#     def test_stream_src_forward_with_empty_by(self):
#         """Test forward method with an empty 'by' dictionary."""
#         k = g.Var('k')
#         src = g.StreamSrc(Append('s'), g.NodeArgs(k))
#         streamer = src.forward(by={})
#         assert isinstance(streamer, g.Streamer)

#     def test_stream_src_forward_with_valid_by(self):
#         """Test forward method with a valid 'by' dictionary."""
#         k = g.Var('k')
#         src = g.StreamSrc(Append('s'), g.NodeArgs(k))
#         streamer = src.forward(by={k: 'test'})
#         assert isinstance(streamer, g.Streamer)

#     def test_stream_src_forward_with_invalid_by(self):
#         """Test forward method with an invalid 'by' dictionary."""
#         k = g.Var('k')
#         src = g.StreamSrc(Append('s'), g.NodeArgs(k))
#         with pytest.raises(TypeError):
#             src.forward(by="invalid")

#     def test_stream_src_call_with_empty_by(self):
#         """Test __call__ method with an empty 'by' dictionary."""
#         k = g.Var('k')
#         src = g.StreamSrc(Append('s'), g.NodeArgs(k))
#         streamer = src(by={})
#         assert isinstance(streamer, g.Streamer)

#     def test_stream_src_call_with_valid_by(self):
#         """Test __call__ method with a valid 'by' dictionary."""
#         k = g.Var('k')
#         src = g.StreamSrc(Append('s'), g.NodeArgs(k))
#         streamer = src(by={k: 'test'})
#         assert isinstance(streamer, g.Streamer)

#     def test_stream_src_incoming_yields_correct_values(self):
#         """Test that incoming method yields correct values."""
#         k = g.Var('k')
#         src = g.StreamSrc(Append('s'), g.NodeArgs(k))
#         incoming = list(src.incoming())
#         assert len(incoming) == 1
#         assert incoming[0] is k

#     def test_stream_src_with_large_input(self):
#         """Test StreamSrc with a large input."""
#         large_input = g.Var('x' * 10**6)
#         src = g.StreamSrc(Append('s'), g.NodeArgs(large_input))
#         val = src()
#         assert isinstance(val, g.Streamer)

#     def test_stream_src_with_callable_args(self):
#         """Test StreamSrc with callable arguments."""
#         k = g.Var(lambda: 'dynamic_value')
#         src = g.StreamSrc(Append('s'), g.NodeArgs(k))
#         val = src()
#         assert isinstance(val, g.Streamer)

#     def test_stream_src_with_mutable_args(self):
#         """Test StreamSrc with mutable arguments."""
#         mutable_arg = g.Var([1, 2, 3])
#         src = g.StreamSrc(Append('s'), g.NodeArgs(mutable_arg))
#         val = src()
#         assert isinstance(val, g.Streamer)
#         mutable_arg().append(4)
#         assert mutable_arg() == [1, 2, 3, 4]

#     def test_stream_src_with_no_module_stream_method(self):
#         """Test StreamSrc with a module that lacks a 'stream' method."""
#         class InvalidModule:
#             pass

#         k = g.Var('k')
#         with pytest.raises(AttributeError):
#             src = g.StreamSrc(InvalidModule(), g.NodeArgs(k))
#             src()

#     def test_stream_src_returns_a_streamer(self):
#         k = g.Var('k')
#         src = g.StreamSrc(
#             Append('s'), g.NodeArgs(k)
#         )
#         val = src()
        
#         assert isinstance(val, g.Streamer)

#     def test_streamer_returns_value(self):
#         k = g.Var('k')
#         src = g.StreamSrc(
#             Append('s'), g.NodeArgs(k)
#         )
#         val = src()
#         res = val()
        
#         assert res.dx == 'ks'


# class TestStreamLink:

#     def test_call_returns_t_with_streamer(self):

#         writer = WriteOut('hi')
#         t = g.stream_link(writer, g.T('xyz'))
#         partial = t.val()
#         assert partial.dx == 'x'

#     def test_call_returns_undefined_if_t_undefined(self):

#         writer = WriteOut('hi')
#         t = g.stream_link(writer, g.T())
#         assert t.is_undefined()

# # # #     # TODO* make this return partial
#     def test_chained_after_stream_appends(self):

#         writer = WriteOut('hi')
#         append = Append('_t')
#         t = g.stream_link(writer, g.T('xyz'))
#         t = g.link(append, t)
#         assert t.val.dx == 'x_t'

#     def test_stream_completes_the_stream(self):
#         writer = WriteOut('hi')
#         append = Append('_t')

#         for t in g.stream(writer, g.T('xyz')):
#             t = g.link(append, t)
        
#         assert t.val.dx == 'i_t'


# class TestWait:

#     def test_wait_results_in_waiting(self):

#         writer = WriteOut('hi')
#         t = g.stream_link(writer, g.T('xyz'))
#         t = g.wait(t)

#         assert t.val is WAITING

#     def test_wait(self):

#         append = Append('_t')
#         t = g.link(append, g.T('xyz'))
#         t = g.wait(t)

#         assert t.val == 'xyz_t'


# class TestParallelLink:

#     def test_module_outputs_t_with_correct_value(self):

#         append = Append('_t')
#         t = g.link(append, ('x'))
#         assert t.val == 'x_t'
#         assert t.src.mod is append

#     def test_chaining_appends_produces_correct_value(self):

#         append = Append('_t')
#         t = g.link(append, 'x')
#         t = g.link(append, t)
#         assert t.val == 'x_t_t'
