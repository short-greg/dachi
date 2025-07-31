
import asyncio
import pytest
from dachi.proc import _graph as G
from dachi.core import SerialDict
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


@pytest.mark.asyncio
class TestProcNodeHelpers:
    """Covers has_partial(), eval_args() and get_incoming()."""

    # async def test_has_partial_flags_incomplete(self):
    #     part = G.Partial(dx=1, complete=False)
    #     node = G.T(src=lambda **_: 0,
    #                args=SerialDict(data={"p": part}),
    #                is_async=False)
    #     assert node.has_partial() is True

    # async def test_has_partial_ignores_completed(self):
    #     done = G.Partial(dx=1, complete=True, full=[1])
    #     node = G.T(src=lambda **_: 0,
    #                args=SerialDict(data={"p": done}),
    #                is_async=False)
    #     assert node.has_partial() is False

    # async def test_eval_args_returns_concrete_values(self):
    #     v = G.Var(); v.val = 5
    #     part = G.Partial(dx=7, complete=False)
    #     node = G.T(src=lambda **_: 0,
    #                args=SerialDict(data={"v": v, "p": part, "k": 9}),
    #                is_async=False)
    #     out = node.eval_args()
    #     assert dict(out.items()) == {"v": 5, "p": 7, "k": 9}

    async def test_get_incoming_mixes_cached_and_probe(self):
        v1 = G.Var(); v1.val = 1          # cached
        v2 = G.Var()                      # supplied via probe-dict
        node = G.T(src=lambda **_: 0,
                   args=SerialDict(data={"a": v1, "b": v2}),
                   is_async=False)

        incoming = await node.get_incoming({v2: 42})
        assert dict(incoming.items()) == {"a": 1, "b": 42}



# tests/proc/test_dag.py
import asyncio
import pytest

from dachi.proc._graph import DAG, RefT
from dachi.core import ModuleDict, Attr
from dachi.proc import _process as P


# --------------------------------------------------------------------------- #
# Helper stub processes – minimal, no external deps                           #
# --------------------------------------------------------------------------- #

class _Const(P.Process):
    """Synchronous constant-returning Process."""
    def __init__(self, value):
        self.value = value
        self.calls = 0                      # for caching assertions

    def forward(self):
        self.calls += 1
        return self.value


class _Add(P.Process):
    """Adds two numbers; tracks call count for caching assertions."""
    def __init__(self):
        self.calls = 0

    def forward(self, a, b):
        self.calls += 1
        return a + b


class _AsyncConst(P.AsyncProcess):
    """Asynchronous constant-returning Process."""
    def __init__(self, value):
        self.value = value
        self.calls = 0

    async def aforward(self):
        self.calls += 1
        await asyncio.sleep(0)              # prove async path
        return self.value


# --------------------------------------------------------------------------- #
# A DAG subclass exposing an async method for the “string-node” pathway       #
# --------------------------------------------------------------------------- #

class _MethodDAG(DAG):
    async def five(self):
        return 5


# --------------------------------------------------------------------------- #
# Test-classes – one behaviour per test                                       #
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
class TestDAG:
    """`__post_init__` initialises empty containers."""

    async def test_defaults(self):

        dag = DAG()
        assert isinstance(dag._nodes, ModuleDict)
        assert dag._nodes._module_dict == {}
        assert isinstance(dag._args, Attr)
        assert dag._args.data == {}


    async def test_resolve_process_chain(self):
        dag = DAG()
        dag._nodes = ModuleDict(
            data={
                "a": _Const(1),
                "b": _Const(2),
                "sum": _Add(),
            }
        )
        dag._args.data = {
            "a": {},
            "b": {},
            "sum": {"a": RefT("a"), "b": RefT("b")},
        }

        by = {}
        out = await dag._sub("sum", by)
        assert out == 3
        assert by["sum"] == 3
        # call-count proves caching via *by*
        assert dag._nodes["sum"].calls == 1
        _ = await dag._sub("sum", by)
        assert dag._nodes["sum"].calls == 1

    async def test_resolve_async_process(self):
        dag = DAG()
        dag._nodes = ModuleDict(
            data={"x": _AsyncConst(7)}
        )
        dag._args.data = {"x": {}}
        assert await dag._sub("x", {}) == 7

    async def test_resolve_string_node(self):
        dag = _MethodDAG()
        dag._nodes = ModuleDict(data={"foo": "five"})
        dag._args.data = {"foo": {}}
        assert await dag._sub("foo", {}) == 5

    async def test_missing_node_raises_keyerror(self):
        dag = DAG()
        dag._nodes = ModuleDict(data={})
        dag._args.data = {}
        with pytest.raises(KeyError):
            await dag._sub("ghost", {})

    async def test_missing_method_raises_valueerror(self):
        dag = DAG()
        dag._nodes = ModuleDict(data={"foo": "bar"})
        dag._args.data = {"foo": {}}
        with pytest.raises(ValueError):
            await dag._sub("foo", {})

    async def test_circular_reference_recursion_error(self):
        dag = DAG()
        dag._nodes = ModuleDict(data={"a": _Const(0)})
        dag._args.data = {"a": {"loop": RefT("a")}}
        with pytest.raises(ExceptionGroup):
            await dag._sub("a", {})

    async def test_single_output_tuple(self):
        dag = DAG()
        dag._nodes = ModuleDict(
            data={
                "x": _Const(10),
            }
        )
        dag._args.data = {"x": {}}
        dag._outputs.data = ["x"]

        out = await dag.aforward()
        assert out == (10,)

    async def test_multiple_outputs_order_preserved(self):
        dag = DAG()
        dag._nodes = ModuleDict(
            data={
                "a": _Const(1),
                "b": _Const(2),
            }
        )
        dag._args.data = {"a": {}, "b": {}}
        dag._outputs.set(["b", "a"])          # reverse on purpose

        out = await dag.aforward()
        assert out == (2, 1)

    async def test_empty_outputs_returns_empty_tuple(self):
        dag = DAG()
        dag._outputs.set([])
        assert await dag.aforward() == ()

    async def test_probe_dict_short_circuits(self):
        c = _Const(99)
        dag = DAG()
        dag._nodes = ModuleDict(data={"x": c})
        dag._args.data = {"x": {}}
        dag._outputs.set(["x"])

        probe = {"x": 123}
        out = await dag.aforward(probe)
        assert out == (123,)
        # underlying process never called
        assert c.calls == 0



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
