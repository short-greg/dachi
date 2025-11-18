
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
        v = G.V()
        v.val = 123  # simulate previously‑computed value

        assert await v.aforward() == 123

    # "by" dict override path
    @pytest.mark.asyncio
    async def test_aforward_uses_by_mapping(self):
        """If the probe dictionary carries an entry for *self* that is **not** a
        :class:`Partial`, that value should short‑circuit evaluation."""
        v = G.V()

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
        v = G.V()
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
        v = G.V(val=2)

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
        v = G.V()

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
        v = G.V()

        incoming = list(v.incoming())
        assert len(incoming) == 0


from dachi.proc._graph import (
    t,             # sync-task factory
    async_t,       # async-task factory
    V,           # constant/input node
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
        x_var = V()
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
        v1 = G.V(); v1.val = 1          # cached
        v2 = G.V()                      # supplied via probe-dict
        node = G.T(src=lambda **_: 0,
                   args=SerialDict(data={"a": v1, "b": v2}),
                   is_async=False)

        incoming = await node.get_incoming({v2: 42})
        assert dict(incoming.items()) == {"a": 1, "b": 42}



# tests/proc/test_dag.py
import asyncio
import pytest

from dachi.proc._graph import DataFlow, Ref
from dachi.core import ModuleDict, Runtime
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

class _MethodDAG(DataFlow):
    async def five(self):
        return 5


# --------------------------------------------------------------------------- #
# Test-classes – one behaviour per test                                       #
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
class TestDAG:
    """`__post_init__` initialises empty containers."""

    async def test_defaults(self):

        dag = DataFlow()
        assert isinstance(dag.nodes, ModuleDict)
        assert dag.nodes._module_dict == {}
        assert isinstance(dag._args, Runtime)
        assert dag._args.data == {}


    async def test_resolve_process_chain(self):
        dag = DataFlow()
        const_a = _Const(1)
        const_b = _Const(2)
        add_proc = _Add()

        dag.link("a", const_a)
        dag.link("b", const_b)
        dag.link("sum", add_proc, a=Ref("a"), b=Ref("b"))

        by = {}
        out = await dag._sub("sum", by)
        assert out == 3
        assert by["sum"] == 3
        # call-count proves caching via *by*
        assert add_proc.calls == 1
        _ = await dag._sub("sum", by)
        assert add_proc.calls == 1

    async def test_resolve_async_process(self):
        dag = DataFlow()
        dag.link("x", _AsyncConst(7))
        assert await dag._sub("x", {}) == 7

    async def test_resolve_string_node(self):
        dag = _MethodDAG()
        dag.link("foo", "five")
        assert await dag._sub("foo", {}) == 5

    async def test_missing_node_raises_keyerror(self):
        dag = DataFlow()
        with pytest.raises(KeyError):
            await dag._sub("ghost", {})

    async def test_missing_method_raises_valueerror(self):
        dag = DataFlow()
        dag.link("foo", "bar")
        with pytest.raises(ValueError):
            await dag._sub("foo", {})

    async def test_single_output_tuple(self):
        dag = DataFlow()
        dag.link("x", _Const(10))
        dag.set_out(["x"])

        out = await dag.aforward()
        assert out == (10,)

    async def test_multiple_outputs_order_preserved(self):
        dag = DataFlow()
        dag.link("a", _Const(1))
        dag.link("b", _Const(2))
        dag.set_out(["b", "a"])          # reverse on purpose

        out = await dag.aforward()
        assert out == (2, 1)

    async def test_empty_outputs_returns_empty_tuple(self):
        dag = DataFlow()
        dag.set_out([])
        assert await dag.aforward() == ()

    async def test_probe_dict_short_circuits(self):
        c = _Const(99)
        dag = DataFlow()
        dag.link("x", c)
        dag.set_out(["x"])

        probe = {"x": 123}
        out = await dag.aforward(probe)
        assert out == (123,)
        # underlying process never called
        assert c.calls == 0

    async def test_link_prevents_duplicate_names(self):
        """link() should raise ValueError when adding a node with a duplicate name"""
        dag = DataFlow()
        dag.link('node1', _Const(1))
        with pytest.raises(ValueError, match="already exists"):
            dag.link('node1', _Const(2))

    async def test_circular_reference_detection_stateless(self):
        """visited set should not leak between calls to _sub()"""
        dag = DataFlow()
        dag.link("a", _Const(1))

        await dag._sub("a", {})

        await dag._sub("a", {})

    async def test_to_node_graph_handles_unsupported_types(self):
        """to_node_graph() should raise ValueError for unsupported node types"""
        dag = DataFlow()
        dag.link("bad", "string_node")

        with pytest.raises(ValueError, match="Node must be"):
            dag.to_node_graph()


@pytest.mark.asyncio
class TestDAGLink:
    """Tests for DAG.link() method"""

    async def test_link_adds_process_node(self):
        """link() should add a Process node and return RefT"""
        dag = DataFlow()
        proc = _Const(42)
        ref = dag.link('test', proc)

        assert isinstance(ref, Ref)
        assert ref.name == 'test'
        assert 'test' in dag.nodes
        assert dag.nodes['test'] is proc

    async def test_link_adds_asyncprocess_node(self):
        """link() should add an AsyncProcess node"""
        dag = DataFlow()
        proc = _AsyncConst(7)
        ref = dag.link('async_test', proc)

        assert 'async_test' in dag.nodes
        assert dag.nodes['async_test'] is proc

    async def test_link_with_reft_arguments(self):
        """link() should store RefT arguments correctly"""
        dag = DataFlow()
        inp_ref = dag.add_inp('input', val=5)
        proc_ref = dag.link('proc', _Const(10), x=inp_ref)

        assert dag._args.data['proc']['x'] is inp_ref

    async def test_link_with_mixed_arguments(self):
        """link() should handle both RefT and literal arguments"""
        dag = DataFlow()
        inp_ref = dag.add_inp('input', val=5)
        proc_ref = dag.link('proc', _Add(), a=inp_ref, b=10)

        assert dag._args.data['proc']['a'] is inp_ref
        assert dag._args.data['proc']['b'] == 10

    async def test_link_returns_reft(self):
        """link() should return a RefT with the correct name"""
        dag = DataFlow()
        ref = dag.link('test', _Const(1))

        assert isinstance(ref, Ref)
        assert ref.name == 'test'


@pytest.mark.asyncio
class TestDAGAddInp:
    """Tests for DAG.add_inp() method"""

    async def test_add_inp_creates_var_node(self):
        """add_inp() should create a Var node with the given value"""
        dag = DataFlow()
        ref = dag.add_inp('input', val=42)

        assert isinstance(ref, Ref)
        assert ref.name == 'input'
        assert 'input' in dag.nodes
        assert isinstance(dag.nodes['input'], G.V)
        assert dag.nodes['input'].val == 42

    async def test_add_inp_with_undefined(self):
        """add_inp() should handle UNDEFINED values"""
        dag = DataFlow()
        ref = dag.add_inp('undef', val=UNDEFINED)

        assert dag.nodes['undef'].val is UNDEFINED

    async def test_add_inp_returns_reft(self):
        """add_inp() should return a RefT reference"""
        dag = DataFlow()
        ref = dag.add_inp('test', val=5)

        assert isinstance(ref, Ref)
        assert ref.name == 'test'

    async def test_add_inp_multiple_inputs(self):
        """add_inp() should allow adding multiple input nodes"""
        dag = DataFlow()
        ref1 = dag.add_inp('input1', val=1)
        ref2 = dag.add_inp('input2', val=2)
        ref3 = dag.add_inp('input3', val=3)

        assert 'input1' in dag.nodes
        assert 'input2' in dag.nodes
        assert 'input3' in dag.nodes
        assert dag.nodes['input1'].val == 1
        assert dag.nodes['input2'].val == 2
        assert dag.nodes['input3'].val == 3

    async def test_add_inp_prevents_duplicates(self):
        """add_inp() should raise ValueError for duplicate names"""
        dag = DataFlow()
        dag.add_inp('input', val=1)

        with pytest.raises(ValueError, match="already exists"):
            dag.add_inp('input', val=2)


@pytest.mark.asyncio
class TestDAGSetOut:
    """Tests for DAG.set_out() method"""

    async def test_set_out_with_list(self):
        """set_out() should accept a list of output names"""
        dag = DataFlow()
        dag.link('a', _Const(1))
        dag.link('b', _Const(2))
        dag.set_out(['a', 'b'])

        assert dag.outputs == ['a', 'b']

    async def test_set_out_with_string(self):
        """set_out() should accept a single string output"""
        dag = DataFlow()
        dag.link('single', _Const(42))
        dag.set_out('single')

        assert dag.outputs == 'single'

    async def test_set_out_updates_outputs(self):
        """set_out() should allow updating outputs multiple times"""
        dag = DataFlow()
        dag.link('a', _Const(1))
        dag.link('b', _Const(2))

        dag.set_out(['a'])
        assert dag.outputs == ['a']

        dag.set_out(['b'])
        assert dag.outputs == ['b']

    async def test_set_out_validates_node_exists(self):
        """set_out() should raise ValueError if node doesn't exist"""
        dag = DataFlow()
        dag.link('a', _Const(1))

        with pytest.raises(ValueError, match="does not exist"):
            dag.set_out('nonexistent')

        with pytest.raises(ValueError, match="does not exist"):
            dag.set_out(['a', 'nonexistent'])


@pytest.mark.asyncio
class TestDAGSub:
    """Tests for DAG.sub() method"""

    async def test_sub_creates_independent_dag(self):
        """sub() should create an independent DAG instance"""
        dag = DataFlow()
        dag.link('a', _Const(1))
        dag.link('b', _Const(2))

        sub = dag.sub(outputs=['a'], by={})

        sub.link('c', _Const(3))
        assert 'c' not in dag.nodes
        assert 'c' in sub.nodes

    async def test_sub_includes_specified_nodes(self):
        """sub() should include only the specified output nodes"""
        dag = DataFlow()
        dag.link('a', _Const(1))
        dag.link('b', _Const(2))

        sub = dag.sub(outputs=['a'], by={})

        assert 'a' in sub.nodes
        assert 'b' not in sub.nodes

    async def test_sub_with_invalid_node_raises(self):
        """sub() should raise ValueError for non-existent nodes"""
        dag = DataFlow()
        dag.link('a', _Const(1))

        with pytest.raises(ValueError, match="does not exist"):
            dag.sub(outputs=['nonexistent'], by={})

    async def test_sub_preserves_args(self):
        """sub() should preserve node arguments"""
        dag = DataFlow()
        inp_ref = dag.add_inp('input', val=5)
        dag.link('proc', _Add(), a=inp_ref, b=10)

        sub = dag.sub(outputs=['proc'], by={})

        assert 'proc' in sub._args.data
        assert sub._args.data['proc']['a'] is inp_ref
        assert sub._args.data['proc']['b'] == 10

    async def test_sub_sets_outputs(self):
        """sub() should set the outputs to the specified nodes"""
        dag = DataFlow()
        dag.link('a', _Const(1))
        dag.link('b', _Const(2))

        sub = dag.sub(outputs=['a', 'b'], by={})

        assert sub.outputs == ['a', 'b']


@pytest.mark.asyncio
class TestDAGReplace:
    """Tests for DAG.replace() method"""

    async def test_replace_updates_node(self):
        """replace() should update a node and affect execution"""
        dag = DataFlow()
        dag.link('a', _Const(1))
        dag.set_out('a')

        result1 = await dag.aforward()
        assert result1 == 1

        dag.replace('a', _Const(42))
        result2 = await dag.aforward()
        assert result2 == 42

    async def test_replace_preserves_connections(self):
        """replace() should preserve connections to other nodes"""
        dag = DataFlow()
        dag.link('a', _Const(5))
        dag.link('b', _Add(), a=Ref('a'), b=10)
        dag.set_out('b')

        dag.replace('a', _Const(20))
        result = await dag.aforward()
        assert result == 30

    async def test_replace_nonexistent_raises(self):
        """replace() should raise ValueError for non-existent nodes"""
        dag = DataFlow()

        with pytest.raises(ValueError, match="does not exist"):
            dag.replace('nonexistent', _Const(1))

    async def test_replace_affects_execution(self):
        """replace() should immediately affect subsequent executions"""
        dag = DataFlow()
        dag.link('x', _Const(10))
        dag.set_out('x')

        await dag.aforward()

        dag.replace('x', _Const(99))
        result = await dag.aforward()
        assert result == 99


@pytest.mark.asyncio
class TestDAGOutOverride:
    """Tests for out_override parameter in aforward()"""

    async def test_out_override_changes_outputs(self):
        """out_override should override the default outputs"""
        dag = DataFlow()
        dag.link('a', _Const(1))
        dag.link('b', _Const(2))
        dag.set_out(['a'])

        result = await dag.aforward(out_override=['b'])
        assert result == (2,)

    async def test_out_override_with_string(self):
        """out_override should accept a single string"""
        dag = DataFlow()
        dag.link('a', _Const(10))
        dag.link('b', _Const(20))
        dag.set_out(['a'])

        result = await dag.aforward(out_override='b')
        assert result == 20

    async def test_out_override_with_list(self):
        """out_override should accept a list of outputs"""
        dag = DataFlow()
        dag.link('a', _Const(1))
        dag.link('b', _Const(2))
        dag.link('c', _Const(3))
        dag.set_out(['a'])

        result = await dag.aforward(out_override=['b', 'c'])
        assert result == (2, 3)

    async def test_out_override_doesnt_modify_dag(self):
        """out_override should not modify the DAG's outputs"""
        dag = DataFlow()
        dag.link('a', _Const(1))
        dag.link('b', _Const(2))
        dag.set_out(['a'])

        await dag.aforward(out_override=['b'])

        assert dag.outputs == ['a']

    async def test_out_override_none_uses_default(self):
        """out_override=None should use the default outputs"""
        dag = DataFlow()
        dag.link('a', _Const(5))
        dag.set_out(['a'])

        result = await dag.aforward(out_override=None)
        assert result == (5,)


@pytest.mark.asyncio
class TestDAGOutputTypes:
    """Tests for output type handling in aforward()"""

    async def test_string_output_returns_single_value(self):
        """String output should return a single value, not a tuple"""
        dag = DataFlow()
        dag.link('a', _Const(42))
        dag.set_out('a')

        result = await dag.aforward()
        assert result == 42
        assert not isinstance(result, tuple)

    async def test_list_output_returns_tuple(self):
        """List output should return a tuple"""
        dag = DataFlow()
        dag.link('a', _Const(1))
        dag.link('b', _Const(2))
        dag.set_out(['a', 'b'])

        result = await dag.aforward()
        assert result == (1, 2)
        assert isinstance(result, tuple)

    async def test_empty_output_returns_none(self):
        """Empty or None output should return None"""
        dag = DataFlow()
        dag.link('a', _Const(1))

        result = await dag.aforward()
        assert result is None


@pytest.mark.asyncio
class TestDAGGraphConversion:
    """Tests for from_node_graph() and to_node_graph() methods"""

    async def test_from_node_graph_simple(self):
        """from_node_graph() should create a DAG from a simple graph"""
        var = G.V(val=5, name='input')
        t1 = G.t(_Const(10), _name='proc1', x=var)

        dag = DataFlow.from_node_graph([var, t1])

        assert 'input' in dag.nodes
        assert 'proc1' in dag.nodes

    async def test_from_node_graph_complex(self):
        """from_node_graph() should handle complex dependencies"""
        var1 = G.V(val=1, name='a')
        var2 = G.V(val=2, name='b')
        t1 = G.t(_Add(), _name='sum', a=var1, b=var2)

        dag = DataFlow.from_node_graph([var1, var2, t1])

        assert 'a' in dag.nodes
        assert 'b' in dag.nodes
        assert 'sum' in dag.nodes
        assert isinstance(dag._args.data['sum']['a'], Ref)
        assert isinstance(dag._args.data['sum']['b'], Ref)

    async def test_from_node_graph_requires_names(self):
        """from_node_graph() should raise ValueError if nodes lack names"""
        var = G.V(val=5)

        with pytest.raises(ValueError, match="must have a name"):
            DataFlow.from_node_graph([var])

    async def test_to_node_graph_creates_var_nodes(self):
        """to_node_graph() should create Var nodes"""
        dag = DataFlow()
        dag.add_inp('input', val=10)

        nodes = dag.to_node_graph()

        assert len(nodes) == 1
        assert isinstance(nodes[0], G.V)
        assert nodes[0].name == 'input'
        assert nodes[0].val == 10

    async def test_to_node_graph_creates_t_nodes(self):
        """to_node_graph() should create T nodes"""
        dag = DataFlow()
        inp_ref = dag.add_inp('input', val=5)
        dag.link('proc', _Const(10), x=inp_ref)

        nodes = dag.to_node_graph()

        t_nodes = [n for n in nodes if isinstance(n, G.T)]
        assert len(t_nodes) == 1
        assert t_nodes[0].name == 'proc'

    async def test_roundtrip_preserves_structure(self):
        """Converting DAG -> nodes -> DAG should preserve structure"""
        var = G.V(val=5, name='input')
        t1 = G.t(_Const(10), _name='proc1', x=var)

        dag1 = DataFlow.from_node_graph([var, t1])
        dag1.set_out('proc1')

        nodes = dag1.to_node_graph()

        dag2 = DataFlow.from_node_graph(nodes)
        dag2.set_out('proc1')

        names1 = set(dag1.nodes.keys())
        names2 = set(dag2.nodes.keys())
        assert names1 == names2


@pytest.mark.asyncio
class TestDAGIntegration:
    """Integration tests for complex DAG scenarios"""

    async def test_dag_with_parallel_branches(self):
        """DAG should handle multiple independent branches"""
        dag = DataFlow()
        dag.add_inp('x', val=5)
        dag.link('double', _Add(), a=Ref('x'), b=Ref('x'))
        dag.link('triple', _Add(), a=Ref('double'), b=Ref('x'))
        dag.set_out(['double', 'triple'])

        result = await dag.aforward()
        assert result == (10, 15)

    async def test_dag_with_deep_nesting(self):
        """DAG should handle deep dependency chains"""
        dag = DataFlow()
        dag.add_inp('start', val=1)
        dag.link('step1', _Add(), a=Ref('start'), b=1)
        dag.link('step2', _Add(), a=Ref('step1'), b=1)
        dag.link('step3', _Add(), a=Ref('step2'), b=1)
        dag.set_out('step3')

        result = await dag.aforward()
        assert result == 4

    async def test_dag_execution_memoization(self):
        """DAG should memoize node results during execution"""
        dag = DataFlow()
        shared = _Const(5)
        dag.add_inp('x', val=1)
        dag.link('shared', shared)
        dag.link('a', _Add(), a=Ref('shared'), b=Ref('x'))
        dag.link('b', _Add(), a=Ref('shared'), b=2)
        dag.set_out(['a', 'b'])

        await dag.aforward()

        assert shared.calls == 1

    async def test_dag_with_mixed_sync_async(self):
        """DAG should handle both sync and async processes"""
        dag = DataFlow()
        dag.link('sync', _Const(10))
        dag.link('async', _AsyncConst(20))
        dag.set_out(['sync', 'async'])

        result = await dag.aforward()
        assert result == (10, 20)

    async def test_dag_error_propagation(self):
        """Errors in nodes should propagate properly"""
        class ErrorProc(P.Process):
            def forward(self):
                raise ValueError("Test error")

        dag = DataFlow()
        dag.link('error', ErrorProc())
        dag.set_out('error')

        with pytest.raises(ValueError, match="Test error"):
            await dag.aforward()


@pytest.mark.asyncio
class TestDAGEdgeCases:
    """Edge case tests for DataFlow"""

    async def test_empty_dataflow_aforward_returns_none(self):
        """Empty DataFlow with no outputs should return None"""
        dag = DataFlow()
        result = await dag.aforward()
        assert result is None

    async def test_dataflow_with_only_inputs(self):
        """DataFlow with only inputs should work"""
        dag = DataFlow()
        dag.add_inp('x', val=10)
        dag.set_out('x')
        result = await dag.aforward()
        assert result == 10

    async def test_link_with_no_arguments(self):
        """Nodes can be linked without any arguments"""
        dag = DataFlow()
        dag.link('const', _Const(42))
        dag.set_out('const')
        result = await dag.aforward()
        assert result == 42

    async def test_multiple_independent_branches(self):
        """Multiple independent computation branches should work"""
        dag = DataFlow()
        dag.link('a', _Const(1))
        dag.link('b', _Const(2))
        dag.link('c', _Const(3))
        dag.set_out(['a', 'b', 'c'])
        result = await dag.aforward()
        assert result == (1, 2, 3)

    async def test_out_override_preserves_original_outputs(self):
        """out_override should not modify the DataFlow's _outputs"""
        dag = DataFlow()
        dag.link('a', _Const(1))
        dag.link('b', _Const(2))
        dag.set_out('a')

        await dag.aforward(out_override='b')
        assert dag.outputs == 'a'  # Should not be modified


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
