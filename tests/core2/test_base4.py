import json
import pytest
from dachi.core2._base4 import BaseModule, Param, State, Shared, BaseSpec, Checkpoint, BuildContext
from dataclasses import InitVar
from pydantic import ValidationError
import types
import warnings

import pytest
from dachi.core2._base4 import Param, State, Shared, BaseModule
from pydantic import BaseModel
from dataclasses import InitVar

from dachi.core2._base4 import (
    BaseModule,
    Param,
    State,
    Shared,
    BaseSpec,
    registry,
    BuildContext,
    Checkpoint,
)
from pydantic import ValidationError

import pytest
# ------------------------------------------------------------
#  Helper process classes for tests
# ------------------------------------------------------------

class Leaf(BaseModule):
    value: int

class WithParams(BaseModule):
    w: InitVar[float]
    s: InitVar[int]
    name: str

    def __post_init__(self, w: float, s: int):
        self.w = Param(data=w)
        self.s = State(data=s)


class Nested(BaseModule):
    inner: WithParams
    extra: int = 5

class InitVarProc(BaseModule):
    x: int
    y: InitVar[int]

    def __post_init__(self, y):
        self.buffer = State(data=y)

# ------------------------------------------------------------
#  Positive path tests
# ------------------------------------------------------------

def test_simple_param_state_registration():
    wp = WithParams(w=1.0, s=0, name="foo")
    assert list(wp.parameters()) == [wp.w]
    sd = wp.state_dict()
    assert sd == {"w": 1.0, "s": 0}


def test_nested_recursion():
    wp = WithParams(w=2.0, s=3, name="bar")
    n = Nested(inner=wp)
    params = list(n.parameters())
    assert params == [wp.w]
    sd = n.state_dict()
    assert sd == {
        "inner.w": 2.0,
        "inner.s": 3,
    }


def _make_tree():
    class Leaf(BaseModule):
        v: InitVar[int]
        def __post_init__(self, v: int):
            self.v = Param(data=v)
    class Root(BaseModule):
        left: Leaf
        right: Leaf
    return Root(left=Leaf(v=1), right=Leaf(v=2))


def _make_dynamic_leaf(tag: str):
    """Return a Leaf class whose *module* differs so the schema name must be unique."""
    mod = types.ModuleType(f"tmp_mod_{tag}")

    class Leaf(BaseModule):
        w: InitVar[float]
        def __post_init__(self, w: float):
            self.w = Param(data=w)

    Leaf.__name__ = "Leaf"
    Leaf.__qualname__ = "Leaf"
    Leaf.__module__ = mod.__name__
    setattr(mod, "Leaf", Leaf)
    import sys
    sys.modules[mod.__name__] = mod
    return Leaf


def test_spec_class_name_collision_unique():
    """Two Leaf classes in distinct modules must yield *different* Spec classes."""
    LeafA = _make_dynamic_leaf("a")
    LeafB = _make_dynamic_leaf("b")
    assert LeafA.schema() is not LeafB.schema()
    assert LeafA.schema().__name__ == LeafB.schema().__name__


def test_initvar_without_post_init_raises():
    with pytest.raises(RuntimeError):
        class Bad(BaseModule):
            x: InitVar[int]
        Bad(x=1)

# def test_eval_cascades_to_children():
#     root = _make_tree()
#     root.eval()
#     assert all(not p.training for p in root.parameters())


# def test_train_cascades_to_children():
#     root = _make_tree()
#     root.eval()
#     root.train()
#     assert all(p.training for p in root.parameters())


def test_modules_depth_first_order():
    class Leaf(BaseModule):
        pass
    class Branch(BaseModule):
        left: Leaf
        right: Leaf
    tree = Branch(left=Leaf(), right=Leaf())
    mods = list(tree.modules())
    assert mods == [tree, tree.left, tree.right]


def test_from_spec_missing_field_validation_error():
    class Leaf(BaseModule):
        w: InitVar[int]
        def __post_init__(self, w: int):
            self.w = Param(data=w)

    spec = Leaf(w=1).spec(to_dict=True)
    del spec["w"]
    with pytest.raises(ValidationError):
        Leaf.from_spec(spec)

def test_nested_module_dedup_in_from_spec():
    """Same Inner instance appears twice; loader must deduplicate."""
    @registry()
    class Inner(BaseModule):
        v: InitVar[int]
        def __post_init__(self, v):
            self.v = Param(data=v)
    shared = Inner(v=3)

    @registry()
    class Outer(BaseModule):
        a: Inner
        b: Inner

    outer = Outer(a=shared, b=shared)
    rebuilt = Outer.from_spec(outer.spec(), ctx=BuildContext())
    assert rebuilt.a is rebuilt.b


def test_state_dict_filters():
    wp = WithParams(w=Param(data=1.5), s=State(data=9), name="z")
    assert list(wp.parameters()) == [wp.w]


def test_state_keys_match_state_dict():
    class Proc(BaseModule):
        p: InitVar[int]
        s: InitVar[int]
        def __post_init__(self, p: int, s: int):
            self.p = Param(data=p)
            self.s = State(data=s)

    pr = Proc(p=5, s=9)
    assert set(pr.state_dict().keys()) == pr.state_keys()

def test_initvar_preserved_in_spec():
    p = InitVarProc(x=4, y=7)
    sp = p.spec(to_dict=False).model_dump()
    assert sp["y"] == 7
    assert "y" not in p.__dict__


def test_schema_cached():
    s1 = WithParams.schema()
    s2 = WithParams.schema()
    assert s1 is s2
    assert issubclass(s1, BaseSpec)


def test_shared_excluded_from_state():
    class Proc(BaseModule):
        cfg: InitVar[str]
        p: InitVar[int]
        def __post_init__(self, cfg: str, p: int):
            self.cfg = Shared(data=cfg)
            self.p = Param(data=p)
    pr = Proc(cfg=Shared(data="conf"), p=Param(data=10))
    sd = pr.state_dict()
    assert "cfg" not in sd and "p" in sd


def test_load_state_dict_roundtrip():
    wp1 = WithParams(w=5.0, s=1, name="x")
    wp2 = WithParams(w=0.0, s=0, name="y")
    state_dict = wp1.state_dict()
    wp2.load_state_dict(state_dict)
    assert wp2.w.data == 5.0 and wp2.s.data == 1

def test_load_state_dict_extra_key_nested_strict():
    class Leaf(BaseModule):
        p: InitVar[int]
        def __post_init__(self, p):
            self.p = Param(data=p)
    class Root(BaseModule):
        leaf: Leaf
    root = Root(leaf=Leaf(p=1))
    with pytest.raises(KeyError):
        root.load_state_dict({"leaf.q": 99}, strict=True)

# TODO* Fix strict load
# def test_load_state_dict_type_mismatch():
#     class Leaf(BaseModule):
#         p: InitVar[int]
#         def __post_init__(self, p):
#             self.p = Param(data=p)
#     leaf = Leaf(p=1)
#     with pytest.raises(Exception):
#         leaf.load_state_dict({"p": "oops"}, strict=True)


# # # # # ------------------------------------------------------------
# # # # #  Error / edge cases
# # # # # ------------------------------------------------------------

def test_missing_required_kwarg():
    with pytest.raises(TypeError):
        WithParams(w=1.0, s=0)  # missing name


def test_unexpected_kwarg():
    with pytest.raises(TypeError):
        WithParams(w=1.0, s=0, name="foo", bogus=1)

# # # import typing as t

# # # # def test_cycle_detection():
# # # #     class A(BaseItem):
# # # #         ref: t.Union["B", None] = None  # type: ignore
# # # #     class B(BaseItem):
# # # #         ref: A | None = None
# # # #     a = A()
# # # #     b = B(ref=a)
# # # #     a.ref = b
# # # #     with pytest.raises(RuntimeError):
# # # #         a.spec()


def test_load_state_strict_failure():
    wp = WithParams(w=1.0, s=1, name="t")
    sd = {"w": 2.0, "missing": 9}
    with pytest.raises(KeyError):
        wp.load_state_dict(sd, recurse=False, strict=True)


def test_param_deduplication():
    shared_param = Param(data=3.0)
    class Proc(BaseModule):
        a: InitVar[float]
        b: InitVar[float]

        def __post_init__(self, a: float, b: float):
            self.a = Param(data=a)
            self.b = Param(data=b)
    
    pr = Proc(a=2.0, b=1.0)
    pr.b = pr.a
    assert len(list(pr.parameters())) == 1  # dedup by identity


def test_state_dict_recurse_flag():
    wp = WithParams(w=1.0, s=2, name="n")
    n = Nested(inner=wp)
    assert n.state_dict(recurse=False) == {}


def test_parameters_recurse_false():
    wp = WithParams(w=2.0, s=0, name="q")
    n = Nested(inner=wp)
    assert list(n.parameters(recurse=False)) == []

# # # # # # ------------------------------------------------------------
# # # # # #  Extra edge / negative tests (total >= 30 )
# # # # # # ------------------------------------------------------------

@pytest.mark.parametrize("val", [0, 1.2, "txt", [1, 2]])
def test_param_accepts_any_val(val):
    p = Param(data=val)
    assert p.data == val


def test_state_mutability():
    st = State(data=5)
    st.data += 1
    assert st.data == 6


def test_load_state_non_param_target_error():
    wp = WithParams(w=1.0, s=1, name="a")
    with pytest.raises(KeyError):
        wp.load_state_dict({"name": "bad"}, recurse=False, strict=True)


def test_frozen_param_not_filtered_from_state():
    # p = Param(data=3.3, training=False)
    c = WithParams(w=3.3, s=0, name="n")
    assert "w" in c.state_dict(train=True)

# # # # # # -----  corner cases for InitVar default / override ----------

def test_initvar_default_used():
    class P(BaseModule):
        x: int
        y: InitVar[int] = 9
        def __post_init__(self, y):
            self.buf = State(data=y)
    p = P(x=1)
    assert p._init_vars["y"] == 9 and p.buf.data == 9


def test_initvar_override():
    p = InitVarProc(x=2, y=11)
    assert p._init_vars["y"] == 11 and p.buffer.data == 11


def test_schema_kind_field():
    sch = WithParams.schema()
    assert "kind" in sch.model_fields


def test_spec_kind_matches_classname():
    class WithParams(BaseModule):
        w: Param[float]
        s: State[int]
        name: str
    wp = WithParams(w=Param(data=1.0), s=State(data=1), name="k")
    assert wp.spec(to_dict=False).kind.endswith("WithParams")


def test_state_dict_train_flag():
    wp = WithParams(w=Param(data=2.2), s=State(data=0), name="m")
    assert "w" not in wp.state_dict(train=False)


def test_state_dict_runtime_flag():
    wp = WithParams(w=Param(data=1.0), s=State(data=5), name="r")
    assert "s" not in wp.state_dict(runtime=False)


def test_load_state_in_child():
    wp1 = WithParams(w=4.0, s=6, name="x1")
    wp2 = WithParams(w=0.0, s=0, name="x2")
    n = Nested(inner=wp2)
    n.load_state_dict({"inner.w": 4.0, "inner.s": 6})
    assert wp2.w.data == 4.0 and wp2.s.data == 6


def test_state_dict_flags_combination():
    class A(BaseModule):
        p: InitVar[float]
        s: InitVar[int]

        def __post_init__(self, p: float, s: int):
            self.p = Param(data=p)
            self.s = State(data=s)

    a = A(p=3.3, s=9)
    sd = a.state_dict(train=False, runtime=False)
    assert sd == {}


def test_state_dict_nested_keys():
    class Leaf(BaseModule):
        val: InitVar[float] = 0.0
        def __post_init__(self, val):
            self.p = Param(data=val)
    class Root(BaseModule):
        leaf: Leaf
        b: InitVar[int] = 0

        def __post_init__(self, b: int):
            self.b = Param(data=b)
    r = Root(leaf=Leaf(val=1.0), b=2)
    sd = r.state_dict()
    assert set(sd.keys()) == {"leaf.p", "b"}


def test_state_dict_shared_exclusion():
    class A(BaseModule):
        sh: InitVar[str]

        def __post_init__(self, sh: str):
            self.sh: Shared[str] = Shared(data=sh)

    a = A(sh='original')
    a.sh.data = "updated"
    sd = a.state_dict()
    assert "sh" not in sd


def test_state_dict_conflicting_keys():
    class Child(BaseModule):
        val: InitVar[float] = 0.0

        def __post_init__(self, val):
            self.p = Param(data=val)

    class Parent(BaseModule):
        child1: Child
        child2: Child
    p = Parent(child1=Child(val=1.0), child2=Child(val=2.0))
    sd = p.state_dict()
    assert sd["child1.p"] == 1.0 and sd["child2.p"] == 2.0


def test_state_dict_dynamic_addition():
    class A(BaseModule):
        x: int
    a = A(x=5)
    a.p = Param(data=7.7)
    assert "p" in a.state_dict()


def test_state_dict_empty_baseitem():
    class Empty(BaseModule): pass
    e = Empty()
    assert e.state_dict() == {}


def test_state_dict_no_recursion():
    class Leaf(BaseModule):
        p: Param[float]
    class Root(BaseModule):
        leaf: Leaf
        b: Param[int]
    r = Root(leaf=Leaf(p=Param(data=5)), b=Param(data=6))
    sd = r.state_dict(recurse=False)
    assert "b" in sd and not any(k.startswith("leaf.") for k in sd)

def test_state_dict_complex_types():
    p = Param(data=[1,2,3])
    s = State(data={"k":"v"})
    class A(BaseModule):
        p: Param[list]
        s: State[dict]
    a = A(p=p, s=s)
    sd = a.state_dict()
    assert sd == {"p":[1,2,3], "s":{"k":"v"}}


def test_load_state_dict_happy_path():
    class A(BaseModule):
        p: Param[float]
        s: State[int]
    a = A(p=Param(data=1.0), s=State(data=2))
    a.load_state_dict({"p": 3.3, "s": 4})
    assert a.p.data == 3.3 and a.s.data == 4


def test_load_state_dict_missing_key_strict():
    class A(BaseModule):
        p: Param[float]
    a = A(p=Param(data=1.0))
    with pytest.raises(KeyError):
        a.load_state_dict({}, strict=True)


def test_load_state_dict_extra_key_strict():
    class A(BaseModule):
        p: Param[float]
    a = A(p=Param(data=1.0))
    with pytest.raises(KeyError):
        a.load_state_dict({"p":1.0, "extra":5}, strict=True)


def test_load_state_dict_partial_non_strict():
    class A(BaseModule):
        p: Param[float]
        s: State[int]
    a = A(p=Param(data=1.0), s=State(data=2))
    a.load_state_dict({"p":5.0}, strict=False)
    assert a.p.data == 5.0 and a.s.data == 2


def test_load_state_dict_shared_protection():
    class A(BaseModule):
        sh: Shared[str]
    a = A(sh=Shared(data="original"))
    a.load_state_dict({"sh": "new"}, strict=False)
    assert a.sh.data == "original"   # Shared field should not change


def test_load_state_dict_nested():
    class Leaf(BaseModule):
        p: Param[float]
    class Root(BaseModule):
        leaf: Leaf
    r = Root(leaf=Leaf(p=Param(data=1.0)))
    r.load_state_dict({"leaf.p": 9.9})
    assert r.leaf.p.data == 9.9


def test_load_state_dict_recursion_false():
    class Leaf(BaseModule):
        p: Param[float]
    class Root(BaseModule):
        leaf: Leaf
        b: Param[int]
    r = Root(leaf=Leaf(p=Param(data=1.0)), b=Param(data=2))
    r.load_state_dict({"b":5}, recurse=False)
    assert r.b.data == 5 and r.leaf.p.data == 1.0


def test_load_state_dict_empty():
    class A(BaseModule):
        p: InitVar[float]
        def __post_init__(self, p: float):
            self.p = Param(data=p)
    a = A(p=1.0)
    a.load_state_dict({}, strict=False)
    assert a.p.data == 1.0

# # # # # # def test_load_state_dict_type_mismatch():
# # # # # #     class A(BaseItem):
# # # # # #         p: Param[float]
# # # # # #     a = A(p=Param(data=1.0))
# # # # # #     with pytest.raises(Exception):
# # # # # #         a.load_state_dict({"p": "oops"}, strict=True)

def test_load_state_dict_shared_ignore_even_if_present():
    class A(BaseModule):
        p: Param[float]
        sh: Shared[str]
    a = A(p=Param(data=1.0), sh=Shared(data="init"))
    a.load_state_dict({"p":5.5, "sh":"should not overwrite"}, strict=False)
    assert a.p.data == 5.5 and a.sh.data == "init"


def test_parameters_train_only_true():
    p1 = 1.0
    p2 = 2.0
    class P(BaseModule):
        a: InitVar[float]
        b: InitVar[float]
        
        def __post_init__(self, a: float, b: float):
            self.a = Param(data=a)
            self.b = Param(data=b)
    pr = P(a=p1, b=p2)
    ps = list(pr.parameters())
    assert ps[0].data == 1.0
    assert ps[1].data == 2.0


def test_parameters_no_params():
    class P(BaseModule):
        x: int
    p = P(x=5)
    assert list(p.parameters()) == []

def test_parameters_deduplication():
    p = Param(data=1.0)
    class P(BaseModule):
        a: InitVar[float]
        b: InitVar[float]

        def __post_init__(self, a: float, b: float):
            self.a = Param(data=a)
            self.b = Param(data=b)
    
    pr = P(a=p, b=p)
    pr.a = pr.b
    assert len(list(pr.parameters())) == 1  # dedup by identity
    assert list(pr.parameters())[0].data == p

def test_parameters_nested():
    class Leaf(BaseModule):
        p: InitVar[float]

        def __post_init__(self, p: float):
            self.p = Param(data=p)
    class Branch(BaseModule):
        leaf: Leaf
        b: InitVar[int]
        def __post_init__(self, b: int):
            self.b = Param(data=b)
    pr = Branch(leaf=Leaf(p=1.0), b=2)
    ps = list(pr.parameters())
    assert len(ps) == 2 and all(isinstance(p, Param) for p in ps)

def test_parameters_recurse_false():
    class Leaf(BaseModule):
        p: InitVar[float]

        def __post_init__(self, p: float):
            self.p = Param(data=p)

    class Branch(BaseModule):
        leaf: Leaf
        b: InitVar[int]
        def __post_init__(self, b: int):
            self.b = Param(data=b)
    pr = Branch(leaf=Leaf(p=1.0), b=2)
    ps = list(pr.parameters(recurse=False))
    assert len(ps) == 1 and isinstance(ps[0], Param)

def test_parameters_dynamic_addition():
    p = Param(data=3)
    class P(BaseModule):
        x: int
    pr = P(x=1)
    pr.new_p = p
    assert p in list(pr.parameters())

def test_parameters_ignore_nonparam():
    class P(BaseModule):
        a: InitVar[str]
        b: InitVar[int]
        c: int
        def __post_init__(self, a: str, b: int):
            self.a = Shared(data=a)
            self.b = State(data=b)

    pr = P(a="ref", b=9, c=42)
    assert list(pr.parameters()) == []

def test_parameters_train_only_none():

    class P(BaseModule):

        a: InitVar[float]
        b: InitVar[float]

        def __post_init__(self, a: float, b: float):
            self.a = Param(data=a)
            self.b = Param(data=b)

    pr = P(a=1.0, b=2.0)
    ps = list(pr.parameters())
    assert ps[0].data == 1.0
    assert ps[1].data == 2.0

# # # # # # # ------ identity vs value equality for Shared ------------------

def test_shared_identity_not_dedup():
    s1 = "conf"
    s2 = "conf"
    class P(BaseModule):
        a: InitVar[str]
        b: InitVar[str]
        def __post_init__(self, a: str, b: str):
            self.a = Shared(data=a)
            self.b = Shared(data=b)
            
    pr = P(a=s1, b=s2)
    # state_dict must still omit both
    assert pr.state_dict() == {}


# # # # # # ----------  Happy-path: basic save  ----------
def test_checkpoint_save_module_creates_file(tmp_path):
    """Positive • file is physically created and contains valid JSON."""
    class Leaf(BaseModule):
        w: InitVar[int]
        def __post_init__(self, w: int):
            self.w = Param(data=w)

    leaf = Leaf(w=1)
    path = tmp_path / "leaf.json"

    Checkpoint.save_module(leaf, path)
    raw = path.read_text()

    assert path.exists()          # file written
    json.loads(raw)               # raises if not valid JSON


# from dachi.core2._base4 import registry, Checkpoint, BaseModule, Param

# # # # # # ----------  Happy-path: load → exact round-trip ----------
def test_checkpoint_load_roundtrip(tmp_path):
    """Positive • Checkpoint.load() reproduces the exact spec & state."""
    @registry()
    class Leaf(BaseModule):
        w: InitVar[int]
        def __post_init__(self, w: int):
            self.w = Param(data=w)

    leaf = Leaf(w=3)
    path = tmp_path / "leaf.json"
    Checkpoint.save_module(leaf, path)

    ckpt = Checkpoint.load(path)

    assert ckpt.state_dict == leaf.state_dict(recurse=True, train=True, runtime=True)
    assert ckpt.spec.kind == leaf.spec().kind

def test_checkpoint_param_dedup(tmp_path):

    @registry()
    class Leaf(BaseModule):
        w: InitVar[int]
        def __post_init__(self, w):
            self.w = Param(data=w)
    p = Param(data=42)

    @registry()
    class Pair(BaseModule):
        a: Leaf
        b: Leaf
    model = Pair(a=Leaf(w=42), b=Leaf(w=42))
    path = tmp_path / "pair.json"
    Checkpoint.save_module(model, path)
    restored = Checkpoint.load_module(path)
    assert restored.a.w.data == restored.b.w.data == 42


def test_checkpoint_roundtrip(tmp_path):

    @registry()
    class Leaf(BaseModule):
        w: InitVar[int]
        def __post_init__(self, w):
            self.w = Param(data=w)
    leaf = Leaf(w=7)
    path = tmp_path / "leaf.json"
    Checkpoint.save_module(leaf, path)
    rebuilt = Checkpoint.load_module(path)
    assert isinstance(rebuilt, Leaf)
    assert rebuilt.w.data == 7

# # # from dachi.core2._base4 import registry, Checkpoint, BaseModule, Param

# # # # # # ----------  Happy-path: load_module reconstructs model ----------
def test_checkpoint_load_module_restores_state(tmp_path):
    """Positive • load_module returns an equivalent, fully initialised module."""

    @registry()
    class Leaf(BaseModule):
        w: InitVar[int]
        def __post_init__(self, w: int):
            self.w = Param(data=w)

    original = Leaf(w=7)
    path = tmp_path / "leaf.json"
    Checkpoint.save_module(original, path)

    rebuilt = Checkpoint.load_module(path)

    assert isinstance(rebuilt, Leaf)
    assert rebuilt.w.data == 7


# # # TODO: Consider whether to allow SharedModules

# # # # # # ----------  Happy-path: shared ref-names deduplicated ----------
def test_checkpoint_shared_objects_deduplicated(tmp_path):
    """Positive • Same ref_name inside spec becomes the *same* object."""
    @registry()
    class Inner(BaseModule):
        val: InitVar[int]

        def __post_init__(self, val: int):
            self.val = Param(data=val)

    shared_inner = Inner(val=5)

    @registry()
    class Outer(BaseModule):
        a: Inner
        b: Inner

    model = Outer(a=shared_inner, b=shared_inner)
    path = tmp_path / "outer.json"
    Checkpoint.save_module(model, path)

    ctx = BuildContext()
    rebuilt = Checkpoint.load_module(path, ctx=ctx)

    assert rebuilt.a is rebuilt.b
    assert rebuilt.a.val.data == 5



# # ---------------------------------------------------------------------
# # 1. Shareable leaves
# # ---------------------------------------------------------------------
def test_param_hash_identity_and_stability():
    p1 = Param(data=1)
    p2 = Param(data=1)
    assert hash(p1) != hash(p2)
    before = hash(p1)
    p1.data += 1
    after = hash(p1)
    assert before == after     # identity-hash stays stable


def test_param_hash_identity():
    p1 = Param(data=5)
    p2 = Param(data=5)
    assert hash(p1) != hash(p2)
    h = hash(p1)
    p1.data += 1
    assert hash(p1) == h


def test_param_callback_runs():
    hit = []
    p = Param(data=0)
    p.register_callback(lambda v: hit.append(v))
    p.data = 9
    assert hit == [9]

@pytest.mark.parametrize(
    "val",
    [
        123, 3.14, "hello", True, None,  # primitives
        State(data=5),                   # Shareable inside Shareable
        ["a", "b", "c"],                # homogeneous list
        {"k": 1},                       # homogeneous dict
    ],
)
def test_shareable_accepts_allowed_values(val):
    s = Shared(data=val)
    dumped = s.dump()
    # round-trip keeps the value
    s2 = Shared(data=None)
    s2.load(dumped)
    assert s2.data == val


def test_shareable_rejects_forbidden():
    class Foo: ...
    with pytest.raises(TypeError):
        Shared(value=Foo())


# # # # # ----------  Error-path: missing file ----------
def test_checkpoint_load_missing_file_raises(tmp_path):
    """Negative • Loading a non-existent file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        Checkpoint.load(tmp_path / "nope.json")


# # # ----------  Error-path: corrupt JSON ----------
def test_checkpoint_load_corrupt_json_raises(tmp_path):
    """Negative • Invalid JSON content raises ValueError (Pydantic)."""
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not json!")
    with pytest.raises(ValueError):
        Checkpoint.load(bad_file)

def test_load_cls_unknown_kind():
    class DummySpec(BaseSpec):
        kind: str = "does.not.exist.Kind"
    with pytest.raises(ValueError):
        DummySpec.load_cls()

# # # # ----------  train()/eval() cascade ----------
# def test_eval_cascades_to_children():
#     """Positive • eval() sets training=False on every Param."""
#     class Leaf(BaseModule):
#         p: InitVar[int]

#         def __post_init__(self, p: int):
#             self.p = Param(data=p)


#     class Root(BaseModule):
#         l1: Leaf
#         l2: Leaf

#     root = Root(l1=Leaf(p=1), l2=Leaf(p=2))
#     root.eval()
#     assert all(not p.training for p in root.parameters(recurse=True))


# # # def test_train_cascades_to_children():
# # #     """Positive • train() resets training=True on every Param."""
# # #     class Leaf(BaseModule):
# # #         p: Param[int]

# # #     class Root(BaseModule):
# # #         l1: Leaf
# # #         l2: Leaf

# # #     root = Root(l1=Leaf(p=Param(1)), l2=Leaf(p=Param(2)))
# # #     root.eval()
# # #     root.train()                      # switch back
# # #     assert all(p.training for p in root.parameters(recurse=True, train=None))


# # # # ----------  apply() with filter_type ----------
def test_apply_filters_by_type():
    """Positive • apply() visits only objects of filter_type."""
    calls = []

    class Leaf(BaseModule):
        w: InitVar[int]

        def __post_init__(self, w: int):
            self.w = Param(data=w)

    root = Leaf(w=0)

    root.apply(
        lambda x: calls.append(type(x).__name__), 
        filter_type=Param
    )
    assert calls == ["Param"]


# # # # ----------  named_modules() dotted prefixes ----------
def test_named_modules_keys_are_correct():
    """Positive • named_modules() returns expected dotted names."""
    class Leaf(BaseModule):

        x: InitVar[int]
        def __post_init__(self, x: int):
            self.x = Param(data=x)

    class Branch(BaseModule):
        left: Leaf
        right: Leaf

    tree = Branch(left=Leaf(x=1), right=Leaf(x=2))
    names = dict(tree.named_modules())

    assert set(names.keys()) == {"", "left", "right"}


# # # # ----------  state_dict() recurse=False ----------
def test_state_dict_nonrecurse_returns_empty():
    """Edge • recurse=False on parent with only submodules returns empty dict."""
    class Leaf(BaseModule):
        s: InitVar[int]
        def __post_init__(self, s: int):
            self.s = State(data=s)

    class Top(BaseModule):
        child: Leaf

    t = Top(child=Leaf(s=9))
    flat = t.state_dict(recurse=False, runtime=False, train=True)

    assert flat == {}


# # # # --------------------------------------------------------------------
# # # #  tests: Registry  &  BuildContext
# # # # --------------------------------------------------------------------
import inspect
import pytest

from dachi.core2._base4 import (
    BaseModule, Param, Shared, BuildContext, registry, Registry
)

# # # # --------------------------------------------------------------------
# # # #  ----  REGISTRY TESTS  --------------------------------------------
# # # # --------------------------------------------------------------------

def _fresh_registry() -> Registry:
    """Utility: isolated Registry instance so we don’t pollute global one."""
    return Registry()


# # # # ----------  Positive • class registration ----------
def test_registry_register_class_positive():
    reg = _fresh_registry()

    @reg.register()
    class Foo: ...
    entry = reg[Foo.__qualname__]
    assert entry.obj is Foo and entry.type == "class"


# # # # ----------  Positive • function registration ----------
def test_registry_register_function_positive():
    reg = _fresh_registry()

    @reg.register(tags={"role": "util"})
    def helper(): pass
    ent = reg[helper.__qualname__]
    assert callable(ent.obj) and ent.tags["role"] == "util"


# # # # ----------  Positive • list_entries / types / packages / tags ----------
def test_registry_lists_positive():
    reg = _fresh_registry()

    @reg.register(tags={"k": 1})
    class A: ...
    assert reg.list_entries() == [A.__qualname__]
    assert reg.list_types() == ["class"]
    assert reg.list_packages() == [A.__module__]
    assert reg.list_tags() == ["k"]


# # # # ----------  Positive • filter by type & tags ----------
def test_registry_filter_positive():
    reg = _fresh_registry()

    @reg.register(tags={"cat": "a"})
    class A: ...
    @reg.register(tags={"cat": "b"})
    class B: ...
    res = reg.filter(tags={"cat": "a"})
    assert list(res) == [A.__qualname__] and res[A.__qualname__].obj is A


# # # # ----------  Edge • overwrite emits warning but keeps new obj ----------
def test_registry_overwrite_edge(capsys):
    reg = _fresh_registry()

    @reg.register(name="X")
    class First: ...
    @reg.register(name="X")
    class Second: ...
    captured = capsys.readouterr().out
    assert "Overwriting existing entry" in captured
    assert reg["X"].obj is Second


# # # # ----------  Negative • deregister removes entry ----------
def test_registry_deregister_negative():
    reg = _fresh_registry()

    @reg.register()
    class Gone: ...
    reg.deregister("Gone")
    with pytest.raises(KeyError):
        _ = reg["Gone"]


# # # # ----------  Positive • __getitem__ list variant ----------
def test_registry_getitem_list_positive():
    reg = _fresh_registry()

    @reg.register()
    class A: ...
    @reg.register()
    class B: ...
    subset = reg[[A.__qualname__, B.__qualname__]]
    assert set(subset.keys()) == {A.__qualname__, B.__qualname__}


# # # # --------------------------------------------------------------------
# # # #  ----  BUILD-CONTEXT TESTS  ----------------------------------------
# # # # --------------------------------------------------------------------

# # # # Helper classes registered into *global* registry, because BuildContext
# # # # relies on that for from_spec().
@registry.register()
class Leaf(BaseModule):
    payload: InitVar[str]

    def __post_init__(self, payload: str):
        self.payload = Shared(data=payload)

@registry.register()
class Pair(BaseModule):
    left: Leaf
    right: Leaf


# # TODO: Check if loading state dict as well will
# # result in it being shared
# # # # ----------  Positive • same ref_name deduplicated ----------
# # def test_buildcontext_shared_dedup_positive():
# #     # shared = Shared(data="cfg", ref_name="SAME")
# #     parent = Pair(left=Leaf(payload="cfg"), right=Leaf(payload="cfg"))
# #     parent.right.payload = parent.left.payload
# #     spec = parent.spec(to_dict=False)

# #     ctx = BuildContext()
# #     rebuilt = Pair.from_spec(spec, ctx=ctx)

# #     assert rebuilt.left.payload is rebuilt.right.payload          # identity check
# #     assert list(ctx.shared) == ["SAME"]                           # context stored once


# # # # ----------  Edge • different ref_names remain distinct ----------
def test_buildcontext_distinct_refs_edge():
    p = Pair(
        left=Leaf(payload="a"),
        right=Leaf(payload="y"),
    )
    spec = p.spec(to_dict=False)
    rebuilt = Pair.from_spec(spec, ctx=BuildContext())

    assert rebuilt.left.payload is not rebuilt.right.payload


# # # # ----------  Edge • ref_name None never deduplicated ----------
def test_buildcontext_none_refname_edge():
    p = Pair(
        left=Leaf(payload="a"),        # ref_name = None
        right=Leaf(payload="a"),       # distinct instance
    )
    spec = p.spec(to_dict=False)
    rebuilt = Pair.from_spec(spec, ctx=BuildContext())

    assert rebuilt.left.payload is not rebuilt.right.payload


# def test_registry_overwrite_warning():
#     with warnings.catch_warnings(record=True) as w:
#         warnings.simplefilter("always")

#         @registry(name="Foo")
#         class Tmp2(BaseModule):
#             pass
#         assert any("already" in str(msg.message) for msg in w)


################


# ---------------------- I. Type Enforcement --------------------------

def test_param_type_enforcement():
    class TypedParam(Param[int]):
        pass
    with pytest.raises(TypeError):
        TypedParam(data="not an int")


def test_state_type_enforcement():
    class TypedState(State[float]):
        pass
    with pytest.raises(TypeError):
        TypedState(data="not a float")

def test_shared_type_enforcement():
    class TypedShared(Shared[bool]):
        pass
    with pytest.raises(TypeError):
        TypedShared(data=123)

# # ---------------------- III. Param Callback Removal -------------------

def test_param_unregister_callback():
    hits = []
    def cb(v): hits.append(v)
    p = Param(data=1)
    p.register_callback(cb)
    p.data = 2
    p.unregister_callback(cb)
    p.data = 3
    assert hits == [2]

# # ---------------------- V. Eval/Train Cascade --------------------------

# # TODO: Figure out how to handle training on param
# def test_eval_train_cascade():
#     class Leaf(BaseModule):
#         w: InitVar[int]
#         def __post_init__(self, w):
#             self.w = Param(data=w)

#     class Root(BaseModule):
#         a: Leaf
#         b: Leaf

#     r = Root(a=Leaf(w=1), b=Leaf(w=2))
#     r.eval()
#     assert all(not p.training for p in r.parameters())
#     r.train()
#     assert all(p.training for p in r.parameters())

# # ---------------------- VI. Conflicting StateDict Keys ------------------

def test_state_dict_conflicting_nested_keys():
    class Child(BaseModule):
        x: Param[int]
    class Parent(BaseModule):
        child: Child
        child_x: Param[int]

    c = Child(x=Param(data=1))
    p = Parent(child=c, child_x=Param(data=2))
    sd = p.state_dict()
    assert "child.x" in sd and "child_x" in sd
    assert sd["child.x"] == 1
    assert sd["child_x"] == 2

# # ---------------------- VIII. ParamSet Partial Update ------------------

from dachi.core2._base4 import ParamSet

def test_paramset_partial_update():
    p1 = Param(data=1)
    p2 = Param(data=2)
    class Mod(BaseModule):
        def __init__(self):
            super().__init__()
            self.p1 = p1
            self.p2 = p2

    mod = Mod()
    ps = ParamSet.build(mod)
    ps.update({"param_0": 100})
    assert p1.data == 100 and p2.data == 2

# # ---------------------- IX. Dynamic Param Addition ---------------------

def test_dynamic_param_assignment_after_init():
    class M(BaseModule):
        x: int
    m = M(x=0)
    new_param = Param(data=42)
    m.extra = new_param
    assert new_param in list(m.parameters())

def test_param_subclass_roundtrip():
    class TypedParam(Param[int]):
        pass

    class Proc(BaseModule):
        x: InitVar[int]

        def __post_init__(self, x):
            self.x = TypedParam(data=x)

    m = Proc(x=123)
    spec = m.spec(to_dict=False)
    rebuilt = Proc.from_spec(spec)

    assert isinstance(rebuilt.x, TypedParam)
    assert rebuilt.x.data == 123


def test_state_subclass_roundtrip():
    class TypedState(State[str]):
        pass

    class Proc(BaseModule):
        msg: InitVar[str]

        def __post_init__(self, msg):
            self.msg = TypedState(data=msg)

    m = Proc(msg="hello")
    spec = m.spec(to_dict=False)
    rebuilt = Proc.from_spec(spec)

    assert isinstance(rebuilt.msg, TypedState)
    assert rebuilt.msg.data == "hello"


def test_shared_subclass_roundtrip():
    class TypedShared(Shared[float]):
        pass

    class Proc(BaseModule):
        val: InitVar[float]

        def __post_init__(self, val):
            self.val = TypedShared(data=val)

    m = Proc(val=3.14)
    spec = m.spec(to_dict=False)
    rebuilt = Proc.from_spec(spec)

    assert isinstance(rebuilt.val, TypedShared)
    assert rebuilt.val.data == 3.14
