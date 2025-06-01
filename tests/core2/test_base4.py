import json
import pytest
from dachi.core2._base4 import BaseModule, Param, State, Shared, BaseSpec, Checkpoint
from dataclasses import InitVar

# ------------------------------------------------------------
#  Helper process classes for tests
# ------------------------------------------------------------

class Leaf(BaseModule):
    value: int

class WithParams(BaseModule):
    w: Param[float]
    s: State[int]
    name: str

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
    wp = WithParams(w=Param(data=1.0), s=State(data=0), name="foo")
    assert list(wp.parameters()) == [wp.w]
    sd = wp.state_dict()
    assert sd == {"w": 1.0, "s": 0}


def test_nested_recursion():
    wp = WithParams(w=Param(data=2.0), s=State(data=3), name="bar")
    n = Nested(inner=wp)
    params = list(n.parameters())
    assert params == [wp.w]
    sd = n.state_dict()
    assert sd == {
        "inner.w": 2.0,
        "inner.s": 3,
    }


def test_state_dict_filters():
    wp = WithParams(w=Param(data=1.5, training=False), s=State(data=9), name="z")
    # train only
    assert list(wp.parameters(train_only=True)) == []
    # frozen only
    assert list(wp.parameters(train_only=False)) == [wp.w]


def test_initvar_preserved_in_spec():
    p = InitVarProc(x=4, y=7)
    sp = p.spec().model_dump()
    assert sp["y"] == 7
    assert "y" not in p.__dict__


def test_schema_cached():
    s1 = WithParams.schema()
    s2 = WithParams.schema()
    assert s1 is s2
    assert issubclass(s1, BaseSpec)


def test_shared_excluded_from_state():
    class Proc(BaseModule):
        cfg: Shared[str]
        p: Param[int]
    pr = Proc(cfg=Shared(data="conf"), p=Param(data=10))
    sd = pr.state_dict()
    assert "cfg" not in sd and "p" in sd


def test_load_state_dict_roundtrip():
    wp1 = WithParams(w=Param(data=5.0), s=State(data=1), name="x")
    wp2 = WithParams(w=Param(data=0.0), s=State(data=0), name="y")
    state_dict = wp1.state_dict()
    wp2.load_state_dict(state_dict)
    assert wp2.w.data == 5.0 and wp2.s.data == 1

# # ------------------------------------------------------------
# #  Error / edge cases
# # ------------------------------------------------------------

def test_missing_required_kwarg():
    with pytest.raises(TypeError):
        WithParams(w=Param(data=1.0), s=State(data=0))  # missing name


def test_unexpected_kwarg():
    with pytest.raises(TypeError):
        WithParams(w=Param(data=1.0), s=State(data=0), name="foo", bogus=1)

import typing as t

# def test_cycle_detection():
#     class A(BaseItem):
#         ref: t.Union["B", None] = None  # type: ignore
#     class B(BaseItem):
#         ref: A | None = None
#     a = A()
#     b = B(ref=a)
#     a.ref = b
#     with pytest.raises(RuntimeError):
#         a.spec()


def test_load_state_strict_failure():
    wp = WithParams(w=Param(data=1.0), s=State(data=1), name="t")
    sd = {"w": 2.0, "missing": 9}
    with pytest.raises(KeyError):
        wp.load_state_dict(sd, recurse=False, strict=True)


def test_param_deduplication():
    shared_param = Param(data=3.0)
    class Proc(BaseModule):
        a: Param[float]
        b: Param[float]
    pr = Proc(a=shared_param, b=shared_param)
    assert len(list(pr.parameters())) == 1  # dedup by identity


def test_state_dict_recurse_flag():
    wp = WithParams(w=Param(data=1.0), s=State(data=2), name="n")
    n = Nested(inner=wp)
    assert n.state_dict(recurse=False) == {}


def test_parameters_recurse_false():
    wp = WithParams(w=Param(data=2.0), s=State(data=0), name="q")
    n = Nested(inner=wp)
    assert list(n.parameters(recurse=False)) == []

# # ------------------------------------------------------------
# #  Extra edge / negative tests (total >= 30 )
# # ------------------------------------------------------------

@pytest.mark.parametrize("val", [0, 1.2, "txt", [1, 2]])
def test_param_accepts_any_val(val):
    p = Param(data=val)
    assert p.data == val


# def test_shared_hashable():
#     s1 = Shared(val="abc")
#     s2 = Shared(val="abc")
#     assert hash(s1) == hash(s2)


def test_state_mutability():
    st = State(data=5)
    st.data += 1
    assert st.data == 6


def test_load_state_non_param_target_error():
    wp = WithParams(w=Param(data=1.0), s=State(data=1), name="a")
    with pytest.raises(KeyError):
        wp.load_state_dict({"name": "bad"}, recurse=False, strict=True)


def test_frozen_param_not_filtered_from_state():
    p = Param(data=3.3, training=False)
    c = WithParams(w=p, s=State(data=0), name="n")
    assert "w" in c.state_dict(train=True)

# # -----  corner cases for InitVar default / override ----------

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
    assert wp.spec().kind.endswith("WithParams")


def test_state_dict_train_flag():
    wp = WithParams(w=Param(data=2.2), s=State(data=0), name="m")
    assert "w" not in wp.state_dict(train=False)


def test_state_dict_runtime_flag():
    wp = WithParams(w=Param(data=1.0), s=State(data=5), name="r")
    assert "s" not in wp.state_dict(runtime=False)


def test_load_state_in_child():
    wp1 = WithParams(w=Param(data=4.0), s=State(data=6), name="x1")
    wp2 = WithParams(w=Param(data=0.0), s=State(data=0), name="x2")
    n = Nested(inner=wp2)
    n.load_state_dict({"inner.w": 4.0, "inner.s": 6})
    assert wp2.w.data == 4.0 and wp2.s.data == 6


def test_state_dict_flags_combination():
    class A(BaseModule):
        p: Param[float]
        s: State[int]
    a = A(p=Param(data=3.3), s=State(data=9))
    sd = a.state_dict(train=False, runtime=False)
    assert sd == {}

def test_state_dict_nested_keys():
    class Leaf(BaseModule):
        p: Param[float]
    class Root(BaseModule):
        leaf: Leaf
        b: Param[int]
    r = Root(leaf=Leaf(p=Param(data=1.0)), b=Param(data=2))
    sd = r.state_dict()
    assert set(sd.keys()) == {"leaf.p", "b"}

def test_state_dict_shared_exclusion():
    class A(BaseModule):
        sh: Shared[str]
    a = A(sh=Shared(data="foo"))
    sd = a.state_dict()
    assert "sh" not in sd

def test_state_dict_conflicting_keys():
    class Child(BaseModule):
        p: Param[float]
    class Parent(BaseModule):
        child1: Child
        child2: Child
    p = Parent(child1=Child(p=Param(data=1.0)), child2=Child(p=Param(data=2.0)))
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
        p: Param[float]
    a = A(p=Param(data=1.0))
    a.load_state_dict({}, strict=False)
    assert a.p.data == 1.0

# def test_load_state_dict_type_mismatch():
#     class A(BaseItem):
#         p: Param[float]
#     a = A(p=Param(data=1.0))
#     with pytest.raises(Exception):
#         a.load_state_dict({"p": "oops"}, strict=True)

def test_load_state_dict_shared_ignore_even_if_present():
    class A(BaseModule):
        p: Param[float]
        sh: Shared[str]
    a = A(p=Param(data=1.0), sh=Shared(data="init"))
    a.load_state_dict({"p":5.5, "sh":"should not overwrite"}, strict=False)
    assert a.p.data == 5.5 and a.sh.data == "init"


def test_parameters_train_only_true():
    p1 = Param(data=1.0, training=True)
    p2 = Param(data=2.0, training=False)
    class P(BaseModule):
        a: Param[float]
        b: Param[float]
    pr = P(a=p1, b=p2)
    assert list(pr.parameters(train_only=True)) == [p1]


def test_parameters_no_params():
    class P(BaseModule):
        x: int
    p = P(x=5)
    assert list(p.parameters()) == []

def test_parameters_deduplication():
    p = Param(data=1.0)
    class P(BaseModule):
        a: Param[float]
        b: Param[float]
    pr = P(a=p, b=p)
    assert list(pr.parameters()) == [p]

def test_parameters_nested():
    class Leaf(BaseModule):
        p: Param[float]
    class Branch(BaseModule):
        leaf: Leaf
        b: Param[int]
    pr = Branch(leaf=Leaf(p=Param(data=1.0)), b=Param(data=2))
    ps = list(pr.parameters())
    assert len(ps) == 2 and all(isinstance(p, Param) for p in ps)

def test_parameters_recurse_false():
    class Leaf(BaseModule):
        p: Param[float]
    class Branch(BaseModule):
        leaf: Leaf
        b: Param[int]
    pr = Branch(leaf=Leaf(p=Param(data=1.0)), b=Param(data=2))
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
        a: Shared[str]
        b: State[int]
        c: int
    pr = P(a=Shared(data="ref"), b=State(data=0), c=42)
    assert list(pr.parameters()) == []

def test_parameters_train_only_none():
    p1 = Param(data=1.0, training=True)
    p2 = Param(data=2.0, training=False)
    class P(BaseModule):
        a: Param[float]
        b: Param[float]
    pr = P(a=p1, b=p2)
    ps = list(pr.parameters(train_only=None))
    assert set(ps) == {p1, p2}

# # ------ identity vs value equality for Shared ------------------

def test_shared_identity_not_dedup():
    s1 = Shared(data="conf")
    s2 = Shared(data="conf")
    class P(BaseModule):
        a: Shared[str]
        b: Shared[str]
    pr = P(a=s1, b=s2)
    # state_dict must still omit both
    assert pr.state_dict() == {}



# ----------  Happy-path: basic save  ----------
def test_checkpoint_save_module_creates_file(tmp_path):
    """Positive • file is physically created and contains valid JSON."""
    class Leaf(BaseModule):
        w: Param[int]

    leaf = Leaf(w=Param(1))
    path = tmp_path / "leaf.json"

    Checkpoint.save_module(leaf, path)
    raw = path.read_text()

    assert path.exists()          # file written
    json.loads(raw)               # raises if not valid JSON


# ----------  Happy-path: load → exact round-trip ----------
def test_checkpoint_load_roundtrip(tmp_path):
    """Positive • Checkpoint.load() reproduces the exact spec & state."""
    class Leaf(BaseModule):
        w: Param[int]

    leaf = Leaf(w=Param(3))
    path = tmp_path / "leaf.json"
    Checkpoint.save_module(leaf, path)

    ckpt = Checkpoint.load(path)

    assert ckpt.state_dict == leaf.state_dict(recurse=True, train=True, runtime=True)
    assert ckpt.spec.kind == leaf.spec().kind


# ----------  Happy-path: load_module reconstructs model ----------
def test_checkpoint_load_module_restores_state(tmp_path):
    """Positive • load_module returns an equivalent, fully initialised module."""
    class Leaf(BaseModule):
        w: Param[int]

    original = Leaf(w=Param(7))
    path = tmp_path / "leaf.json"
    Checkpoint.save_module(original, path)

    rebuilt = Checkpoint.load_module(path)

    assert isinstance(rebuilt, Leaf)
    assert rebuilt.w.data == 7


# ----------  Happy-path: shared ref-names deduplicated ----------
def test_checkpoint_shared_objects_deduplicated(tmp_path):
    """Positive • Same ref_name inside spec becomes the *same* object."""
    class Inner(BaseModule):
        val: Param[int]

    shared_inner = Shared(ref_name="X", data=Inner(val=Param(5)))

    class Outer(BaseModule):
        a: Inner
        b: Inner

    model = Outer(a=shared_inner, b=shared_inner)
    path = tmp_path / "outer.json"
    Checkpoint.save_module(model, path)

    ctx = BuildContext()
    rebuilt = Checkpoint.load_module(path, context=ctx)

    assert rebuilt.a is rebuilt.b            # identity dedup
    assert rebuilt.a.val.data == 5


# ----------  Error-path: missing file ----------
def test_checkpoint_load_missing_file_raises(tmp_path):
    """Negative • Loading a non-existent file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        Checkpoint.load(tmp_path / "nope.json")


# ----------  Error-path: corrupt JSON ----------
def test_checkpoint_load_corrupt_json_raises(tmp_path):
    """Negative • Invalid JSON content raises ValueError (Pydantic)."""
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not json!")
    with pytest.raises(ValueError):
        Checkpoint.load(bad_file)
python
コピーする
編集する
# test_base4_misc.py
# -------------------------------------------------
# Small, focused tests for the remaining public helpers.
# -------------------------------------------------
import pytest
from _base4 import Param, State, BaseModule


# ----------  train()/eval() cascade ----------
def test_eval_cascades_to_children():
    """Positive • eval() sets training=False on every Param."""
    class Leaf(BaseModule):
        p: Param[int]

    class Root(BaseModule):
        l1: Leaf
        l2: Leaf

    root = Root(l1=Leaf(p=Param(1)), l2=Leaf(p=Param(2)))
    root.eval()
    assert all(not p.training for p in root.parameters(recurse=True, train=None))


def test_train_cascades_to_children():
    """Positive • train() resets training=True on every Param."""
    class Leaf(BaseModule):
        p: Param[int]

    class Root(BaseModule):
        l1: Leaf
        l2: Leaf

    root = Root(l1=Leaf(p=Param(1)), l2=Leaf(p=Param(2)))
    root.eval()
    root.train()                      # switch back
    assert all(p.training for p in root.parameters(recurse=True, train=None))


# ----------  apply() with filter_type ----------
def test_apply_filters_by_type():
    """Positive • apply() visits only objects of filter_type."""
    calls = []

    class Leaf(BaseModule):
        w: Param[int]

    root = Leaf(w=Param(0))

    root.apply(lambda x: calls.append(type(x).__name__), filter_type=Param)
    assert calls == ["Param"]


# ----------  named_modules() dotted prefixes ----------
def test_named_modules_keys_are_correct():
    """Positive • named_modules() returns expected dotted names."""
    class Leaf(BaseModule):
        x: Param[int]

    class Branch(BaseModule):
        left: Leaf
        right: Leaf

    tree = Branch(left=Leaf(x=Param(1)), right=Leaf(x=Param(2)))
    names = dict(tree.named_modules())

    assert set(names.keys()) == {"", "left", "right"}


# ----------  state_dict() recurse=False ----------
def test_state_dict_nonrecurse_returns_empty():
    """Edge • recurse=False on parent with only submodules returns empty dict."""
    class Leaf(BaseModule):
        s: State[int]

    class Top(BaseModule):
        child: Leaf

    t = Top(child=Leaf(s=State(9)))
    flat = t.state_dict(recurse=False, runtime=False, train=True)

    assert flat == {}
