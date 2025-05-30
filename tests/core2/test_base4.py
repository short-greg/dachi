import pytest
from dachi.core2._base4 import BaseItem, Param, State, Shared, BaseSpec
from dataclasses import InitVar

# ------------------------------------------------------------
#  Helper process classes for tests
# ------------------------------------------------------------

class Leaf(BaseItem):
    value: int

class WithParams(BaseItem):
    w: Param[float]
    s: State[int]
    name: str

class Nested(BaseItem):
    inner: WithParams
    extra: int = 5

class InitVarProc(BaseItem):
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
    class Proc(BaseItem):
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
        wp.load_state_dict(sd, recurse=False)


def test_param_deduplication():
    shared_param = Param(data=3.0)
    class Proc(BaseItem):
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
        wp.load_state_dict({"name": "bad"}, recurse=False)


def test_frozen_param_not_filtered_from_state():
    p = Param(data=3.3, training=False)
    c = WithParams(w=p, s=State(data=0), name="n")
    assert "w" in c.state_dict(train=True)

# # -----  corner cases for InitVar default / override ----------

def test_initvar_default_used():
    class P(BaseItem):
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
    class WithParams(BaseItem):
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


def test_parameters_train_only_true():
    p1 = Param(data=1.0, training=True)
    p2 = Param(data=2.0, training=False)
    class P(BaseItem):
        a: Param[float]
        b: Param[float]
    pr = P(a=p1, b=p2)
    assert list(pr.parameters(train_only=True)) == [p1]

# # ------ identity vs value equality for Shared ------------------

def test_shared_identity_not_dedup():
    s1 = Shared(data="conf")
    s2 = Shared(data="conf")
    class P(BaseItem):
        a: Shared[str]
        b: Shared[str]
    pr = P(a=s1, b=s2)
    # state_dict must still omit both
    assert pr.state_dict() == {}

