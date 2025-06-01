# test_itemlist.py
import pytest
from dachi.core2._base4 import BaseModule, Param, State, BaseSpec
from dachi.core2._structs import ModuleList

# ---------------------- helper child class -------------------------
class Leaf(BaseModule):
    w: Param[float]
    s: State[int]

def make_leaf(val=1.0, step=0):
    return Leaf(w=Param(data=val), s=State(data=step))

# ------------------------ __init__ tests ---------------------------
def test_itemlist_init_happy():
    lst = ModuleList([make_leaf(), make_leaf()])
    assert len(lst) == 2
    # registry should have keys "0", "1"
    assert set(lst._modules.keys()) == {"0", "1"}

def test_itemlist_init_single_element():
    lst = ModuleList([make_leaf()])
    assert len(lst) == 1 and "0" in lst._modules

def test_itemlist_init_empty_error():
    with pytest.raises(ValueError):
        ModuleList([])

def test_itemlist_init_type_error():
    with pytest.raises(TypeError):
        ModuleList([42])             # not BaseProcess

# # ------------------------ append tests -----------------------------
def test_itemlist_append():
    lst = ModuleList([make_leaf()])
    new_leaf = make_leaf(2.0)
    lst.append(new_leaf)
    assert len(lst) == 2
    assert lst[1] is new_leaf
    assert "1" in lst._modules

def test_itemlist_append_type_error():
    lst = ModuleList([make_leaf()])
    with pytest.raises(TypeError):
        lst.append(123)

# # ------------------------ setitem tests ----------------------------
def test_itemlist_setitem():
    lst = ModuleList([make_leaf(), make_leaf()])
    repl = make_leaf(9.9)
    lst[1] = repl
    assert lst[1].w.data == 9.9
    assert lst._modules["1"] is repl

def test_itemlist_setitem_type_error():
    lst = ModuleList([make_leaf()])
    with pytest.raises(TypeError):
        lst[0] = "oops"

def test_itemlist_setitem_index_error():
    lst = ModuleList([make_leaf()])
    with pytest.raises(IndexError):
        lst[5] = make_leaf()

# # ------------------------ spec / schema ----------------------------
def test_itemlist_spec_roundtrip():
    lst = ModuleList([make_leaf(3.3)])
    spec = lst.spec()
    print(spec)
    assert isinstance(spec, BaseSpec) and spec.modules[0].w.data == 3.3

# ------------------------ state_dict --------------------------------
def test_itemlist_state_dict_flags():
    l1 = make_leaf(1.0, 10)
    l2 = make_leaf(2.0, 20)
    lst = ModuleList([l1, l2])
    sd_train_only = lst.state_dict(runtime=False)
    assert sd_train_only == [{"w":1.0}, {"w":2.0}]

def test_itemlist_state_dict_roundtrip():
    lst1 = ModuleList([make_leaf(5), make_leaf(6)])
    sd = lst1.state_dict()
    lst2 = ModuleList([make_leaf(0), make_leaf(0)])
    lst2.load_state_dict(sd, strict=True)
    assert lst2.state_dict() == sd

# ------------------------ load_state_dict ---------------------------
def test_itemlist_load_state_dict_strict_len_mismatch():
    lst = ModuleList([make_leaf()])
    with pytest.raises(KeyError):
        lst.load_state_dict([{"w":1.0}, {"w":2.0}], strict=True)

def test_itemlist_load_state_dict_non_strict():
    lst = ModuleList([make_leaf()])
    lst.load_state_dict([], strict=False)   # no error

def test_itemlist_load_state_dict_type_error():
    lst = ModuleList([make_leaf()])
    with pytest.raises(TypeError):
        lst.load_state_dict("not a list", strict=False)

# ------------------------ parameters dedup --------------------------
def test_itemlist_parameters_dedup():
    shared_param = Param(data=7)
    leaf1 = Leaf(w=shared_param, s=State(data=0))
    leaf2 = Leaf(w=shared_param, s=State(data=1))
    lst = ModuleList([leaf1, leaf2])
    params = list(lst.parameters())
    assert params == [shared_param]   # deduplicated by id

# ------------------------ named_parameters / named_states ----------
def test_itemlist_named_parameters_states():
    lst = ModuleList([make_leaf()])
    p_names = dict(lst.named_parameters()).keys()
    s_names = dict(lst.named_states()).keys()
    assert p_names == {"0.w"} and s_names == {"0.s"}

# ---------- Positive: mixed‑type init should fail on 2nd element ----------

def test_modulelist_mixed_type_init_error():
    with pytest.raises(TypeError):
        ModuleList([Leaf(w=Param(1), s=State(0)), 123])


# ---------- Positive: iteration preserves insertion order ----------

def test_modulelist_iter_order():
    m1, m2 = Leaf(w=Param(1), s=State(0)), Leaf(w=Param(2), s=State(0))
    lst = ModuleList([m1, m2])
    assert list(iter(lst)) == [m1, m2]


# ---------- Edge: negative index access behaves like list ----------

def test_modulelist_negative_index_getitem():
    m1, m2 = Leaf(w=Param(1), s=State(0)), Leaf(w=Param(2), s=State(0))
    lst = ModuleList([m1, m2])
    assert lst[-1] is m2


# ---------- Negative: calling schema() on raw class raises ----------

def test_modulelist_schema_on_raw_class_raises():
    with pytest.raises(TypeError):
        ModuleList.schema()


# ---------- Positive: spec → from_spec round‑trip ----------

def test_modulelist_from_spec_roundtrip():
    m1, m2 = Leaf(w=Param(3), s=State(3)), Leaf(w=Param(4), s=State(4))
    lst = ModuleList([m1, m2])
    spec = lst.spec(to_dict=False)
    rebuilt = ModuleList.from_spec(spec)
    assert len(rebuilt) == 2 and rebuilt[0].w.data == 3


# ---------- Positive: __setitem__ removes old attribute ----------

def test_modulelist_setitem_removes_old_attr():
    lst = ModuleList([Leaf(w=Param(1), s=State(0))])
    assert hasattr(lst, "0")
    lst[0] = Leaf(w=Param(9), s=State(9))
    # new attr re‑registered under same name, old attr gone implicitly
    assert getattr(lst, "0").w.data == 9


# ---------- Edge: append after replacement gets unique name ----------

def test_modulelist_append_generates_monotonic_names():
    lst = ModuleList()
    lst.append(Leaf(w=Param(1), s=State(0)))  # name "0"
    lst.append(Leaf(w=Param(2), s=State(0)))  # name "1"
    lst[0] = Leaf(w=Param(3), s=State(0))     # replaces index 0 (still name "0")
    lst.append(Leaf(w=Param(4), s=State(0)))  # should get name "2", not "1" again
    assert hasattr(lst, "2")


# ---------- Positive: train / eval cascade across children ----------

def test_modulelist_train_eval_cascade():
    lst = ModuleList([
        Leaf(w=Param(1), s=State(0)),
        Leaf(w=Param(2), s=State(0)),
    ])
    lst.eval()
    assert all(not p.training for p in lst.parameters(recurse=True, train=None))
    lst.train()
    assert all(p.training for p in lst.parameters(recurse=True, train=None))


# ---------- Edge: duplicate child references ----------

def test_modulelist_duplicate_child_objects():
    leaf = Leaf(w=Param(5), s=State(5))
    lst = ModuleList([leaf, leaf])
    # parameters() should deduplicate by identity
    assert len(list(lst.parameters(recurse=True, dedup=True))) == 1
    # state_dict should still record two entries (list semantics)
    state_list = lst.state_dict(recurse=True, train=True, runtime=True)
    assert state_list[0]["w"].data == 5 and state_list[1]["w"].data == 5

