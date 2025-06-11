# test_itemlist.py
import pytest
from dachi.core2._base4 import BaseModule, Param, State, BaseSpec, registry
from dachi.core2._structs import ModuleList
from dataclasses import InitVar

# ---------------------- helper child class -------------------------

@registry.register()
class Leaf(BaseModule):
    w: InitVar[float]
    s: InitVar[int]

    def __post_init__(self, w: float, s: int):

        self.w = Param(w)
        self.s = State(s)

def make_leaf(val=1.0, step=0):
    return Leaf(w=val, s=step)

# ------------------------ __init__ tests ---------------------------
def test_itemlist_init_happy():
    lst = ModuleList(items=[make_leaf(), make_leaf()])
    assert len(lst) == 2
    # registry should have keys "0", "1"
    assert set(lst._modules.keys()) == {"0", "1"}

def test_itemlist_init_single_element():
    lst = ModuleList(items=[make_leaf()])
    assert len(lst) == 1 and "0" in lst._modules

def test_itemlist_init_empty_error():
    lst = ModuleList(items=[])
    assert len(lst) == 0

def test_itemlist_init_type_error():
    with pytest.raises(TypeError):
        ModuleList(items=[42])             # not BaseProcess

# # # ------------------------ append tests -----------------------------
def test_itemlist_append():
    lst = ModuleList(items=[make_leaf()])
    new_leaf = make_leaf(2.0)
    lst.append(new_leaf)
    assert len(lst) == 2
    assert lst[1] is new_leaf
    assert "1" in lst._modules

def test_itemlist_append_type_error():
    lst = ModuleList(items=[make_leaf()])
    with pytest.raises(TypeError):
        lst.append(123)

# # # ------------------------ setitem tests ----------------------------
def test_itemlist_setitem():
    lst = ModuleList(items=[make_leaf(), make_leaf()])
    repl = make_leaf(9.9)
    lst[1] = repl
    assert lst[1].w.data == 9.9
    assert lst._modules["1"] is repl

def test_itemlist_setitem_type_error():
    lst = ModuleList(items=[make_leaf()])
    with pytest.raises(TypeError):
        lst[0] = "oops"

def test_itemlist_setitem_index_error():
    lst = ModuleList(items=[make_leaf()])
    with pytest.raises(IndexError):
        lst[5] = make_leaf()

# # # ------------------------ spec / schema ----------------------------
def test_itemlist_spec_roundtrip():
    lst = ModuleList(items=[make_leaf(3.3)])
    spec = lst.spec()

    assert isinstance(spec, BaseSpec)
    assert spec.items[0].w == 3.3

# # ------------------------ state_dict --------------------------------
def test_itemlist_state_dict_flags():
    l1 = make_leaf(1.0, 10)
    l2 = make_leaf(2.0, 20)
    lst = ModuleList(items=[l1, l2])
    sd_train_only = lst.state_dict(runtime=False)
    print(sd_train_only)
    assert sd_train_only == {"0.w":1.0, "1.w":2.0}

def test_itemlist_state_dict_roundtrip():
    lst1 = ModuleList(items=[make_leaf(5), make_leaf(6)])
    sd = lst1.state_dict()
    lst2 = ModuleList(items=[make_leaf(0), make_leaf(0)])
    lst2.load_state_dict(sd, strict=True)
    assert lst2.state_dict() == sd

# # ------------------------ load_state_dict ---------------------------
def test_itemlist_load_state_dict_strict_len_mismatch():
    lst = ModuleList(items=[make_leaf()])
    with pytest.raises(KeyError):
        lst.load_state_dict({
            "items": [{"w":1.0}, {"w": 2.0}]
        }, strict=True)

def test_itemlist_load_state_dict_non_strict():
    lst = ModuleList(items=[make_leaf()])
    lst.load_state_dict({}, strict=False)   # no error

def test_itemlist_load_state_dict_type_error():
    lst = ModuleList(items=[make_leaf()])
    with pytest.raises(TypeError):
        lst.load_state_dict("not a list", strict=False)

# # ------------------------ parameters dedup --------------------------
def test_itemlist_parameters_dedup():
    leaf1 = Leaf(w=7, s=0)
    leaf2 = Leaf(w=7, s=1)
    leaf2.w = leaf1.w
    lst = ModuleList(items=[leaf1, leaf2])
    params = list(lst.parameters())
    assert params[0] == leaf2.w   # deduplicated by id

# # ------------------------ named_parameters / named_states ----------
def test_itemlist_named_parameters_states():
    lst = ModuleList(items=[make_leaf()])
    p_names = dict(lst.named_parameters()).keys()
    s_names = dict(lst.named_states()).keys()
    assert p_names == {"0.w"} and s_names == {"0.s"}

# # ---------- Positive: mixed‑type init should fail on 2nd element ----------

def test_modulelist_mixed_type_init_error():
    with pytest.raises(TypeError):
        ModuleList(items=[Leaf(w=1, s=0), 123])


# # ---------- Positive: iteration preserves insertion order ----------

def test_modulelist_iter_order():
    m1, m2 = Leaf(w=1, s=0), Leaf(w=2, s=0)
    lst = ModuleList(items=[m1, m2])
    assert list(iter(lst)) == [m1, m2]


# # ---------- Edge: negative index access behaves like list ----------

def test_modulelist_negative_index_getitem():
    m1, m2 = Leaf(
        w=1, s=0), Leaf(w=2, s=0)
    lst = ModuleList(items=[m1, m2])
    assert lst[-1] is m2


# # ---------- Negative: calling schema() on raw class raises ----------

def test_modulelist_schema_on_raw_class_raises():
    schema = ModuleList.schema()
    assert issubclass(schema, BaseSpec)


# # ---------- Positive: spec → from_spec round‑trip ----------

def test_modulelist_from_spec_roundtrip():
    m1, m2 = Leaf(w=3, s=3), Leaf(w=4, s=4)
    lst = ModuleList(items=[m1, m2])
    spec = lst.spec(to_dict=False)
    rebuilt = ModuleList.from_spec(spec)
    assert len(rebuilt) == 2 and rebuilt[0].w.data == 3


# # ---------- Positive: __setitem__ removes old attribute ----------

def test_modulelist_setitem_removes_old_attr():
    lst = ModuleList(
        items=[Leaf(w=Param(1), s=State(0))]
    )
    assert hasattr(lst, "0")
    lst[0] = Leaf(w=9, s=9)
    # new attr re‑registered under same name, old attr gone implicitly
    assert getattr(lst, "0").w.data == 9


# # ---------- Edge: append after replacement gets unique name ----------

def test_modulelist_append_generates_monotonic_names():
    lst = ModuleList(items=[])
    lst.append(Leaf(w=1, s=0))  # name "0"
    lst.append(Leaf(w=1, s=0))  # name "1"
    lst[0] = Leaf(w=3, s=0)     # replaces index 0 (still name "0")
    lst.append(Leaf(w=4, s=0))  # should get name "2", not "1" again
    assert hasattr(lst, "2")


# # ---------- Positive: train / eval cascade across children ----------

# def test_modulelist_train_eval_cascade():
#     lst = ModuleList([
#         Leaf(w=1, s=0),
#         Leaf(w=2, s=0),
#     ])
#     lst.eval()
#     assert all(not p.training for p in lst.parameters(recurse=True, train=None))
#     lst.train()
#     assert all(p.training for p in lst.parameters(recurse=True, train=None))


# # ---------- Edge: duplicate child references ----------

def test_modulelist_duplicate_child_objects():
    leaf = Leaf(w=5, s=5)
    lst = ModuleList(items=[leaf, leaf])
    # parameters() should deduplicate by identity
    assert len(list(lst.parameters(recurse=True))) == 1
    # state_dict should still record two entries (list semantics)
    state_list = lst.state_dict(recurse=True, train=True, runtime=True)
    print(list(state_list.keys()))
    assert state_list['0.w'] == 5 and state_list['1.w'] == 5

