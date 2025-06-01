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
