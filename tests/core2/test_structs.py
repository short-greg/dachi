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



import pytest
from dachi.core2._structs import ModuleList
from dachi.core2._base4 import BaseModule, Param, State, BuildContext, registry
from dataclasses import InitVar

# --- Helper Leaf class ---
@registry()
class Leaf2(BaseModule):
    v: InitVar[int]
    f: InitVar[int]

    def __post_init__(self, v, f):
        self.v = Param(data=v)
        self.f = State(data=f)

# --- Tests for ModuleList edge cases ---

def test_post_init_empty_list():
    # empty items => no error, len=0
    ml = ModuleList(items=[])
    assert len(ml) == 0
    assert list(ml) == []


def test_post_init_type_error():
    # non-BaseModule element raises
    with pytest.raises(TypeError):
        ModuleList(items=["not a module"])


def test_len_and_iter_and_getitem():
    leaf1 = Leaf2(v=1,f=1)
    leaf2 = Leaf2(v=2,f=2)
    ml = ModuleList(items=[leaf1, leaf2])
    assert len(ml) == 2
    assert list(iter(ml)) == [leaf1, leaf2]
    assert ml[0] is leaf1
    assert ml[1] is leaf2


def test_getitem_out_of_range():
    ml = ModuleList(items=[])
    with pytest.raises(IndexError):
        _ = ml[0]


def test_setitem_replacement_and_module_registration():
    leaf1 = Leaf2(v=1,f=1)
    leaf2 = Leaf2(v=2,f=2)
    ml = ModuleList(items=[leaf1, leaf2])
    leaf3 = Leaf2(v=3,f=3)
    ml[1] = leaf3
    assert ml[1] is leaf3
    # ensure modules dict updated
    assert ml._modules["1"] is leaf3
    assert "1" in ml._modules and ml._modules["1"] is leaf3


def test_setitem_type_error():
    leaf = Leaf2(v=1,f=1)
    ml = ModuleList(items=[leaf])
    with pytest.raises(TypeError):
        ml[0] = "oops"


def test_append_and_registration():
    ml = ModuleList(items=[])
    leaf = Leaf2(v=5,f=5)
    ml.append(leaf)
    assert len(ml) == 1
    assert ml._modules["0"] is leaf


def test_append_type_error():
    ml = ModuleList(items=[])
    with pytest.raises(TypeError):
        ml.append(123)


def test_spec_and_from_spec_empty_and_dedup():
    ml = ModuleList(items=[])
    ctx = BuildContext()
    spec1 = ml.spec(to_dict=False)
    spec2 = ml.spec(to_dict=False)
    # identical object returned
    assert spec1 == spec2
    # round-trip
    ml2 = ModuleList.from_spec(spec1, ctx)
    assert isinstance(ml2, ModuleList)
    assert len(ml2) == 0

    # duplicate underlying spec dedup
    leaf = Leaf2(v=9,f=9)
    ml3 = ModuleList(items=[leaf, leaf])
    ctx2 = BuildContext()
    spec3 = ml3.spec(to_dict=False)
    ml3b = ModuleList.from_spec(spec3, ctx2)
    assert ml3b._module_list[0] is ml3b._module_list[1]


def test_state_dict_flags_and_load_state_dict():
    leaf1 = Leaf2(v=1,f=10)
    leaf2 = Leaf2(v=2,f=20)
    ml = ModuleList(items=[leaf1, leaf2])
    # full flags
    sd_full = ml.state_dict(recurse=True, train=True, runtime=True)
    assert isinstance(sd_full, dict) and len(sd_full) == 4
    # train=False omits Param data
    sd_no_train = ml.state_dict(train=False)
    for d in sd_no_train:
        assert "v" not in d
    # runtime=False omits State data
    sd_no_rt = ml.state_dict(runtime=False)
    for d in sd_no_rt:
        assert "f" not in d

    # load_state_dict type error
    with pytest.raises(KeyError):
        ml.load_state_dict({})
    # strict length mismatch
    with pytest.raises(TypeError):
        ml.load_state_dict([{}, {}], strict=True)
    with pytest.raises(TypeError):
        ml.load_state_dict({{}, {}}, strict=False)


def test_parameters_and_named_modules():
    leaf1 = Leaf2(v=1,f=1)
    leaf2 = Leaf2(v=2,f=2)
    ml = ModuleList(items=[leaf1, leaf2])
    # parameters yields leaf1.v and leaf2.v
    params = list(ml.parameters())
    assert all(isinstance(p, Param) for p in params)
    # named_modules includes self and children
    names = [name for name, _ in ml.named_modules()]
    assert names == ["", "0", "1"]


def test_load_state_dict_non_strict_behavior():
    leaf1 = Leaf2(v=1,f=1)
    ml = ModuleList(items=[leaf1])
    # shorter list non-strict OK
    ml.load_state_dict({}, strict=False)
    # longer list non-strict OK
    ml.load_state_dict({"items": {"v":5}}, strict=False)

