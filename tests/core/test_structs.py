# test_itemlist.py
import pytest
from dataclasses import InitVar

# local
from dachi.core._base import BaseModule, Param, Attr, BaseSpec, registry
from dachi.core._structs import ModuleList, ModuleDict


@registry.register()
class Leaf3(BaseModule):
    w: InitVar[float]
    s: InitVar[int]

    def __post_init__(self, w: float, s: int):

        self.w = Param(w)
        self.s = Attr(s)

# --- Helper Leaf class ---
@registry()
class Leaf4(BaseModule):
    v: InitVar[int]
    f: InitVar[int]

    def __post_init__(self, v, f):
        self.v = Param(data=v)
        self.f = Attr(data=f)


def make_leaf(val=1.0, step=0):
    return Leaf3(w=val, s=step)


class TestModuleList:
    def test_itemlist_init_happy(self):
        lst = ModuleList(items=[make_leaf(), make_leaf()])
        assert len(lst) == 2
        # registry should have keys "0", "1"
        assert set(lst._modules.keys()) == {"0", "1"}

    def test_itemlist_init_single_element(self):
        lst = ModuleList(items=[make_leaf()])
        assert len(lst) == 1 and "0" in lst._modules

    def test_itemlist_init_empty_error(self):
        lst = ModuleList(items=[])
        assert len(lst) == 0

    def test_itemlist_init_type_error(self):
        with pytest.raises(TypeError):
            ModuleList(items=[42])             # not BaseProcess

    def test_itemlist_append(self):
        lst = ModuleList(items=[make_leaf()])
        new_leaf = make_leaf(2.0)
        lst.append(new_leaf)
        assert len(lst) == 2
        assert lst[1] is new_leaf
        assert "1" in lst._modules

    def test_itemlist_append_type_error(self):
        lst = ModuleList(items=[make_leaf()])
        with pytest.raises(TypeError):
            lst.append(123)

    def test_itemlist_setitem(self):
        lst = ModuleList(items=[make_leaf(), make_leaf()])
        repl = make_leaf(9.9)
        lst[1] = repl
        assert lst[1].w.data == 9.9
        assert lst._modules["1"] is repl

    def test_itemlist_setitem_type_error(self):
        lst = ModuleList(items=[make_leaf()])
        with pytest.raises(TypeError):
            lst[0] = "oops"

    def test_itemlist_setitem_index_error(self):
        lst = ModuleList(items=[make_leaf()])
        with pytest.raises(IndexError):
            lst[5] = make_leaf()

    def test_itemlist_spec_roundtrip(self):
        lst = ModuleList(items=[make_leaf(3.3)])
        spec = lst.spec()

        assert isinstance(spec, BaseSpec)
        assert spec.data[0].w == 3.3

    def test_itemlist_state_dict_flags(self):
        l1 = make_leaf(1.0, 10)
        l2 = make_leaf(2.0, 20)
        lst = ModuleList(items=[l1, l2])
        sd_train_only = lst.state_dict(runtime=False)
        print(sd_train_only)
        assert sd_train_only == {"0.w":1.0, "1.w":2.0}

    def test_itemlist_state_dict_roundtrip(self):
        lst1 = ModuleList(items=[make_leaf(5), make_leaf(6)])
        sd = lst1.state_dict()
        lst2 = ModuleList(items=[make_leaf(0), make_leaf(0)])
        lst2.load_state_dict(sd, strict=True)
        assert lst2.state_dict() == sd

    def test_itemlist_load_state_dict_strict_len_mismatch(self):
        lst = ModuleList(items=[make_leaf()])
        with pytest.raises(KeyError):
            lst.load_state_dict({
                "items": [{"w":1.0}, {"w": 2.0}]
            }, strict=True)

    def test_itemlist_load_state_dict_non_strict(self):
        lst = ModuleList(items=[make_leaf()])
        lst.load_state_dict({}, strict=False)   # no error

    def test_itemlist_load_state_dict_type_error(self):
        lst = ModuleList(items=[make_leaf()])
        with pytest.raises(TypeError):
            lst.load_state_dict("not a list", strict=False)

#     # # ------------------------ parameters dedup --------------------------
    def test_itemlist_parameters_dedup(self):
        leaf1 = Leaf3(w=7, s=0)
        leaf2 = Leaf3(w=7, s=1)
        leaf2.w = leaf1.w
        lst = ModuleList(items=[leaf1, leaf2])
        params = list(lst.parameters())
        assert params[0] == leaf2.w   # deduplicated by id

#     # # ------------------------ named_parameters / named_states ----------
    def test_itemlist_named_parameters_states(self):
        lst = ModuleList(items=[make_leaf()])
        p_names = dict(lst.named_parameters()).keys()
        s_names = dict(lst.named_states()).keys()
        assert p_names == {"0.w"} and s_names == {"0.s"}

#     # # ---------- Positive: mixed‑type init should fail on 2nd element ----------

    def test_modulelist_mixed_type_init_error(self):
        with pytest.raises(TypeError):
            ModuleList(items=[Leaf3(w=1, s=0), 123])


    def test_modulelist_iter_order(self):
        m1, m2 = Leaf3(w=1, s=0), Leaf3(w=2, s=0)
        lst = ModuleList(items=[m1, m2])
        assert list(iter(lst)) == [m1, m2]


#     # # ---------- Edge: negative index access behaves like list ----------

    def test_modulelist_negative_index_getitem(self):
        m1, m2 = Leaf3(
            w=1, s=0), Leaf3(w=2, s=0)
        lst = ModuleList(items=[m1, m2])
        assert lst[-1] is m2


    # # ---------- Negative: calling schema() on raw class raises ----------

    def test_modulelist_schema_on_raw_class_raises(self):
        schema = ModuleList.schema()
        assert issubclass(schema, BaseSpec)


#     # # ---------- Positive: spec → from_spec round‑trip ----------

    def test_modulelist_from_spec_roundtrip(self):
        m1, m2 = Leaf3(w=3, s=3), Leaf3(w=4, s=4)
        lst = ModuleList(items=[m1, m2])
        spec = lst.spec(to_dict=False)
        rebuilt = ModuleList.from_spec(spec)
        assert len(rebuilt) == 2 and rebuilt[0].w.data == 3


#     # # ---------- Positive: __setitem__ removes old attribute ----------

    def test_modulelist_setitem_removes_old_attr(self):
        lst = ModuleList(
            items=[Leaf3(w=Param(1), s=Attr(0))]
        )
        assert hasattr(lst, "0")
        lst[0] = Leaf3(w=9, s=9)
        # new attr re‑registered under same name, old attr gone implicitly
        assert getattr(lst, "0").w.data == 9


#     # # ---------- Edge: append after replacement gets unique name ----------

    def test_modulelist_append_generates_monotonic_names(self):
        lst = ModuleList(items=[])
        lst.append(Leaf3(w=1, s=0))  # name "0"
        lst.append(Leaf3(w=1, s=0))  # name "1"
        lst[0] = Leaf3(w=3, s=0)     # replaces index 0 (still name "0")
        lst.append(Leaf3(w=4, s=0))  # should get name "2", not "1" again
        assert hasattr(lst, "2")


#     # # ---------- Positive: train / eval cascade across children ----------

#     # def test_modulelist_train_eval_cascade():
#     #     lst = ModuleList([
#     #         Leaf(w=1, s=0),
#     #         Leaf(w=2, s=0),
#     #     ])
#     #     lst.eval()
#     #     assert all(not p.training for p in lst.parameters(recurse=True, train=None))
#     #     lst.train()
#     #     assert all(p.training for p in lst.parameters(recurse=True, train=None))


    def test_modulelist_duplicate_child_objects(self):
        leaf = Leaf3(w=5, s=5)
        lst = ModuleList(items=[leaf, leaf])
        # parameters() should deduplicate by identity
        assert len(list(lst.parameters(recurse=True))) == 1
        # state_dict should still record two entries (list semantics)
        state_list = lst.state_dict(recurse=True, train=True, runtime=True)
        print(list(state_list.keys()))
        assert state_list['0.w'] == 5 and state_list['1.w'] == 5



#     # --- Tests for ModuleList edge cases ---

    def test_post_init_empty_list(self):
        # empty items => no error, len=0
        ml = ModuleList(items=[])
        assert len(ml) == 0
        assert list(ml) == []


    def test_post_init_type_error(self):
        # non-BaseModule element raises
        with pytest.raises(TypeError):
            ModuleList(items=["not a module"])


    def test_len_and_iter_and_getitem(self):
        leaf1 = Leaf4(v=1,f=1)
        leaf2 = Leaf4(v=2,f=2)
        ml = ModuleList(items=[leaf1, leaf2])
        assert len(ml) == 2
        assert list(iter(ml)) == [leaf1, leaf2]
        assert ml[0] is leaf1
        assert ml[1] is leaf2


    def test_getitem_out_of_range(self):
        ml = ModuleList(items=[])
        with pytest.raises(IndexError):
            _ = ml[0]


    def test_setitem_replacement_and_module_registration(self):
        leaf1 = Leaf4(v=1,f=1)
        leaf2 = Leaf4(v=2,f=2)
        ml = ModuleList(items=[leaf1, leaf2])
        leaf3 = Leaf4(v=3,f=3)
        ml[1] = leaf3
        assert ml[1] is leaf3
        # ensure modules dict updated
        assert ml._modules["1"] is leaf3
        assert "1" in ml._modules and ml._modules["1"] is leaf3


    def test_setitem_type_error(self):
        leaf = Leaf4(v=1,f=1)
        ml = ModuleList(items=[leaf])
        with pytest.raises(TypeError):
            ml[0] = "oops"


    def test_append_and_registration(self):
        ml = ModuleList(items=[])
        leaf = Leaf4(v=5,f=5)
        ml.append(leaf)
        assert len(ml) == 1
        assert ml._modules["0"] is leaf


    def test_append_type_error(self):
        ml = ModuleList(items=[])
        with pytest.raises(TypeError):
            ml.append(123)


    def test_spec_and_from_spec_empty_and_dedup(self):
        ml = ModuleList(items=[])
        ctx = dict()
        spec1 = ml.spec(to_dict=False)
        spec2 = ml.spec(to_dict=False)
        # identical object returned
        assert spec1 == spec2
        # round-trip
        ml2 = ModuleList.from_spec(spec1, ctx)
        assert isinstance(ml2, ModuleList)
        assert len(ml2) == 0

        # duplicate underlying spec dedup
        leaf = Leaf4(v=9,f=9)
        ml3 = ModuleList(items=[leaf, leaf])
        ctx2 = dict()
        spec3 = ml3.spec(to_dict=False)
        ml3b = ModuleList.from_spec(spec3, ctx2)
        assert ml3b._module_list[0] is ml3b._module_list[1]


    def test_state_dict_flags_and_load_state_dict(self):
        leaf1 = Leaf4(v=1,f=10)
        leaf2 = Leaf4(v=2,f=20)
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


    def test_parameters_and_named_modules(self):
        leaf1 = Leaf4(v=1,f=1)
        leaf2 = Leaf4(v=2,f=2)
        ml = ModuleList(items=[leaf1, leaf2])
        # parameters yields leaf1.v and leaf2.v
        params = list(ml.parameters())
        assert all(isinstance(p, Param) for p in params)
        # named_modules includes self and children
        names = [name for name, _ in ml.named_modules()]
        assert names == ["", "0", "1"]


    def test_load_state_dict_non_strict_behavior(self):
        leaf1 = Leaf4(v=1,f=1)
        ml = ModuleList(items=[leaf1])
        # shorter list non-strict OK
        ml.load_state_dict({}, strict=False)
        # longer list non-strict OK
        ml.load_state_dict({"items": {"v":5}}, strict=False)



# # @registry()
# # class Leaf(BaseModule):
# #     w: InitVar[float]
# #     s: InitVar[int]
# #     def __post_init__(self, w: float, s: int):
# #         self.w = Param(w)
# #         self.s = State(s)

# # def make_leaf(val=1.0, step=0):
# #     return Leaf(w=val, s=step)

class TestModuleDict:

    def test_moduledict_getitem_key_error(self):
        d = ModuleDict(items={"x": make_leaf()})
        with pytest.raises(KeyError):
            _ = d["not_there"]

    def test_moduledict_key_must_be_string(self):
        d = ModuleDict(items={})
        with pytest.raises(TypeError):
            d[123] = make_leaf()

    def test_moduledict_registers_modules_correctly(self):
        leaf = make_leaf()
        d = ModuleDict(items={"leaf1": leaf})
        assert d._modules["leaf1"] is leaf

    def test_moduledict_from_spec_shared_instance_deduplicated(self):
        shared = make_leaf(4.2)
        d1 = ModuleDict(items={"a": shared, "b": shared})
        spec = d1.spec(to_dict=False)
        rebuilt = ModuleDict.from_spec(spec, ctx=dict())
        assert rebuilt["a"] is rebuilt["b"]

    def test_moduledict_from_spec_invalid_value_type(self):
        # Spec values must be BaseSpec instances
        class DummySpec(BaseSpec):
            kind: str = "invalid.kind"
        with pytest.raises(TypeError):
            ModuleDict.from_spec_hook("data", {"x": "not_a_spec"}, ctx={})

    def test_moduledict_spec_hook_invalid_name(self):
        d = ModuleDict(items={})
        with pytest.raises(ValueError):
            d.spec_hook(name="unknown", val={}, to_dict=True)

    def test_moduledict_from_spec_hook_invalid_name(self):
        with pytest.raises(ValueError):
            ModuleDict.from_spec_hook(name="not_items", val={}, ctx={})


    def test_moduledict_init_happy(self):
        d = ModuleDict(items={"a": make_leaf(), "b": make_leaf()})
        assert len(d) == 2 and "a" in d._modules and "b" in d._modules

    def test_moduledict_init_single(self):
        d = ModuleDict(items={"x": make_leaf()})
        assert len(d) == 1 and "x" in d._modules

    def test_moduledict_init_empty(self):
        d = ModuleDict(items={})
        assert len(d) == 0

    def test_moduledict_allows_primitives(self):
        d = ModuleDict(items={"good": 123})
        assert d["good"] == 123

    def test_moduledict_setitem(self):
        d = ModuleDict(items={})
        leaf = make_leaf(2.0)
        d["foo"] = leaf
        assert d["foo"] is leaf and "foo" in d._modules

    def test_moduledict_setitem_allows_strings(self):
        d = ModuleDict(items={})
        d["success"] = "not a module"
        assert d["success"] == "not a module"

    def test_moduledict_setitem_type_error(self):
        d = ModuleDict(items={})
        with pytest.raises(TypeError):
            d["fail"] = object() # not a BaseModule or primitive

    def test_moduledict_getitem(self):
        leaf = make_leaf()
        d = ModuleDict(items={"a": leaf})
        assert d["a"] is leaf

    def test_moduledict_len_iter_items(self):
        leaf1, leaf2 = make_leaf(1), make_leaf(2)
        d = ModuleDict(items={"x": leaf1, "y": leaf2})
        assert len(d) == 2
        assert set(d.keys()) == {"x", "y"}
        assert set(d.values()) == {leaf1, leaf2}
        assert dict(d.items()) == {"x": leaf1, "y": leaf2}

    def test_moduledict_overwrite_removes_old_attr(self):
        d = ModuleDict(items={"key": make_leaf(1)})
        d["key"] = make_leaf(9)
        assert d["key"].w.data == 9

    def test_moduledict_spec_roundtrip(self):
        d = ModuleDict(items={"a": make_leaf(3.3)})
        spec = d.spec()
        assert isinstance(spec, BaseSpec)
        assert spec.data["a"].w == 3.3

    def test_moduledict_from_spec_roundtrip(self):
        d1 = ModuleDict(items={"a": make_leaf(3.3), "b": make_leaf(1.1)})
        spec = d1.spec(to_dict=False)
        d2 = ModuleDict.from_spec(spec, ctx=dict())
        assert isinstance(d2, ModuleDict)
        assert d2["a"].w.data == 3.3

    def test_moduledict_state_dict_flags(self):
        d = ModuleDict(items={"a": make_leaf(1.0, 5), "b": make_leaf(2.0, 10)})
        sd = d.state_dict(runtime=False)
        assert sd == {"a.w": 1.0, "b.w": 2.0}

    def test_moduledict_state_dict_roundtrip(self):
        d1 = ModuleDict(items={"x": make_leaf(3, 4), "y": make_leaf(5, 6)})
        sd = d1.state_dict()
        d2 = ModuleDict(items={"x": make_leaf(0, 0), "y": make_leaf(0, 0)})
        d2.load_state_dict(sd, strict=True)
        assert d2.state_dict() == sd

    def test_moduledict_load_state_strict_fail(self):
        d = ModuleDict(items={"a": make_leaf()})
        with pytest.raises(KeyError):
            d.load_state_dict({"a.w": 1.0, "z.w": 2.0}, strict=True)

    def test_moduledict_load_state_non_strict(self):
        d = ModuleDict(items={"a": make_leaf()})
        d.load_state_dict({}, strict=False)

    def test_moduledict_load_state_type_error(self):
        d = ModuleDict(items={"a": make_leaf()})
        with pytest.raises(TypeError):
            d.load_state_dict("not a dict")

    def test_moduledict_named_parameters_states(self):
        d = ModuleDict(items={"k": make_leaf()})
        pnames = dict(d.named_parameters()).keys()
        snames = dict(d.named_states()).keys()
        assert "k.w" in pnames and "k.s" in snames

    def test_moduledict_named_modules_keys(self):
        d = ModuleDict(items={"a": make_leaf()})
        names = dict(d.named_modules()).keys()
        assert set(names) == {"", "a"}

    def test_moduledict_param_deduplication(self):
        leaf = make_leaf(2.2)
        d = ModuleDict(items={"first": leaf, "second": leaf})
        params = list(d.parameters())
        assert len(params) == 1

    def test_moduledict_shared_module_instance(self):
        shared = make_leaf(7.7)
        d = ModuleDict(items={"x": shared, "y": shared})
        state = d.state_dict()
        assert state["x.w"] == state["y.w"] == 7.7
