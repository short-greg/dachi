from __future__ import annotations

import pytest
import pydantic

from dachi.core._base import Module, Param, Runtime, PrivateParam, PrivateRuntime
from dachi.core._structs import ModuleList, ModuleDict


class SimpleModule(Module):
    value: int = pydantic.Field(default=0, frozen=True)
    _value: Param[int] = PrivateParam(instance_field="value")


class SimpleModule2(Module):
    x: int = pydantic.Field(default=0, frozen=True)
    y: int = pydantic.Field(default=0, frozen=True)
    _x: Param[int] = PrivateParam(instance_field="x")
    _y: Runtime[int] = PrivateRuntime(instance_field="y")


class TestModuleListBasicOperations:

    def test_init_when_empty_creates_empty_list(self):
        ml = ModuleList[SimpleModule](vals=[])
        assert len(ml) == 0

    def test_init_when_given_items_contains_all_items(self):
        m1, m2 = SimpleModule(value=10), SimpleModule(value=20)
        ml = ModuleList[SimpleModule](vals=[m1, m2])
        assert len(ml) == 2
        assert ml[0] is m1
        assert ml[1] is m2

    def test_len_when_called_returns_item_count(self):
        ml = ModuleList[SimpleModule](vals=[SimpleModule(), SimpleModule()])
        assert len(ml) == 2

    def test_getitem_when_valid_index_returns_item(self):
        m1, m2 = SimpleModule(value=10), SimpleModule(value=20)
        ml = ModuleList[SimpleModule](vals=[m1, m2])
        assert ml[0] is m1
        assert ml[1] is m2

    def test_getitem_when_negative_index_returns_from_end(self):
        m1, m2, m3 = SimpleModule(value=1), SimpleModule(value=2), SimpleModule(value=3)
        ml = ModuleList[SimpleModule](vals=[m1, m2, m3])
        assert ml[-1] is m3
        assert ml[-2] is m2

    def test_getitem_when_out_of_bounds_raises_index_error(self):
        ml = ModuleList[SimpleModule](vals=[SimpleModule()])
        with pytest.raises(IndexError):
            _ = ml[10]

    def test_iter_when_called_yields_items_in_order(self):
        m1, m2 = SimpleModule(value=1), SimpleModule(value=2)
        ml = ModuleList[SimpleModule](vals=[m1, m2])
        assert list(ml) == [m1, m2]

    def test_append_when_module_adds_to_end(self):
        ml = ModuleList[SimpleModule](vals=[])
        m = SimpleModule(value=42)
        ml.append(m)
        assert len(ml) == 1
        assert ml[0] is m

    def test_append_when_not_module_raises_type_error(self):
        ml = ModuleList[SimpleModule](vals=[])
        with pytest.raises(TypeError, match="ModuleList accepts only BaseModule instances"):
            ml.append(42)

    def test_setitem_when_valid_replaces_item(self):
        m1, m2 = SimpleModule(value=1), SimpleModule(value=2)
        ml = ModuleList[SimpleModule](vals=[m1])
        ml[0] = m2
        assert ml[0] is m2

    def test_setitem_when_not_module_raises_type_error(self):
        ml = ModuleList[SimpleModule](vals=[SimpleModule()])
        with pytest.raises(TypeError, match="ModuleList accepts only BaseModule instances"):
            ml[0] = "not a module"

    def test_setitem_when_out_of_bounds_raises_index_error(self):
        ml = ModuleList[SimpleModule](vals=[SimpleModule()])
        with pytest.raises(IndexError, match="out of bounds"):
            ml[10] = SimpleModule()

    def test_aslist_when_called_returns_list_copy(self):
        m1, m2 = SimpleModule(value=1), SimpleModule(value=2)
        ml = ModuleList[SimpleModule](vals=[m1, m2])
        result = ml.aslist
        assert result == [m1, m2]
        assert result is not ml.vals


class TestModuleListModulesAndNaming:

    def test_modules_when_not_recursive_yields_self_and_children(self):
        m1, m2 = SimpleModule(value=1), SimpleModule(value=2)
        ml = ModuleList[SimpleModule](vals=[m1, m2])

        mods = list(ml.modules(recurse=False))
        assert ml in mods
        assert m1 in mods
        assert m2 in mods

    def test_modules_when_recursive_yields_all_descendants(self):
        m_inner = SimpleModule(value=1)
        ml_inner = ModuleList[SimpleModule](vals=[m_inner])
        ml_outer = ModuleList[ModuleList](vals=[ml_inner])

        mods = list(ml_outer.modules(recurse=True))
        assert ml_outer in mods
        assert ml_inner in mods
        assert m_inner in mods

    def test_modules_when_duplicate_reference_deduplicates(self):
        m = SimpleModule(value=42)
        ml = ModuleList[SimpleModule](vals=[m, m, m])

        mods = list(ml.modules())
        assert len([mod for mod in mods if mod is m]) == 1

    def test_modules_when_filter_provided_only_yields_matching(self):
        m1, m2, m3 = SimpleModule(value=1), SimpleModule(value=10), SimpleModule(value=20)
        ml = ModuleList[SimpleModule](vals=[m1, m2, m3])

        def filter_func(mod):
            return isinstance(mod, SimpleModule) and mod._value.data >= 10

        filtered = list(ml.modules(f=filter_func, _skip_self=True))
        assert m1 not in filtered
        assert m2 in filtered
        assert m3 in filtered

    def test_named_modules_when_called_returns_indexed_names(self):
        m1, m2 = SimpleModule(value=1), SimpleModule(value=2)
        ml = ModuleList[SimpleModule](vals=[m1, m2])

        named = dict(ml.named_modules())
        assert "" in named
        assert "0" in named
        assert "1" in named

    def test_named_modules_when_nested_uses_dot_notation(self):
        m = SimpleModule(value=1)
        ml_inner = ModuleList[SimpleModule](vals=[m])
        ml_outer = ModuleList[ModuleList](vals=[ml_inner])

        named = dict(ml_outer.named_modules())
        assert "" in named
        assert "0" in named
        assert "0.0" in named

    def test_named_modules_when_skip_self_excludes_self(self):
        m1 = SimpleModule(value=1)
        ml = ModuleList[SimpleModule](vals=[m1])

        named = dict(ml.named_modules(_skip_self=True))
        assert "" not in named
        assert "0" in named


class TestModuleListEdgeCases:

    def test_empty_list_has_zero_length(self):
        ml = ModuleList[SimpleModule](vals=[])
        assert len(ml) == 0
        assert list(ml) == []

    def test_multiple_appends_work_correctly(self):
        ml = ModuleList[SimpleModule](vals=[])
        for i in range(5):
            ml.append(SimpleModule(value=i))
        assert len(ml) == 5
        assert ml[4]._value.data == 4

    def test_very_deep_nesting_does_not_crash(self):
        current = ModuleList[SimpleModule](vals=[SimpleModule(value=0)])
        for i in range(20):
            current = ModuleList[ModuleList](vals=[current])

        mods = list(current.modules(recurse=True))
        assert isinstance(mods, list)


class TestModuleDictBasicOperations:

    def test_init_when_empty_creates_empty_dict(self):
        md = ModuleDict[SimpleModule](vals={})
        assert len(md) == 0

    def test_init_when_given_items_contains_all_items(self):
        m1, m2 = SimpleModule(value=1), SimpleModule(value=2)
        md = ModuleDict[SimpleModule](vals={"a": m1, "b": m2})
        assert len(md) == 2
        assert md["a"] is m1
        assert md["b"] is m2

    def test_len_when_called_returns_item_count(self):
        md = ModuleDict[SimpleModule](vals={"a": SimpleModule(), "b": SimpleModule()})
        assert len(md) == 2

    def test_getitem_when_key_exists_returns_value(self):
        m = SimpleModule(value=42)
        md = ModuleDict[SimpleModule](vals={"test": m})
        assert md["test"] is m

    def test_getitem_when_key_missing_raises_key_error(self):
        md = ModuleDict[SimpleModule](vals={})
        with pytest.raises(KeyError):
            _ = md["missing"]

    def test_setitem_when_module_adds_to_dict(self):
        md = ModuleDict[SimpleModule](vals={})
        m = SimpleModule(value=42)
        md["key"] = m
        assert md["key"] is m
        assert "key" in md.vals

    def test_setitem_when_non_string_key_raises_type_error(self):
        md = ModuleDict[SimpleModule](vals={})
        with pytest.raises(TypeError, match="Keys must be strings"):
            md[123] = SimpleModule()

    def test_setitem_when_invalid_value_raises_type_error(self):
        md = ModuleDict[SimpleModule](vals={})
        with pytest.raises(TypeError, match="Values must be Module instances or primitives"):
            md["bad"] = object()

    def test_iter_when_called_yields_keys(self):
        md = ModuleDict[SimpleModule](vals={"a": SimpleModule(), "b": SimpleModule()})
        assert set(md) == {"a", "b"}

    def test_keys_when_called_returns_keys(self):
        md = ModuleDict[SimpleModule](vals={"a": SimpleModule(), "b": SimpleModule()})
        assert set(md.keys()) == {"a", "b"}

    def test_values_when_called_returns_values(self):
        m1, m2 = SimpleModule(value=1), SimpleModule(value=2)
        md = ModuleDict[SimpleModule](vals={"a": m1, "b": m2})
        values = list(md.values())
        assert m1 in values
        assert m2 in values
        assert len(values) == 2

    def test_get_when_key_exists_returns_value(self):
        m = SimpleModule(value=42)
        md = ModuleDict[SimpleModule](vals={"key": m})
        assert md.get("key") is m

    def test_get_when_key_missing_returns_default(self):
        md = ModuleDict[SimpleModule](vals={})
        assert md.get("missing") is None
        assert md.get("missing", "default") == "default"


class TestModuleDictModulesAndNaming:

    def test_modules_when_not_recursive_yields_self_and_values(self):
        m1, m2 = SimpleModule(value=1), SimpleModule(value=2)
        md = ModuleDict[SimpleModule](vals={"a": m1, "b": m2})

        mods = list(md.modules(recurse=False))
        assert md in mods
        assert m1 in mods
        assert m2 in mods

    def test_modules_when_recursive_yields_all_descendants(self):
        m = SimpleModule(value=1)
        md_inner = ModuleDict[SimpleModule](vals={"inner": m})
        md_outer = ModuleDict[ModuleDict](vals={"outer": md_inner})

        mods = list(md_outer.modules(recurse=True))
        assert md_outer in mods
        assert md_inner in mods
        assert m in mods

    def test_modules_when_duplicate_reference_deduplicates(self):
        m = SimpleModule(value=42)
        md = ModuleDict[SimpleModule](vals={"a": m, "b": m, "c": m})

        mods = list(md.modules())
        assert len([mod for mod in mods if mod is m]) == 1

    def test_modules_when_filter_provided_only_yields_matching(self):
        m1, m2 = SimpleModule(value=1), SimpleModule(value=10)
        md = ModuleDict[SimpleModule](vals={"small": m1, "large": m2})

        def filter_func(mod):
            return isinstance(mod, SimpleModule) and mod._value.data >= 10

        filtered = list(md.modules(f=filter_func, _skip_self=True))
        assert m1 not in filtered
        assert m2 in filtered

    def test_named_modules_when_called_returns_keyed_names(self):
        m1, m2 = SimpleModule(value=1), SimpleModule(value=2)
        md = ModuleDict[SimpleModule](vals={"first": m1, "second": m2})

        named = dict(md.named_modules())
        assert "" in named
        assert "first" in named
        assert "second" in named

    def test_named_modules_when_nested_uses_dot_notation(self):
        m = SimpleModule(value=1)
        md_inner = ModuleDict[SimpleModule](vals={"inner": m})
        md_outer = ModuleDict[ModuleDict](vals={"outer": md_inner})

        named = dict(md_outer.named_modules())
        assert "" in named
        assert "outer" in named
        assert "outer.inner" in named

    def test_named_modules_when_skip_self_excludes_self(self):
        m = SimpleModule(value=1)
        md = ModuleDict[SimpleModule](vals={"key": m})

        named = dict(md.named_modules(_skip_self=True))
        assert "" not in named
        assert "key" in named


class TestModuleDictEdgeCases:

    def test_empty_dict_has_zero_length(self):
        md = ModuleDict[SimpleModule](vals={})
        assert len(md) == 0
        assert list(md) == []

    def test_setitem_overwrites_existing_key(self):
        m1, m2 = SimpleModule(value=1), SimpleModule(value=2)
        md = ModuleDict[SimpleModule](vals={"key": m1})
        md["key"] = m2
        assert md["key"] is m2

    def test_very_deep_nesting_does_not_crash(self):
        current = ModuleDict[SimpleModule](vals={"leaf": SimpleModule(value=0)})
        for i in range(20):
            current = ModuleDict[ModuleDict](vals={f"level{i}": current})

        mods = list(current.modules(recurse=True))
        assert isinstance(mods, list)
