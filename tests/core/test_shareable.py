import pytest

from pydantic import BaseModel
from dachi.core._shareable import (
    ShareableItem,
    Param,
    Runtime,
    Shared,
)

class TestShareableItem:

    def test_get_when_data_is_set_returns_value(self):
        item = ShareableItem[int](data=42)
        assert item.get() == 42

    def test_get_when_data_is_none_returns_none(self):
        item = ShareableItem[int](data=None)
        assert item.get() is None

    def test_set_when_called_updates_data(self):
        item = ShareableItem[int](data=42)
        item.set(100)
        assert item.data == 100

    def test_set_when_called_returns_value(self):
        item = ShareableItem[int](data=42)
        result = item.set(100)
        assert result == 100

    def test_empty_when_data_is_none_returns_true(self):
        item = ShareableItem[int](data=None)
        assert item.empty() is True

    def test_empty_when_data_exists_returns_false(self):
        item = ShareableItem[int](data=42)
        assert item.empty() is False

    def test_setattr_data_when_called_uses_set_method(self):
        item = ShareableItem[int](data=42)
        item.data = 100
        assert item.data == 100

    def test_dump_when_data_is_basemodel_returns_dict(self):
        class TestModel(BaseModel):
            x: int
            y: str

        model = TestModel(x=1, y="test")
        item = ShareableItem[TestModel](data=model)
        result = item.dump()
        assert result == {"data": {"x": 1, "y": "test"}}

    def test_dump_when_data_is_primitive_returns_primitive(self):
        item = ShareableItem[int](data=42)
        result = item.dump()
        assert result == {"data": 42}

    def test_spec_schema_returns_json_schema(self):
        item = ShareableItem[int](data=42)
        schema = item.to_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema or "$defs" in schema

    def test_eq_when_data_equal_returns_true(self):
        item1 = ShareableItem[int](data=42)
        item2 = ShareableItem[int](data=42)
        assert item1 == item2

    def test_eq_when_data_different_returns_false(self):
        item1 = ShareableItem[int](data=42)
        item2 = ShareableItem[int](data=100)
        assert not (item1 == item2)

    def test_eq_when_comparing_to_non_shareable_returns_false(self):
        item = ShareableItem[int](data=42)
        assert not (item == 42)

    def test_ne_when_data_different_returns_true(self):
        item1 = ShareableItem[int](data=42)
        item2 = ShareableItem[int](data=100)
        assert item1 != item2

    def test_ne_when_data_equal_returns_false(self):
        item1 = ShareableItem[int](data=42)
        item2 = ShareableItem[int](data=42)
        assert not (item1 != item2)

    def test_lt_when_data_less_returns_true(self):
        item1 = ShareableItem[int](data=10)
        item2 = ShareableItem[int](data=20)
        assert item1 < item2

    def test_lt_when_data_greater_or_equal_returns_false(self):
        item1 = ShareableItem[int](data=20)
        item2 = ShareableItem[int](data=10)
        assert not (item1 < item2)

    def test_le_when_data_less_or_equal_returns_true(self):
        item1 = ShareableItem[int](data=10)
        item2 = ShareableItem[int](data=10)
        assert item1 <= item2

    def test_gt_when_data_greater_returns_true(self):
        item1 = ShareableItem[int](data=20)
        item2 = ShareableItem[int](data=10)
        assert item1 > item2

    def test_ge_when_data_greater_or_equal_returns_true(self):
        item1 = ShareableItem[int](data=20)
        item2 = ShareableItem[int](data=20)
        assert item1 >= item2

    def test_hash_returns_id(self):
        item = ShareableItem[int](data=42)
        assert hash(item) == id(item)

    def test_call_when_invoked_sets_data_and_returns(self):
        item = ShareableItem[int](data=42)
        result = item(100)
        assert item.data == 100
        assert result == 100

    def test_str_returns_data_string(self):
        item = ShareableItem[int](data=42)
        assert str(item) == "42"

    def test_repr_returns_class_and_data(self):
        item = ShareableItem[int](data=42)
        assert repr(item) == "ShareableItem[int](data=42)"


class TestShareableItemCallbacks:

    def test_register_callback_when_called_adds_to_list(self):
        item = ShareableItem[int](data=42)
        callback = lambda old, new: None
        item.register_callback(callback)
        assert item.has_callback(callback)

    def test_has_callback_when_registered_returns_true(self):
        item = ShareableItem[int](data=42)
        callback = lambda old, new: None
        item.register_callback(callback)
        assert item.has_callback(callback) is True

    def test_has_callback_when_not_registered_returns_false(self):
        item = ShareableItem[int](data=42)
        callback = lambda old, new: None
        assert item.has_callback(callback) is False

    def test_unregister_callback_when_exists_returns_true_and_removes(self):
        item = ShareableItem[int](data=42)
        callback = lambda old, new: None
        item.register_callback(callback)
        result = item.unregister_callback(callback)
        assert result is True
        assert not item.has_callback(callback)

    def test_unregister_callback_when_not_exists_returns_false(self):
        item = ShareableItem[int](data=42)
        callback = lambda old, new: None
        result = item.unregister_callback(callback)
        assert result is False

    def test_set_when_data_changes_invokes_all_callbacks(self):
        item = ShareableItem[int](data=42)
        call_count = {"count": 0}

        def callback(old, new):
            call_count["count"] += 1

        item.register_callback(callback)
        item.register_callback(callback)
        item.set(100)
        assert call_count["count"] == 2

    def test_set_when_data_changes_passes_old_and_new_values(self):
        item = ShareableItem[int](data=42)
        values = {"old": None, "new": None}

        def callback(old, new):
            values["old"] = old
            values["new"] = new

        item.register_callback(callback)
        item.set(100)
        assert values["old"] == 42
        assert values["new"] == 100

    def test_update_data_hook_when_called_invokes_callbacks(self):
        item = ShareableItem[int](data=42)
        called = {"value": False}

        def callback(old, new):
            called["value"] = True

        item.register_callback(callback)
        item.update_data_hook(42, 100)
        assert called["value"] is True


class TestShareableItemArithmetic:

    def test_add_when_with_shareable_creates_new_instance(self):
        item1 = ShareableItem[int](data=10)
        item2 = ShareableItem[int](data=20)
        result = item1 + item2
        assert result is not item1
        assert result is not item2
        assert result.data == 30

    def test_add_when_with_shareable_does_not_mutate_self(self):
        item1 = ShareableItem[int](data=10)
        item2 = ShareableItem[int](data=20)
        result = item1 + item2
        assert item1.data == 10
        assert item2.data == 20

    def test_add_when_with_primitive_creates_new_instance(self):
        item = ShareableItem[int](data=10)
        result = item + 20
        assert result is not item
        assert result.data == 30
        assert item.data == 10

    def test_radd_when_called_creates_new_instance(self):
        item = ShareableItem[int](data=10)
        result = 20 + item
        assert result is not item
        assert result.data == 30
        assert item.data == 10

    def test_iadd_when_called_creates_new_instance(self):
        item = ShareableItem[int](data=10)
        original_id = id(item)
        item += 20
        assert id(item) != original_id
        assert item.data == 30

    def test_sub_when_with_shareable_creates_new_instance(self):
        item1 = ShareableItem[int](data=30)
        item2 = ShareableItem[int](data=10)
        result = item1 - item2
        assert result is not item1
        assert result.data == 20
        assert item1.data == 30

    def test_sub_when_with_primitive_creates_new_instance(self):
        item = ShareableItem[int](data=30)
        result = item - 10
        assert result is not item
        assert result.data == 20
        assert item.data == 30

    def test_rsub_when_called_creates_new_instance(self):
        item = ShareableItem[int](data=10)
        result = 30 - item
        assert result is not item
        assert result.data == 20
        assert item.data == 10

    def test_isub_when_called_creates_new_instance(self):
        item = ShareableItem[int](data=30)
        original_id = id(item)
        item -= 10
        assert id(item) != original_id
        assert item.data == 20

    def test_mul_when_with_shareable_creates_new_instance(self):
        item1 = ShareableItem[int](data=5)
        item2 = ShareableItem[int](data=3)
        result = item1 * item2
        assert result is not item1
        assert result.data == 15
        assert item1.data == 5

    def test_mul_when_with_primitive_creates_new_instance(self):
        item = ShareableItem[int](data=5)
        result = item * 3
        assert result is not item
        assert result.data == 15
        assert item.data == 5

    def test_rmul_when_called_creates_new_instance(self):
        item = ShareableItem[int](data=5)
        result = 3 * item
        assert result is not item
        assert result.data == 15
        assert item.data == 5

    def test_imul_when_called_creates_new_instance(self):
        item = ShareableItem[int](data=5)
        original_id = id(item)
        item *= 3
        assert id(item) != original_id
        assert item.data == 15

    def test_truediv_when_with_shareable_creates_new_instance(self):
        item1 = ShareableItem[float](data=20.0)
        item2 = ShareableItem[float](data=4.0)
        result = item1 / item2
        assert result is not item1
        assert result.data == 5.0
        assert item1.data == 20.0

    def test_truediv_when_with_primitive_creates_new_instance(self):
        item = ShareableItem[float](data=20.0)
        result = item / 4.0
        assert result is not item
        assert result.data == 5.0
        assert item.data == 20.0

    def test_rtruediv_when_called_creates_new_instance(self):
        item = ShareableItem[float](data=4.0)
        result = 20.0 / item
        assert result is not item
        assert result.data == 5.0
        assert item.data == 4.0

    def test_itruediv_when_called_creates_new_instance(self):
        item = ShareableItem[float](data=20.0)
        original_id = id(item)
        item /= 4.0
        assert id(item) != original_id
        assert item.data == 5.0

    def test_floordiv_when_with_shareable_creates_new_instance(self):
        item1 = ShareableItem[int](data=20)
        item2 = ShareableItem[int](data=3)
        result = item1 // item2
        assert result is not item1
        assert result.data == 6
        assert item1.data == 20

    def test_floordiv_when_with_primitive_creates_new_instance(self):
        item = ShareableItem[int](data=20)
        result = item // 3
        assert result is not item
        assert result.data == 6
        assert item.data == 20

    def test_rfloordiv_when_called_creates_new_instance(self):
        item = ShareableItem[int](data=3)
        result = 20 // item
        assert result is not item
        assert result.data == 6
        assert item.data == 3

    def test_ifloordiv_when_called_creates_new_instance(self):
        item = ShareableItem[int](data=20)
        original_id = id(item)
        item //= 3
        assert id(item) != original_id
        assert item.data == 6

    def test_mod_when_with_shareable_creates_new_instance(self):
        item1 = ShareableItem[int](data=20)
        item2 = ShareableItem[int](data=3)
        result = item1 % item2
        assert result is not item1
        assert result.data == 2
        assert item1.data == 20

    def test_mod_when_with_primitive_creates_new_instance(self):
        item = ShareableItem[int](data=20)
        result = item % 3
        assert result is not item
        assert result.data == 2
        assert item.data == 20

    def test_rmod_when_called_creates_new_instance(self):
        item = ShareableItem[int](data=3)
        result = 20 % item
        assert result is not item
        assert result.data == 2
        assert item.data == 3

    def test_imod_when_called_creates_new_instance(self):
        item = ShareableItem[int](data=20)
        original_id = id(item)
        item %= 3
        assert id(item) != original_id
        assert item.data == 2

    def test_pow_when_with_shareable_creates_new_instance(self):
        item1 = ShareableItem[int](data=2)
        item2 = ShareableItem[int](data=3)
        result = item1 ** item2
        assert result is not item1
        assert result.data == 8
        assert item1.data == 2

    def test_pow_when_with_primitive_creates_new_instance(self):
        item = ShareableItem[int](data=2)
        result = item ** 3
        assert result is not item
        assert result.data == 8
        assert item.data == 2

    def test_rpow_when_called_creates_new_instance(self):
        item = ShareableItem[int](data=3)
        result = 2 ** item
        assert result is not item
        assert result.data == 8
        assert item.data == 3

    def test_ipow_when_called_creates_new_instance(self):
        item = ShareableItem[int](data=2)
        original_id = id(item)
        item **= 3
        assert id(item) != original_id
        assert item.data == 8



class TestParam:

    def test_set_when_not_fixed_updates_data_and_returns(self):
        param = Param[int](data=42)
        result = param.set(100)
        assert param.data == 100
        assert result == 100

    def test_set_when_fixed_raises_runtime_error(self):
        param = Param[int](data=42)
        param.fix()
        with pytest.raises(RuntimeError, match="Cannot set parameter that is fixed"):
            param.set(100)

    def test_is_fixed_when_fixed_true_returns_true(self):
        param = Param[int](data=42)
        param.fix()
        assert param.is_fixed() is True

    def test_is_fixed_when_fixed_false_returns_false(self):
        param = Param[int](data=42)
        assert param.is_fixed() is False

    def test_fix_when_called_sets_fixed_to_true(self):
        param = Param[int](data=42)
        param.fix()
        assert param._fixed is True

    def test_unfix_when_called_sets_fixed_to_false(self):
        param = Param[int](data=42)
        param.fix()
        param.unfix()
        assert param._fixed is False

    def test_create_when_initialized_inherits_shareable_behavior(self):
        param = Param[int](data=42)
        assert param.get() == 42
        assert param.empty() is False


class TestRuntime:

    def test_create_when_initialized_stores_data(self):
        runtime = Runtime[int](data=42)
        assert runtime.data == 42

    def test_inherits_shareable_item_behavior(self):
        runtime = Runtime[int](data=42)
        assert runtime.get() == 42
        runtime.set(100)
        assert runtime.data == 100


class TestShared:

    def test_create_when_initialized_stores_data(self):
        shared = Shared[str](data="test")
        assert shared.data == "test"

    def test_inherits_shareable_item_behavior(self):
        shared = Shared[str](data="test")
        assert shared.get() == "test"
        shared.set("updated")
        assert shared.data == "updated"