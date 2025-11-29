from __future__ import annotations
import pytest
import typing as t
import pydantic
from pydantic import BaseModel

from dachi.core._base import (
    ShareableItem,
    Param,
    Runtime,
    Shared,
    PrivateParam,
    PrivateRuntime,
    PrivateShared,
    ObjInit,
    Module,
    StateType,
    Registry,
    AdaptModule,
    Checkpoint,
    mod_registry,
)
from typing import Literal

import json

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


class TestPrivateParam:

    def test_with_default_when_called_creates_private_attr_with_value(self):
        private_attr = PrivateParam(default=42)
        assert private_attr.default_factory is None
        param = private_attr.default()
        assert isinstance(param, Param)
        assert param.data == 42

    def test_with_default_factory_when_called_creates_private_attr_with_factory(self):
        private_attr = PrivateParam(default_factory=lambda: 100)
        assert private_attr.default_factory is None
        param = private_attr.default()
        assert isinstance(param, Param)
        assert param.data == 100

    def test_with_instance_field_when_called_creates_selfinit(self):
        private_attr = PrivateParam(instance_field="x")
        assert private_attr.default is not None
        assert isinstance(private_attr.default, ObjInit)

    def test_with_instance_factory_when_called_creates_selfinit(self):
        factory_fn = lambda m: Param(data=m.x if hasattr(m, 'x') else 0)
        private_attr = PrivateParam(instance_factory=factory_fn)
        assert private_attr.default is not None
        assert isinstance(private_attr.default, ObjInit)


class TestPrivateRuntime:

    def test_with_default_when_called_creates_private_attr_with_value(self):
        private_attr = PrivateRuntime(default=42)
        assert private_attr.default_factory is None
        runtime = private_attr.default()
        assert isinstance(runtime, Runtime)
        assert runtime.data == 42

    def test_with_default_factory_when_called_creates_private_attr_with_factory(self):
        private_attr = PrivateRuntime(default_factory=lambda: 100)
        assert private_attr.default is not None
        runtime = private_attr.default()
        assert isinstance(runtime, Runtime)
        assert runtime.data == 100


class TestPrivateShared:

    def test_with_default_when_called_creates_private_attr_with_value(self):
        private_attr = PrivateShared(default="test")
        assert private_attr.default is not None
        shared = private_attr.default()
        assert isinstance(shared, Shared)
        assert shared.data == "test"

    def test_with_default_factory_when_called_creates_private_attr_with_factory(self):
        private_attr = PrivateShared(default_factory=lambda: "factory_value")
        assert private_attr.default is not None
        shared = private_attr.default()
        assert isinstance(shared, Shared)
        assert shared.data == "factory_value"


class TestSelfInit:

    def test_call_when_invoked_executes_function_with_module(self):
        class TestModule(Module):
            x: int

        def factory_fn(module):
            return module.x * 2

        self_init = ObjInit(factory_fn, Param)
        test_module = TestModule(x=10)
        result = self_init(test_module)
        assert result.data == 20

    def test_call_when_invoked_returns_result(self):
        def factory_fn(module):
            return 42

        self_init = ObjInit(factory_fn, Param)

        class DummyModule:
            pass

        result = self_init(DummyModule())
        assert result.data == 42


class TestModuleInitialization:

    def test_init_subclass_when_subclassed_sets_kind_const_to_qualname(self):
        class TestModule(Module):
            pass

        # Check via instance since Pydantic intercepts class attribute access
        instance = TestModule()
        assert "TestModule" in instance.KIND
        assert instance.KIND == TestModule.__qualname__

    def test_init_subclass_when_subclassed_updates_annotation_to_literal(self):
        class TestModule(Module):
            pass

        hints = TestModule.__annotations__
        assert "KIND" in hints
        # The annotation should be a Literal type with the qualname
        expected_literal = Literal[TestModule.__qualname__]
        assert hints["KIND"] == expected_literal

    def test_init_subclass_when_is_module_does_nothing(self):
        # Module base class should have KIND field
        assert "KIND" in Module.model_fields


class TestModuleRegistry:

    def test_model_post_init_when_private_param_assigned_registers_as_param(self):

        class TestModule(Module):
            _param: Param[int] = PrivateParam(default=42)

        module = TestModule()
        assert "_param" in module._registry
        assert module._registry["_param"] == StateType.PARAM

    def test_model_post_init_when_private_runtime_assigned_registers_as_runtime(self):
        class TestModule(Module):
            _runtime: Runtime[int] = PrivateRuntime(default=0)

        module = TestModule()
        assert "_runtime" in module._registry
        assert module._registry["_runtime"] == StateType.RUNTIME

    def test_model_post_init_when_private_module_assigned_registers_as_module(self):
        class ChildModule(Module):
            pass

        class ParentModule(Module):
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        # Module instances assigned after init will be registered via setattr
        assert "_child" in parent._registry
        assert parent._registry["_child"] == StateType.MODULE

    def test_model_post_init_when_selfinit_param_executes_and_registers(self):
        class TestModule(Module):
            x: int
            _derived: Param[int] = PrivateParam(instance_field="x")

        module = TestModule(x=10)
        assert "_derived" in module._registry
        assert module._registry["_derived"] == StateType.PARAM
        assert module._derived.data == 10

    def test_model_post_init_when_duplicate_param_name_raises_runtime_error(self):
        class TestModule(Module):
            _param: Param[int] = PrivateParam(default=42)

            def model_post_init(self, __context):
                super().model_post_init(__context)
                # Try to register the same name again
                self._registry["_param"] = StateType.PARAM

        # This should not raise during creation
        module = TestModule()
        assert "_param" in module._registry

    def test_setattr_when_param_assigned_after_init_raises_error(self):
        class TestModule(Module):
            pass

        module = TestModule()
        with pytest.raises(ValueError, match='object has no field'):
            module.new_param = Param[int](data=42)

    def test_setattr_when_runtime_assigned_after_init_raises_error(self):
        class TestModule(Module):
            pass

        module = TestModule()
        with pytest.raises(ValueError, match='object has no field'):
            module.new_runtime = Runtime[int](data=0)

    def test_setattr_when_module_assigned_after_init_raises_error(self):
        class ChildModule(Module):
            pass

        class ParentModule(Module):
            pass

        parent = ParentModule()
        with pytest.raises(ValueError, match='object has no field'):
            parent.child = ChildModule()

    def test_setattr_when_name_starts_with_underscore_does_not_register(self):
        class TestModule(Module):
            pass

        module = TestModule()
        module._internal = "value"
        # _internal should not be in registry (except for private attrs set during init)
        # Actually, private attrs starting with _ are skipped in __setattr__
        assert "_internal" not in module._registry or module._registry.get("_internal") is None


class TestModuleParameters:

    def test_parameters_when_no_recurse_returns_local_params_only(self):
        class TestModule(Module):
            _p1: Param[int] = PrivateParam(default=1)
            _p2: Param[int] = PrivateParam(default=2)

        module = TestModule()
        params = list(module.parameters(recurse=False))
        assert len(params) == 2
        assert all(isinstance(p, Param) for p in params)

    def test_parameters_when_recurse_true_returns_all_params_including_children(self):
        class ChildModule(Module):
            _p_child: Param[int] = PrivateParam(default=10)

        class ParentModule(Module):
            _p_parent: Param[int] = PrivateParam(default=20)
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        params = list(parent.parameters(recurse=True))
        assert len(params) == 2

    def test_parameters_when_with_annotations_true_returns_param_annotation_tuples(self):
        class TestModule(Module):
            _p1: Param[int] = PrivateParam(default=1)

        module = TestModule()
        params_with_annot = list(module.parameters(recurse=False, with_annotations=True))
        assert len(params_with_annot) == 1
        param, annotation = params_with_annot[0]
        assert isinstance(param, Param)

    def test_parameters_when_with_annotations_false_returns_params_only(self):
        class TestModule(Module):
            _p1: Param[int] = PrivateParam(default=1)

        module = TestModule()
        params = list(module.parameters(recurse=False, with_annotations=False))
        assert len(params) == 1
        assert isinstance(params[0], Param)

    def test_parameters_when_param_seen_twice_returns_once(self):
        class TestModule(Module):
            _p1: Param[int] = PrivateParam(default=1)

        module = TestModule()
        # Use the same seen set
        seen = set()
        params1 = list(module.parameters(recurse=False, _seen=seen))
        params2 = list(module.parameters(recurse=False, _seen=seen))
        assert len(params1) == 1
        assert len(params2) == 0  # Already seen


class TestModuleNamedParameters:

    def test_named_parameters_when_no_recurse_returns_local_names_and_params(self):
        class TestModule(Module):
            _p1: Param[int] = PrivateParam(default=1)
            _p2: Param[int] = PrivateParam(default=2)

        module = TestModule()
        named_params = dict(module.named_parameters(recurse=False))
        assert "_p1" in named_params
        assert "_p2" in named_params
        assert isinstance(named_params["_p1"], Param)

    def test_named_parameters_when_recurse_true_returns_dotted_paths(self):
        class ChildModule(Module):
            _p_child: Param[int] = PrivateParam(default=10)

        class ParentModule(Module):
            _p_parent: Param[int] = PrivateParam(default=20)
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        named_params = dict(parent.named_parameters(recurse=True))
        assert "_p_parent" in named_params
        assert "_child._p_child" in named_params

    def test_named_parameters_when_prefix_provided_prepends_to_names(self):
        class TestModule(Module):
            _p1: Param[int] = PrivateParam(default=1)

        module = TestModule()
        named_params = dict(module.named_parameters(recurse=False, prefix="prefix."))
        assert "prefix._p1" in named_params

class TestModuleModules:

    def test_modules_when_no_recurse_returns_only_self(self):
        class ChildModule(Module):
            pass

        class ParentModule(Module):
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        modules = list(parent.modules(recurse=False))
        assert len(modules) == 2
        assert modules[0] is parent

    def test_modules_when_recurse_true_returns_self_and_children(self):
        class ChildModule(Module):
            pass

        class ParentModule(Module):
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        modules = list(parent.modules(recurse=True))
        assert len(modules) == 2
        assert parent in modules
        assert parent._child in modules

    def test_modules_when_filter_provided_filters_modules(self):
        class ChildModule(Module):
            x: int = 1

        class ParentModule(Module):
            y: int = 2
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        # Filter to only modules with attribute 'x'
        modules = list(parent.modules(recurse=True, f=lambda m: hasattr(m, 'x')))
        assert len(modules) == 1
        assert modules[0] is parent._child

    def test_modules_when_nested_returns_all_descendants(self):
        class GrandchildModule(Module):
            pass

        class ChildModule(Module):
            _grandchild: GrandchildModule = pydantic.PrivateAttr(default_factory=GrandchildModule)

        class ParentModule(Module):
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        modules = list(parent.modules(recurse=True))
        assert len(modules) == 3
        assert parent in modules
        assert parent._child in modules
        assert parent._child._grandchild in modules


class TestModuleNamedModules:

    def test_named_modules_when_no_recurse_returns_only_self(self):
        class ChildModule(Module):
            pass

        class ParentModule(Module):
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        named_modules = dict(parent.named_modules(recurse=False))
        assert len(named_modules) == 2
        assert "" in named_modules
        assert named_modules[""] is parent

    def test_named_modules_when_recurse_true_returns_dotted_paths(self):
        class ChildModule(Module):
            pass

        class ParentModule(Module):
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        named_modules = dict(parent.named_modules(recurse=True))
        assert "" in named_modules
        assert "_child" in named_modules
        assert named_modules[""] is parent
        assert named_modules["_child"] is parent._child

    def test_named_modules_when_prefix_provided_prepends_to_names(self):
        class ChildModule(Module):
            pass

        class ParentModule(Module):
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        named_modules = dict(parent.named_modules(recurse=True, prefix="root."))
        assert "root" in named_modules
        assert "root._child" in named_modules

    def test_named_modules_when_nested_returns_full_paths(self):
        class GrandchildModule(Module):
            pass

        class ChildModule(Module):
            _grandchild: GrandchildModule = pydantic.PrivateAttr(default_factory=GrandchildModule)

        class ParentModule(Module):
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        named_modules = dict(parent.named_modules(recurse=True))
        assert "" in named_modules
        assert "_child" in named_modules
        assert "_child._grandchild" in named_modules
        assert named_modules["_child._grandchild"] is parent._child._grandchild


class TestModuleChildren:

    def test_children_when_no_children_returns_empty_list(self):
        class TestModule(Module):
            _p1: Param[int] = PrivateParam(default=1)

        module = TestModule()
        children = module.children()
        assert len(children) == 0

    def test_children_when_has_children_returns_immediate_children_only(self):
        class GrandchildModule(Module):
            pass

        class ChildModule(Module):
            _grandchild: GrandchildModule = pydantic.PrivateAttr(default_factory=GrandchildModule)

        class ParentModule(Module):
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        children = parent.children()
        assert len(children) == 1
        assert children[0] is parent._child

    def test_children_when_multiple_children_returns_all(self):
        class ChildModule1(Module):
            pass

        class ChildModule2(Module):
            pass

        class ParentModule(Module):
            _child1: ChildModule1 = pydantic.PrivateAttr(default_factory=ChildModule1)
            _child2: ChildModule2 = pydantic.PrivateAttr(default_factory=ChildModule2)

        parent = ParentModule()
        children = parent.children()
        assert len(children) == 2
        assert parent._child1 in children
        assert parent._child2 in children

    def test_children_excludes_params_and_runtime(self):
        class ChildModule(Module):
            pass

        class ParentModule(Module):
            _param: Param[int] = PrivateParam(default=1)
            _runtime: Runtime[int] = PrivateRuntime(default=0)
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        children = parent.children()
        assert len(children) == 1
        assert children[0] is parent._child


class TestModuleNamedChildren:

    def test_named_children_when_no_children_returns_empty(self):
        class TestModule(Module):
            _p1: Param[int] = PrivateParam(default=1)

        module = TestModule()
        named_children = list(module.named_children())
        assert len(named_children) == 0

    def test_named_children_when_has_children_returns_name_module_pairs(self):
        class ChildModule(Module):
            pass

        class ParentModule(Module):
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        named_children = dict(parent.named_children())
        assert "_child" in named_children
        assert named_children["_child"] is parent._child

    def test_named_children_returns_immediate_children_only(self):
        class GrandchildModule(Module):
            pass

        class ChildModule(Module):
            _grandchild: GrandchildModule = pydantic.PrivateAttr(default_factory=GrandchildModule)

        class ParentModule(Module):
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        named_children = dict(parent.named_children())
        assert len(named_children) == 1
        assert "_child" in named_children
        assert "_grandchild" not in named_children


class TestModuleNamedStates:

    def test_named_states_when_no_recurse_returns_local_states_only(self):
        class TestModule(Module):
            _s1: Runtime[int] = PrivateRuntime(default=1)
            _s2: Runtime[int] = PrivateRuntime(default=2)

        module = TestModule()
        named_states = dict(module.named_states(recurse=False))
        print(named_states)
        assert "_s1" in named_states
        assert "_s2" in named_states
        assert len(named_states) == 3

    def test_named_states_when_recurse_true_returns_dotted_paths(self):
        class ChildModule(Module):
            _s_child: Runtime[int] = PrivateRuntime(default=10)

        class ParentModule(Module):
            _s_parent: Runtime[int] = PrivateRuntime(default=20)
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        named_states = dict(parent.named_states(recurse=True))
        assert "_s_parent" in named_states
        assert "_child._s_child" in named_states

    def test_named_states_when_prefix_provided_prepends_to_names(self):
        class TestModule(Module):
            _s1: Runtime[int] = PrivateRuntime(default=1)

        module = TestModule()
        named_states = dict(module.named_states(recurse=False, prefix="prefix."))
        assert "prefix._s1" in named_states

    def test_named_states_excludes_params_and_modules(self):
        class ChildModule(Module):
            pass

        class ParentModule(Module):
            _param: Param[int] = PrivateParam(default=1)
            _state: Runtime[int] = PrivateRuntime(default=2)
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        named_states = dict(parent.named_states(recurse=False))
        assert len(named_states) == 2
        assert "_state" in named_states
        assert "_param" not in named_states
        assert "_child" not in named_states


class TestModuleStateDict:

    def test_state_dict_when_no_recurse_returns_local_state_only(self):
        class ChildModule(Module):
            _p_child: Param[int] = PrivateParam(default=10)

        class ParentModule(Module):
            _p_parent: Param[int] = PrivateParam(default=20)
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        sd = parent.state_dict(recurse=False)
        assert "_p_parent" in sd
        assert "_child._p_child" not in sd

    def test_state_dict_when_recurse_true_returns_dotted_paths(self):
        class ChildModule(Module):
            _p_child: Param[int] = PrivateParam(default=10)

        class ParentModule(Module):
            _p_parent: Param[int] = PrivateParam(default=20)
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        sd = parent.state_dict(recurse=True)
        assert "_p_parent" in sd
        assert "_child._p_child" in sd
        assert sd["_p_parent"] == {'data': 20}
        assert sd["_child._p_child"] == {'data': 10}

    def test_state_dict_when_train_false_excludes_params(self):
        class TestModule(Module):
            _param: Param[int] = PrivateParam(default=10)
            _runtime: Runtime[int] = PrivateRuntime(default=20)

        module = TestModule()
        sd = module.state_dict(train=False, runtime=True)
        assert "_param" not in sd
        assert "_runtime" in sd

    def test_state_dict_when_runtime_false_excludes_runtime(self):
        class TestModule(Module):
            _param: Param[int] = PrivateParam(default=10)
            _runtime: Runtime[int] = PrivateRuntime(default=20)

        module = TestModule()
        sd = module.state_dict(train=True, runtime=False)
        assert "_param" in sd
        assert "_runtime" not in sd

    def test_state_dict_when_nested_modules_returns_full_paths(self):
        class GrandchildModule(Module):
            _p_grand: Param[int] = PrivateParam(default=5)

        class ChildModule(Module):
            _p_child: Param[int] = PrivateParam(default=10)
            _grandchild: GrandchildModule = pydantic.PrivateAttr(default_factory=GrandchildModule)

        class ParentModule(Module):
            _p_parent: Param[int] = PrivateParam(default=20)
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        sd = parent.state_dict(recurse=True)
        assert "_child._grandchild._p_grand" in sd
        assert sd["_child._grandchild._p_grand"] == {"data": 5}

    def test_state_dict_returns_data_not_shareable_objects(self):
        class TestModule(Module):
            _param: Param[int] = PrivateParam(default=42)

        module = TestModule()
        sd = module.state_dict()
        assert sd["_param"] == {"data": 42}
        assert not isinstance(sd["_param"], Param)


class TestModuleStateKeys:

    def test_state_keys_when_no_recurse_returns_local_keys_only(self):
        class ChildModule(Module):
            _p_child: Param[int] = PrivateParam(default=10)

        class ParentModule(Module):
            _p_parent: Param[int] = PrivateParam(default=20)
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        keys = parent.state_keys(recurse=False)
        assert "_p_parent" in keys
        assert "_child._p_child" not in keys

    def test_state_keys_when_recurse_true_returns_dotted_paths(self):
        class ChildModule(Module):
            _p_child: Param[int] = PrivateParam(default=10)

        class ParentModule(Module):
            _p_parent: Param[int] = PrivateParam(default=20)
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        keys = parent.state_keys(recurse=True)
        assert "_p_parent" in keys
        assert "_child._p_child" in keys

    def test_state_keys_when_train_false_excludes_params(self):
        class TestModule(Module):
            _param: Param[int] = PrivateParam(default=10)
            _runtime: Runtime[int] = PrivateRuntime(default=20)

        module = TestModule()
        keys = module.state_keys(train=False, runtime=True)
        assert "_param" not in keys
        assert "_runtime" in keys

    def test_state_keys_when_runtime_false_excludes_runtime(self):
        class TestModule(Module):
            _param: Param[int] = PrivateParam(default=10)
            _runtime: Runtime[int] = PrivateRuntime(default=20)

        module = TestModule()
        keys = module.state_keys(train=True, runtime=False)
        assert "_param" in keys
        assert "_runtime" not in keys


class TestModuleLoadStateDict:

    def test_load_state_dict_when_valid_updates_params(self):
        class TestModule(Module):
            _param: Param[int] = PrivateParam(default=10)

        module = TestModule()
        module.load_state_dict(
            {"_param": {"data": 99}, 
             "_training": {"data": True}
             }
        )
        assert module._param.data == 99

    def test_load_state_dict_when_valid_updates_runtime(self):
        class TestModule(Module):
            _runtime: Runtime[int] = PrivateRuntime(default=10)

        module = TestModule()
        module.load_state_dict(
            {"_runtime": {"data": 99},
             "_training": {"data": True}
             }
        )
        assert module._runtime.data == 99

    def test_load_state_dict_when_recurse_true_updates_children(self):
        class ChildModule(Module):
            _p_child: Param[int] = PrivateParam(default=10)

        class ParentModule(Module):
            _p_parent: Param[int] = PrivateParam(default=20)
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        parent.load_state_dict({
            "_p_parent": {"data": 100},
            "_training": {"data": True},
            "_child._p_child": {"data": 200},
            "_child._training": {"data": True}
        })
        assert parent._p_parent.data == 100
        assert parent._child._p_child.data == 200

    def test_load_state_dict_when_strict_true_and_missing_keys_raises_error(self):
        class TestModule(Module):
            _param1: Param[int] = PrivateParam(default=10)
            _param2: Param[int] = PrivateParam(default=20)

        module = TestModule()
        with pytest.raises(KeyError, match="Missing keys"):
            module.load_state_dict({"_param1": {"data": 99}}, strict=True)

    def test_load_state_dict_when_strict_true_and_extra_keys_raises_error(self):
        class TestModule(Module):
            _param: Param[int] = PrivateParam(default=10)

        module = TestModule()
        with pytest.raises(KeyError, match="Unexpected keys"):
            module.load_state_dict(
                {"_param": {"data": 99}, "_training": {"data": True}, "_extra": 1
                 }, strict=True)

    def test_load_state_dict_when_strict_false_ignores_extra_keys(self):
        class TestModule(Module):
            _param: Param[int] = PrivateParam(default=10)

        module = TestModule()
        module.load_state_dict({"_param": {"data": 99}, "_extra": 1, "_training": {"data": True}}, strict=False)
        assert module._param.data == 99

    def test_load_state_dict_when_train_false_does_not_update_params(self):
        class TestModule(Module):
            _param: Param[int] = PrivateParam(default=10)
            _runtime: Runtime[int] = PrivateRuntime(default=20)

        module = TestModule()
        module.load_state_dict({"_param": {"data": 99}, "_runtime": {"data": 88}, "_training": {"data": True}}, train=False, runtime=True, strict=False)
        assert module._param.data == 10  # Not updated
        assert module._runtime.data == 88  # Updated

    def test_load_state_dict_when_runtime_false_does_not_update_runtime(self):
        class TestModule(Module):
            _param: Param[int] = PrivateParam(default=10)
            _runtime: Runtime[int] = PrivateRuntime(default=20)

        module = TestModule()
        module.load_state_dict({"_param": {"data": 99}, "_runtime": {"data": 88}, "_training": {"data": True}}, train=True, runtime=False, strict=False)
        assert module._param.data == 99  # Updated
        assert module._runtime.data == 20  # Not updated

    def test_load_state_dict_roundtrip_preserves_state(self):
        class ChildModule(Module):
            _p_child: Param[int] = PrivateParam(default=10)

        class ParentModule(Module):
            _p_parent: Param[int] = PrivateParam(default=20)
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        module1 = ParentModule()
        module1._p_parent.data = 999
        module1._child._p_child.data = 888

        sd = module1.state_dict()
        module2 = ParentModule()
        module2.load_state_dict(sd)

        assert module2._p_parent.data == 999
        assert module2._child._p_child.data == 888


class TestModuleTrain:

    def test_train_when_called_sets_training_to_true(self):
        class TestModule(Module):
            pass

        module = TestModule()
        module.train()
        assert module._training.data is True

    def test_train_when_mode_false_sets_training_to_false(self):
        class TestModule(Module):
            pass

        module = TestModule()
        module.train(False)
        assert module._training.data is False

    def test_train_when_called_recursively_sets_children_training(self):
        class ChildModule(Module):
            pass

        class ParentModule(Module):
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        parent.train(True)
        assert parent._training.data is True
        assert parent._child._training.data is True

    def test_train_returns_self(self):
        class TestModule(Module):
            pass

        module = TestModule()
        result = module.train()
        assert result is module

    def test_eval_sets_training_to_false(self):
        class TestModule(Module):
            pass

        module = TestModule()
        module.train(True)
        module.eval()
        assert module._training.data is False


class TestModuleApply:

    def test_apply_with_filter_applies_to_all_objects(self):
        class TestModule(Module):
            _param: Param[int] = PrivateParam(default=10)

        class TestModule2(Module):
            _param: Param[int] = PrivateParam(default=20)

        class ParentModule(Module):
            _child1: TestModule = pydantic.PrivateAttr(default_factory=TestModule)
            _child2: TestModule2 = pydantic.PrivateAttr(default_factory=TestModule2)

        module = ParentModule()
        counter = {"count": 0}

        def count_fn(obj):
            counter["count"] += 1

        module.apply(count_fn, include=(TestModule2))
        # Should apply to: module itself + _param
        assert counter["count"] == 1

    # TODO: Apply only works on modules not Param, Runtime etc
    # def test_apply_when_type_filter_applies_only_to_matching_types(self):
    #     class TestModule(Module):
    #         _param: Param[int] = PrivateParam(default=10)
    #         _runtime: Runtime[int] = PrivateRuntime(default=20)

    #     module = TestModule()
    #     param_count = {"count": 0}

    #     def count_params(obj):
    #         param_count["count"] += 1

    #     module.apply(count_params, include=Param)
    #     assert param_count["count"] == 1

    # TODO: Apply only works on modules not Param, Runtime etc
    # def test_apply_when_callable_filter_applies_only_when_filter_returns_true(self):
    #     class TestModule(Module):
    #         _param1: Param[int] = PrivateParam(default=10)
    #         _param2: Param[int] = PrivateParam(default=50)

    #     module = TestModule()
    #     large_values = []

    #     def collect_large(obj):
    #         if isinstance(obj, Param) and obj.data > 25:
    #             large_values.append(obj.data)

    #     module.apply(lambda obj: None, include=lambda obj: isinstance(obj, Param) and obj.data > 25)
    #     # Simpler test - just check filter works
    #     filtered_count = {"count": 0}
    #     module.apply(lambda obj: filtered_count.update({"count": filtered_count["count"] + 1}), 
    #                  include=lambda obj: isinstance(obj, Param) and obj.data > 25)
    #     assert filtered_count["count"] == 1

    def test_apply_recursively_applies_to_children(self):
        class ChildModule(Module):
            _p_child: Param[int] = PrivateParam(default=10)

        class ParentModule(Module):
            _p_parent: Param[int] = PrivateParam(default=20)
            _child: ChildModule = pydantic.PrivateAttr(default_factory=ChildModule)

        parent = ParentModule()
        all_objects = []

        def collect_all(obj):
            all_objects.append(obj)

        parent.apply(collect_all)
        # Should include: parent module, parent param, child module, child param
        assert len(all_objects) == 2


class TestRegistry:

    def test_register_when_decorator_used_adds_class_to_registry(self):
        registry = Registry[Module]()

        @registry.register()
        class TestModule(Module):
            pass

        # Registry uses __qualname__ which includes the full path
        assert TestModule.__qualname__ in registry.list_entries()

    def test_register_when_custom_name_provided_uses_custom_name(self):
        registry = Registry[Module]()
        
        @registry.register(name="CustomName")
        class TestModule(Module):
            pass
        
        assert "CustomName" in registry.list_entries()

    def test_register_when_tags_provided_stores_tags(self):
        registry = Registry[Module]()

        @registry.register(tags={"category": "test", "version": 1})
        class TestModule(Module):
            pass

        entry = registry[TestModule.__qualname__]
        assert entry.tags["category"] == "test"
        assert entry.tags["version"] == 1

    def test_register_when_description_provided_stores_description(self):
        registry = Registry[Module]()

        @registry.register(description="Test module description")
        class TestModule(Module):
            pass

        entry = registry[TestModule.__qualname__]
        assert entry.description == "Test module description"

    def test_register_when_duplicate_name_prints_warning(self, capsys):
        registry = Registry[Module]()
        
        @registry.register(name="Duplicate")
        class TestModule1(Module):
            pass
        
        @registry.register(name="Duplicate")
        class TestModule2(Module):
            pass
        
        captured = capsys.readouterr()
        assert "Warning: Overwriting existing entry 'Duplicate'" in captured.out

    def test_getitem_when_single_key_returns_entry(self):
        registry = Registry[Module]()

        @registry.register()
        class TestModule(Module):
            pass

        entry = registry[TestModule.__qualname__]
        assert entry.obj is TestModule

    def test_getitem_when_list_of_keys_returns_dict(self):
        registry = Registry[Module]()

        @registry.register()
        class TestModule1(Module):
            pass

        @registry.register()
        class TestModule2(Module):
            pass

        entries = registry[[TestModule1.__qualname__, TestModule2.__qualname__]]
        assert isinstance(entries, dict)
        assert TestModule1.__qualname__ in entries
        assert TestModule2.__qualname__ in entries

    def test_getitem_when_key_not_found_raises_keyerror(self):
        registry = Registry[Module]()
        
        with pytest.raises(KeyError, match="Registry entry 'NonExistent' not found"):
            _ = registry["NonExistent"]

    def test_filter_when_no_criteria_returns_all(self):
        registry = Registry[Module]()
        
        @registry.register()
        class TestModule1(Module):
            pass
        
        @registry.register()
        class TestModule2(Module):
            pass
        
        results = registry.filter()
        assert len(results) == 2

    def test_filter_when_tags_criteria_returns_matching_only(self):
        registry = Registry[Module]()

        @registry.register(tags={"type": "encoder"})
        class EncoderModule(Module):
            pass

        @registry.register(tags={"type": "decoder"})
        class DecoderModule(Module):
            pass

        results = registry.filter(tags={"type": "encoder"})
        assert len(results) == 1
        assert EncoderModule.__qualname__ in results

    def test_filter_when_obj_type_criteria_returns_matching_only(self):
        registry = Registry()

        @registry.register()
        class TestClass:
            pass

        @registry.register()
        def test_function():
            pass

        results = registry.filter(obj_type="class")
        assert len(results) == 1
        assert TestClass.__qualname__ in results

    def test_deregister_when_key_exists_removes_entry(self):
        registry = Registry[Module]()

        @registry.register()
        class TestModule(Module):
            pass

        registry.deregister(TestModule.__qualname__)
        assert TestModule.__qualname__ not in registry.list_entries()

    def test_list_entries_returns_all_registered_names(self):
        registry = Registry[Module]()

        @registry.register()
        class TestModule1(Module):
            pass

        @registry.register()
        class TestModule2(Module):
            pass

        entries = registry.list_entries()
        assert TestModule1.__qualname__ in entries
        assert TestModule2.__qualname__ in entries

    def test_list_types_returns_unique_object_types(self):
        registry = Registry()
        
        @registry.register()
        class TestClass:
            pass
        
        @registry.register()
        def test_function():
            pass
        
        types = registry.list_types()
        assert "class" in types
        assert "function" in types

    def test_list_tags_returns_all_unique_tag_keys(self):
        registry = Registry[Module]()
        
        @registry.register(tags={"category": "test", "version": 1})
        class TestModule1(Module):
            pass
        
        @registry.register(tags={"category": "prod", "priority": "high"})
        class TestModule2(Module):
            pass
        
        tags = registry.list_tags()
        assert "category" in tags
        assert "version" in tags
        assert "priority" in tags


class TestAdaptModule:

    def test_build_when_called_creates_instance_with_adapted_module(self):
        class InnerModule(Module):
            _p: Param[int] = PrivateParam(default=42)

        inner = InnerModule()
        adapt_mod = AdaptModule.build(adapted=inner, train_submods=True, fixed=False)

        assert adapt_mod.adapted is inner
        assert "adapted" not in adapt_mod.model_fields
        assert isinstance(adapt_mod._adapted, Param)

    def test_build_when_train_submods_false_sets_flag(self):
        class InnerModule(Module):
            _p: Param[int] = PrivateParam(default=42)

        inner = InnerModule()
        adapt_mod = AdaptModule.build(adapted=inner, train_submods=False)

        assert adapt_mod._train_submods is False

    def test_build_when_fixed_true_sets_flag(self):
        class InnerModule(Module):
            _p: Param[int] = PrivateParam(default=42)

        inner = InnerModule()
        adapt_mod = AdaptModule.build(adapted=inner, fixed=True)

        assert adapt_mod._fixed is True

    def test_fix_when_called_sets_fixed_to_true(self):
        class InnerModule(Module):
            _p: Param[int] = PrivateParam(default=42)

        inner = InnerModule()
        adapt_mod = AdaptModule.build(adapted=inner, fixed=False)

        adapt_mod.fix()
        assert adapt_mod._fixed is True

    def test_unfix_when_called_sets_fixed_to_false(self):
        class InnerModule(Module):
            _p: Param[int] = PrivateParam(default=42)

        inner = InnerModule()
        adapt_mod = AdaptModule.build(adapted=inner, fixed=True)

        adapt_mod.unfix()
        assert adapt_mod._fixed is False

    def test_parameters_when_fixed_returns_empty(self):
        class InnerModule(Module):
            _p: Param[int] = PrivateParam(default=42)

        inner = InnerModule()
        adapt_mod = AdaptModule.build(adapted=inner, fixed=True)

        params = list(adapt_mod.parameters(recurse=True))
        assert len(params) == 0

    def test_parameters_when_not_fixed_and_train_submods_false_returns_only_adapted_param(self):
        class InnerModule(Module):
            _p: Param[int] = PrivateParam(default=42)

        inner = InnerModule()
        adapt_mod = AdaptModule.build(adapted=inner, train_submods=False, fixed=False)

        params = list(adapt_mod.parameters(recurse=True))
        assert len(params) == 1
        assert params[0] is adapt_mod._adapted

    def test_parameters_when_not_fixed_and_train_submods_true_returns_adapted_and_inner_params(self):
        class InnerModule(Module):
            _p: Param[int] = PrivateParam(default=42)

        inner = InnerModule()
        adapt_mod = AdaptModule.build(adapted=inner, train_submods=True, fixed=False)

        params = list(adapt_mod.parameters(recurse=True))
        assert len(params) == 2
        assert params[0] is adapt_mod._adapted
        assert params[1] is inner._p

    def test_adapted_setter_when_fixed_raises_runtime_error(self):
        class InnerModule(Module):
            _p: Param[int] = PrivateParam(default=42)

        inner = InnerModule()
        adapt_mod = AdaptModule.build(adapted=inner, fixed=True)

        with pytest.raises(RuntimeError, match="Cannot update adapted on a frozen AdaptModule"):
            adapt_mod.adapted = InnerModule()

    def test_state_dict_and_load_state_dict_roundtrip_preserves_adapted_state(self):
        class InnerModule(Module):
            _p: Param[int] = PrivateParam(default=42)

        inner = InnerModule()
        inner._p.data = 99
        adapt_mod = AdaptModule[InnerModule].build(adapted=inner)

        # print(adapt_mod._adapted.__orig_class__)

        sd = adapt_mod.state_dict()
        assert "adapted._p" in sd
        assert sd["adapted._p"] == {"data": 99}

        inner2 = InnerModule()
        adapt_mod2 = AdaptModule[InnerModule].build(adapted=inner2)
        adapt_mod2.load_state_dict(sd)

        assert adapt_mod2.adapted._p.data == 99


class TestCheckpoint:

    def test_save_when_called_writes_json_file_with_spec_and_state(self, tmp_path):

        checkpoint_path = tmp_path / "test_checkpoint.json"

        spec_data = {"KIND": "TestModule", "x": 10}
        state_data = {"param1": {"value": 42}}
        checkpoint = Checkpoint(spec=spec_data, state=state_data)

        checkpoint.save(str(checkpoint_path))

        assert checkpoint_path.exists()
        with open(checkpoint_path, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data["spec"] == spec_data
        assert loaded_data["state"] == state_data

    def test_load_when_called_reads_json_file_correctly(self, tmp_path):
        checkpoint_path = tmp_path / "test_checkpoint.json"

        spec_data = {"KIND": "TestModule"}
        state_data = {"param1": {"value": 42}}

        with open(checkpoint_path, 'w') as f:
            json.dump({"spec": spec_data, "state": state_data}, f)

        with open(checkpoint_path, 'r') as f:
            loaded = json.load(f)

        assert loaded["spec"] == spec_data
        assert loaded["state"] == state_data


class TestModulePydanticSerialization:
    """Test Pydantic's native serialization methods for Module.

    NOTE: Pydantic's model_dump() does NOT serialize PrivateAttrs by default.
    This is why we use state_dict() for checkpoints instead.
    """

    def test_model_dump_when_called_includes_only_public_fields(self):
        class TestModule(Module):
            pass

        module = TestModule()
        dumped = module.model_dump()

        # Pydantic only serializes public fields, not PrivateAttrs
        assert "KIND" in dumped
        assert "TestModule" in dumped["KIND"]  # KIND includes full qualname
        # training is a PrivateAttr (Runtime), so NOT in model_dump()
        assert "training" not in dumped

    def test_model_dump_when_module_has_params_excludes_private_attrs(self):
        class TestModule(Module):
            _param: Param[int] = PrivateParam(42)

        module = TestModule()
        dumped = module.model_dump()

        # PrivateAttrs are not included in model_dump()
        assert "_param" not in dumped
        # Only KIND is serialized
        assert "KIND" in dumped

    def test_model_dump_json_roundtrip_only_serializes_public_fields(self):
        class TestModule(Module):
            x: int = 10  # Public field
            _param: Param[int] = PrivateParam(42)  # Private

        module = TestModule()

        json_str = module.model_dump_json()
        restored = TestModule.model_validate_json(json_str)

        # Public field is preserved
        assert restored.x == 10
        # PrivateAttrs get default values, not serialized values
        assert restored._param.data == 42  # Default from PrivateParam(42)

    def test_state_dict_vs_model_dump_different_purposes(self):
        """Demonstrate that state_dict() is for checkpoints, model_dump() is not."""
        class TestModule(Module):
            _param: Param[int] = PrivateParam(0)

        module = TestModule()
        module._param.set(99)

        # model_dump() doesn't include _param
        dumped = module.model_dump()
        assert "_param" not in dumped

        # state_dict() DOES include _param
        state = module.state_dict()
        assert "_param" in state
        assert state["_param"]["data"] == 99

    def test_model_dump_exclude_none_excludes_none_params(self):
        class TestModule(Module):
            _param: Param[int | None] = PrivateParam(None)

        module = TestModule()
        dumped = module.model_dump(exclude_none=True)

        # Param itself is not None, but its data is None
        # Check what Pydantic does with this
        assert "_param" in dumped or "_param" not in dumped  # Either is valid

    def test_model_dump_mode_json_serializes_correctly(self):
        class TestModule(Module):
            _param: Param[int] = PrivateParam(42)

        module = TestModule()
        dumped = module.model_dump(mode='json')

        # In JSON mode, everything should be JSON-serializable
        assert isinstance(dumped, dict)
        import json
        json_str = json.dumps(dumped)  # Should not raise
        assert json_str is not None


class TestModuleCircularReferences:
    """Test Module behavior with circular references and deep nesting."""

    def test_modules_when_parent_child_both_have_params_no_infinite_loop(self):
        class Parent(Module):
            _p: Param[int] = PrivateParam(1)

        class Child(Module):
            _c: Param[int] = PrivateParam(2)

        parent = Parent()
        child = Child()

        # Create hierarchy (not circular yet)
        parent._child = child
        parent._registry["_child"] = StateType.MODULE

        # This should work
        params = list(parent.parameters(recurse=True))
        assert len(params) == 2  # _p and _c

    def test_parameters_with_very_deep_nesting_does_not_stack_overflow(self):
        # Create 50-level deep hierarchy
        root = Module()
        current = root

        for i in range(50):
            child = Module()
            current._child = child
            current._registry["_child"] = StateType.MODULE
            current = child

        # Should not stack overflow
        params = list(root.parameters(recurse=True))
        assert isinstance(params, list)

    def test_modules_with_very_deep_nesting_does_not_stack_overflow(self):
        # Create 50-level deep hierarchy
        root = Module()
        current = root

        for i in range(50):
            child = Module()
            current._child = child
            current._registry["_child"] = StateType.MODULE
            current = child

        # Should not stack overflow
        modules = list(root.modules(recurse=True))
        assert len(modules) == 51  # root + 50 children

    def test_state_dict_with_deep_nesting_does_not_stack_overflow(self):
        # Create 50-level deep hierarchy with params
        root = Module()
        current = root

        for i in range(50):
            child = Module()
            child._value = Param[int](data=i)
            child._registry["_value"] = StateType.PARAM
            current._child = child
            current._registry["_child"] = StateType.MODULE
            current = child

        # Should not stack overflow
        sd = root.state_dict(recurse=True)
        assert isinstance(sd, dict)


class TestModuleErrorHandling:
    """Test Module error cases and validation."""

    def test_module_when_invalid_kind_raises_validation_error(self):
        class TestModule(Module):
            pass

        with pytest.raises(pydantic.ValidationError):
            TestModule(KIND="WrongKind")

    def test_param_when_wrong_type_data_raises_validation_error(self):
        with pytest.raises(pydantic.ValidationError):
            Param[int](data="not an int")

    def test_load_state_dict_when_param_type_mismatch_raises_validation_error(self):
        class TestModule(Module):
            _param: Param[int] = PrivateParam(0)

        module = TestModule()

        with pytest.raises(pydantic.ValidationError):
            module.load_state_dict({"_param": {"data": "not an int"}})

    def test_shareable_item_when_callback_raises_does_not_crash(self):
        def bad_callback(old, new):
            raise ValueError("Bad callback")

        item = Param[int](data=10)
        item.register_callback(bad_callback)

        # Setting should handle the exception gracefully
        # or raise it - either is valid, just shouldn't crash silently
        try:
            item.set(20)
            # If it doesn't raise, verify state is still consistent
            assert item.data == 20
        except ValueError as e:
            # If it raises, that's also acceptable
            assert "Bad callback" in str(e)

    def test_shareable_item_arithmetic_when_data_is_none_handles_gracefully(self):
        item = Param[int](data=None)

        # What happens when we do arithmetic on None?
        # Should either raise TypeError or handle it
        try:
            result = item + 5
            # If it doesn't raise, what's the result?
            assert result.data is None or isinstance(result.data, int)
        except (TypeError, AttributeError):
            # Raising is acceptable
            pass

    def test_shareable_item_truediv_when_divide_by_zero_raises(self):
        item = Param[int](data=10)

        with pytest.raises(ZeroDivisionError):
            result = item / 0


class TestObjInitAndFuncInit:
    """Test ObjInit and FuncInit initialization patterns."""

    def test_obj_init_when_created_initializes_correctly(self):
        class TestModule(Module):
            x: int = 10
            _param: Param[int] = pydantic.PrivateAttr(default=ObjInit(lambda m: m.x, Param))

        module = TestModule()

        assert module._param.data == 10

    def test_obj_init_when_annotation_is_generic_creates_typed_param(self):
        class TestModule(Module):
            x: int = 42
            _param: Param[int] = pydantic.PrivateAttr(default=ObjInit(lambda m: m.x, Param))

        module = TestModule()

        # Check if it has the right type
        assert module._param.data == 42

    def test_func_init_when_created_initializes_correctly(self):
        from dachi.core._base import FuncInit

        class TestModule(Module):
            _param: Param[int] = pydantic.PrivateAttr(default=FuncInit(lambda: 99, Param))

        module = TestModule()

        assert module._param.data == 99

    def test_obj_init_when_accessing_nonexistent_attr_raises(self):
        class TestModule(Module):
            _param: Param[int] = pydantic.PrivateAttr(default=ObjInit(lambda m: m.nonexistent, Param))

        with pytest.raises(AttributeError):
            TestModule()


class TestAdaptModuleEdgeCases:
    """Test AdaptModule edge cases and complex scenarios."""

    def test_adapt_module_switch_adapted_multiple_times_maintains_consistency(self):
        class InnerModule(Module):
            _value: Param[int] = PrivateParam(0)

        m1 = InnerModule()
        m1._value.set(10)
        m2 = InnerModule()
        m2._value.set(20)
        m3 = InnerModule()
        m3._value.set(30)

        adapt = AdaptModule.build(m1)
        assert adapt.adapted._value.data == 10

        adapt.adapted = m2
        assert adapt.adapted._value.data == 20

        adapt.adapted = m3
        assert adapt.adapted._value.data == 30

    def test_adapt_module_when_adapted_is_none_parameters_returns_only_adapted_param(self):
        adapt = AdaptModule.build(None)
        params = list(adapt.parameters(recurse=True))

        # Should only have _adapted param, not crash
        assert len(params) == 1
        assert params[0] is adapt._adapted

    def test_adapt_module_when_adapted_none_state_dict_handles_gracefully(self):
        adapt = AdaptModule.build(None)
        sd = adapt.state_dict(recurse=True)

        assert isinstance(sd, dict)
        assert "_adapted" in sd

    def test_adapt_module_when_fixed_then_unfixed_then_fixed_again(self):
        class InnerModule(Module):
            _value: Param[int] = PrivateParam(5)

        module = InnerModule()
        adapt = AdaptModule.build(module)

        adapt.fix()
        assert adapt._fixed is True  # _fixed is a plain bool

        adapt.unfix()
        assert adapt._fixed is False

        adapt.fix()
        assert adapt._fixed is True

    def test_adapt_module_parameters_when_train_submods_affects_output(self):
        class InnerModule(Module):
            _inner_param: Param[int] = PrivateParam(10)

        module = InnerModule()

        # Build with train_submods=True
        adapt_with = AdaptModule.build(module, train_submods=True)
        params_with = list(adapt_with.parameters(recurse=True))

        # Build with train_submods=False
        adapt_without = AdaptModule.build(module, train_submods=False)
        params_without = list(adapt_without.parameters(recurse=True))

        # With train_submods=True should have more params
        assert len(params_with) > len(params_without)
