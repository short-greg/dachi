import pytest
from pydantic import ValidationError
from typing import List
from dachi.core._old_base2 import BaseItem  # Replace with actual import path


class TestBaseItem:

    def test_cannot_instantiate_baseitem(self):
        with pytest.raises(TypeError, match="BaseItem is abstract"):
            BaseItem()

    def test_subclass_creation_and_spec_generation(self):
        class MyStruct(BaseItem):
            x: int
            y: str = "default"

        # Instantiate
        obj = MyStruct(10)
        assert obj.x == 10
        assert obj.y == "default"

        # Schema generation
        schema_cls = MyStruct.to_schema()
        schema = schema_cls.model_json_schema()
        props = schema['properties']
        assert 'x' in props and props['x']['type'] == 'integer'
        assert 'y' in props and props['y']['type'] == 'string'

        # Spec generation
        spec = obj.to_spec()
        assert spec.x == 10
        assert spec.y == "default"
        assert spec.kind == "TestBaseItem.test_subclass_creation_and_spec_generation.<locals>.MyStruct"

    def test_nested_baseitem(self):
        class Child(BaseItem):
            a: int

        class Parent(BaseItem):
            child: Child
            value: str

        child = Child(5)
        parent = Parent(child, "hi")
        spec = parent.to_spec()
        assert isinstance(spec.child, Child.to_schema())
        assert spec.child.a == 5
        assert spec.value == "hi"

    def test_post_init_called(self):
        called = {}

        class MyItem(BaseItem):
            x: int

            def __post_init__(self):
                called['done'] = True

        _ = MyItem(42)
        assert called.get('done') is True

    def test_invalid_spec_generation_raises(self):
        class NotBaseItem:
            x: int

        with pytest.raises(TypeError):
            NotBaseItem.to_schema()  # Should fail since it's not a BaseItem

    def test_pydantic_core_schema_hook(self):
        class MyData(BaseItem):
            foo: str

        # This should delegate to handler.generate_schema(...)
        class DummyHandler:
            def generate_schema(self, typ):
                return {"mock": f"schema_for_{typ.__name__}"}

        schema = MyData.__get_pydantic_core_schema__(None, DummyHandler())
        assert schema['mock'] == "schema_for_MyData"

    def test_spec_generation_excludes_private_fields(self):
        class HiddenFieldItem(BaseItem):
            x: int
            _y: str = "hidden"

        obj = HiddenFieldItem(1)
        spec = obj.to_spec()
        assert hasattr(spec, 'x')
        assert not hasattr(spec, '_y')

    def test_schema_cache(self):
        # Simulate multiple calls to to_schema() and ensure the schema is reused
        class Reused(BaseItem):
            x: int

        schema1 = Reused.to_schema()
        schema2 = Reused.to_schema()
        assert schema1 is schema2  # Should be same object