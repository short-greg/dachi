import pytest
import typing
from dachi.core._base import (
    BaseModule,
    BaseFieldDescriptor,
    ModFieldDescriptor,
    modfield,
    RestrictedSchemaMixin,
)
from dachi.core import (
    ModuleList,
    ModuleDict,
    ModListFieldDescriptor,
    ModDictFieldDescriptor,
    modlistfield,
    moddictfield,
)


class SimpleModule(BaseModule):
    value: int


class RestrictedModule(BaseModule, RestrictedSchemaMixin):
    data: str

    @classmethod
    def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
        return cls.schema()


class TestBaseFieldDescriptor:
    """Test BaseFieldDescriptor abstract base class"""

    def test_base_field_descriptor_is_abstract(self):
        """BaseFieldDescriptor should be abstract and not instantiable directly"""
        with pytest.raises(TypeError):
            BaseFieldDescriptor(typ=SimpleModule)

    def test_descriptor_stores_owner_and_name_on_set_name(self):
        """__set_name__ should store owner class and field name"""
        descriptor = ModFieldDescriptor(typ=SimpleModule)

        class TestClass(BaseModule):
            field: SimpleModule = descriptor

        assert descriptor._owner is TestClass
        assert descriptor._name == "field"


class TestModFieldDescriptor:
    """Test ModFieldDescriptor for single module fields"""

    def test_modfield_descriptor_creation(self):
        """Can create ModFieldDescriptor"""
        descriptor = ModFieldDescriptor(typ=SimpleModule)
        assert isinstance(descriptor, BaseFieldDescriptor)

    def test_modfield_validates_type_annotation_on_set_name(self):
        """ModFieldDescriptor should validate type is a BaseModule subclass"""

        with pytest.raises(TypeError, match="must be a BaseModule subclass"):

            class TestClass(BaseModule):
                # int is not a BaseModule
                field: int = ModFieldDescriptor(typ=int)

    def test_modfield_allows_union_of_modules(self):
        """ModFieldDescriptor should allow Union of BaseModule subclasses"""

        class Module1(BaseModule):
            x: int

        class Module2(BaseModule):
            y: str

        # Should not raise
        class TestClass(BaseModule):
            field: Module1 | Module2 = ModFieldDescriptor(typ=Module1 | Module2)

    def test_modfield_get_returns_value(self):
        """ModFieldDescriptor.__get__ should return the stored value"""

        class TestClass(BaseModule):
            field: SimpleModule = modfield()

        instance = TestClass(field=SimpleModule(value=42))
        assert isinstance(instance.field, SimpleModule)
        assert instance.field.value == 42

    def test_modfield_set_stores_value(self):
        """ModFieldDescriptor.__set__ should store the value"""

        class TestClass(BaseModule):
            field: SimpleModule = modfield()

        instance = TestClass(field=SimpleModule(value=10))
        new_module = SimpleModule(value=20)
        instance.field = new_module
        assert instance.field is new_module

    def test_modfield_restricted_schema_with_no_variants_returns_base_schema(self):
        """restricted_schema() with variants=None should return unrestricted schema"""

        class TestClass(BaseModule, RestrictedSchemaMixin):
            field: RestrictedModule = modfield()

            @classmethod
            def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
                return cls.schema()

        descriptor = TestClass.field  # Get the descriptor from class
        field_schema, defs = descriptor.restricted_schema(
            filter_schema_cls=RestrictedSchemaMixin, variants=None
        )

        # Should return base schema for the field
        assert "anyOf" in field_schema or "$ref" in field_schema or "type" in field_schema

    def test_modfield_restricted_schema_with_variants_creates_union(self):
        """restricted_schema() with variants should create union of variant schemas"""

        class Module1(BaseModule, RestrictedSchemaMixin):
            x: int

            @classmethod
            def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
                return cls.schema()

        class Module2(BaseModule, RestrictedSchemaMixin):
            y: str

            @classmethod
            def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
                return cls.schema()

        class TestClass(BaseModule, RestrictedSchemaMixin):
            field: Module1 | Module2 = modfield()

            @classmethod
            def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
                return cls.schema()

        descriptor = TestClass.field
        field_schema, defs = descriptor.restricted_schema(
            filter_schema_cls=RestrictedSchemaMixin, variants=[Module1, Module2], _profile="shared"
        )

        # Should create $ref to union in defs
        assert "$ref" in field_schema
        assert len(defs) > 0

    def test_modfield_restricted_schema_inline_profile(self):
        """restricted_schema() with _profile='inline' should create inline oneOf"""

        class Module1(BaseModule, RestrictedSchemaMixin):
            x: int

            @classmethod
            def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
                return cls.schema()

        class Module2(BaseModule, RestrictedSchemaMixin):
            y: str

            @classmethod
            def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
                return cls.schema()

        class TestClass(BaseModule, RestrictedSchemaMixin):
            field: Module1 | Module2 = modfield()

            @classmethod
            def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
                return cls.schema()

        descriptor = TestClass.field
        field_schema, defs = descriptor.restricted_schema(
            filter_schema_cls=RestrictedSchemaMixin, variants=[Module1, Module2], _profile="inline"
        )

        # Should create inline oneOf, not $ref
        assert "oneOf" in field_schema
        assert defs == {}


class TestModListFieldDescriptor:
    """Test ModListFieldDescriptor for ModuleList fields"""

    def test_modlistfield_descriptor_creation(self):
        """Can create ModListFieldDescriptor"""
        descriptor = ModListFieldDescriptor(typ=typing.List[SimpleModule])
        assert isinstance(descriptor, BaseFieldDescriptor)

    def test_modlistfield_restricted_schema_with_variants_creates_array_union(self):
        """restricted_schema() should create array with union of variants"""

        class Module1(BaseModule, RestrictedSchemaMixin):
            x: int

            @classmethod
            def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
                return cls.schema()

        class Module2(BaseModule, RestrictedSchemaMixin):
            y: str

            @classmethod
            def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
                return cls.schema()

        class TestClass(BaseModule, RestrictedSchemaMixin):
            field: ModuleList[Module1 | Module2] = modlistfield()

            @classmethod
            def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
                return cls.schema()

        descriptor = TestClass.field
        field_schema, defs = descriptor.restricted_schema(
            filter_schema_cls=RestrictedSchemaMixin, variants=[Module1, Module2], _profile="shared"
        )

        # Should be array type
        assert field_schema["type"] == "array"
        # Items should reference union
        assert "$ref" in field_schema["items"]
        assert len(defs) > 0


class TestModDictFieldDescriptor:
    """Test ModDictFieldDescriptor for ModuleDict fields"""

    def test_moddictfield_descriptor_creation(self):
        """Can create ModDictFieldDescriptor"""
        descriptor = ModDictFieldDescriptor()
        assert isinstance(descriptor, BaseFieldDescriptor)

    def test_moddictfield_validates_key_type(self):
        """ModDictFieldDescriptor should only allow str or int keys"""

        with pytest.raises(TypeError, match="key type must be str or int"):

            class TestClass(BaseModule):
                field: ModuleDict[float, SimpleModule] = moddictfield()

    def test_moddictfield_restricted_schema_with_variants_creates_dict_union(self):
        """restricted_schema() should create object with additionalProperties union"""

        class Module1(BaseModule, RestrictedSchemaMixin):
            x: int

            @classmethod
            def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
                return cls.schema()

        class Module2(BaseModule, RestrictedSchemaMixin):
            y: str

            @classmethod
            def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
                return cls.schema()

        class TestClass(BaseModule, RestrictedSchemaMixin):
            field: ModuleDict[str, Module1 | Module2] = moddictfield()

            @classmethod
            def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
                return cls.schema()

        descriptor = TestClass.field
        field_schema, defs = descriptor.restricted_schema(
            filter_schema_cls=RestrictedSchemaMixin, variants=[Module1, Module2], _profile="shared"
        )

        # Should be object type
        assert field_schema["type"] == "object"
        # additionalProperties should reference union
        assert "$ref" in field_schema["additionalProperties"]
        assert len(defs) > 0


class TestModFieldFactories:
    """Test factory functions modfield(), modlistfield(), moddictfield()"""

    def test_modfield_factory_creates_descriptor(self):
        """modfield() should create ModFieldDescriptor"""
        descriptor = modfield()
        assert isinstance(descriptor, ModFieldDescriptor)

    def test_modlistfield_factory_creates_descriptor(self):
        """modlistfield() should create ModListFieldDescriptor"""
        descriptor = modlistfield()
        assert isinstance(descriptor, ModListFieldDescriptor)

    def test_moddictfield_factory_creates_descriptor(self):
        """moddictfield() should create ModDictFieldDescriptor"""
        descriptor = moddictfield()
        assert isinstance(descriptor, ModDictFieldDescriptor)

    def test_modfield_factory_usage_in_class(self):
        """modfield() can be used as field default in BaseModule"""

        class TestClass(BaseModule):
            field: SimpleModule = modfield()

        instance = TestClass(field=SimpleModule(value=99))
        assert instance.field.value == 99
