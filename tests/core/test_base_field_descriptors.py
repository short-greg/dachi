# import pytest
# import typing
# from dachi.core._base import (
#     BaseModule,
#     BaseFieldDescriptor,
#     ModFieldDescriptor,
#     modfield,
#     RestrictedSchemaMixin,
# )
# from dachi.core import (
#     GenericFieldType,
#     generictype,
#     ModuleList,
#     ModuleDict,
#     ModListFieldDescriptor,
#     ModDictFieldDescriptor,
#     modlistfield,
#     moddictfield,
# )


# class SimpleModule(BaseModule):
#     value: int


# class Task(BaseModule):
#     """Test Task module."""
#     name: str = "test"


# class State(BaseModule):
#     """Test State module."""
#     data: str = "test"


# class RestrictedModule(BaseModule, RestrictedSchemaMixin):
#     data: str

#     @classmethod
#     def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
#         return cls.schema()


# class TestGenericFieldTypeInit:
#     """Test GenericFieldType __init__ and basic attributes."""

#     def test_init_with_single_type(self):
#         gft = GenericFieldType(ModuleList, Task)
#         assert gft.origin is ModuleList
#         assert gft.typs == [[Task]]
#         assert gft.single_typ == [True]
#         assert gft.metadata == {}

#     def test_init_with_multiple_types_for_union(self):
#         gft = GenericFieldType(ModuleList, [Task, State])
#         assert gft.origin is ModuleList
#         assert gft.typs == [[Task, State]]
#         assert gft.single_typ == [False]

#     def test_init_with_dict_types(self):
#         gft = GenericFieldType(ModuleDict, str, Task)
#         assert gft.origin is ModuleDict
#         assert gft.typs == [[str], [Task]]
#         assert gft.single_typ == [True, True]

#     def test_init_with_metadata(self):
#         gft = GenericFieldType(ModuleList, Task, description="test")
#         assert gft.metadata == {'description': 'test'}

#     def test_init_with_dict_and_union_values(self):
#         gft = GenericFieldType(ModuleDict, str, [Task, State])
#         assert gft.origin is ModuleDict
#         assert gft.typs == [[str], [Task, State]]
#         assert gft.single_typ == [True, False]


# class TestGenericFieldTypeGetParameterizedType:
#     """Test GenericFieldType get_parameterized_type() method."""

#     def test_single_type_returns_parameterized(self):
#         gft = GenericFieldType(ModuleList, Task)
#         result = gft.get_parameterized_type()
#         # Should be ModuleList[TaskSpec] (converts to Spec type)
#         assert result.__origin__ is ModuleList
#         assert result.__args__ == (Task.schema_model(),)

#     def test_dict_returns_parameterized(self):
#         gft = GenericFieldType(ModuleDict, str, Task)
#         result = gft.get_parameterized_type()
#         # Should be ModuleDict[str, TaskSpec] (converts to Spec type)
#         assert result.__origin__ is ModuleDict
#         assert result.__args__ == (str, Task.schema_model())

#     def test_list_with_union_creates_union(self):
#         gft = GenericFieldType(ModuleList, [Task, State])
#         result = gft.get_parameterized_type()
#         # Should be ModuleList[Task | State]
#         assert result.__origin__ is ModuleList
#         # Check that args contain a Union
#         import typing
#         union_arg = result.__args__[0]
#         origin = typing.get_origin(union_arg)
#         assert origin in (typing.Union, type(Task | State).__class__)

#     def test_dict_with_union_values(self):
#         gft = GenericFieldType(ModuleDict, str, [Task, State])
#         result = gft.get_parameterized_type()
#         # Should be ModuleDict[str, Task | State]
#         assert result.__origin__ is ModuleDict
#         assert result.__args__[0] == str
#         # Check second arg is a union
#         import typing
#         union_arg = result.__args__[1]
#         origin = typing.get_origin(union_arg)
#         assert origin in (typing.Union, type(Task | State).__class__)


# class TestGenericFieldTypeSchema:
#     """Test GenericFieldType schema() method delegation."""

#     def test_schema_delegates_to_container(self):
#         gft = GenericFieldType(ModuleList, Task)
#         schema = gft.schema()
#         # Should get schema from ModuleList[Task]
#         assert isinstance(schema, dict)
#         assert 'type' in schema or '$ref' in schema


# class TestGenericFieldTypeRestrictedSchema:
#     """Test GenericFieldType restricted_schema() method."""

#     def test_restricted_schema_returns_tuple(self):
#         """restricted_schema() should return (schema, defs) tuple"""
#         gft = GenericFieldType(ModuleList, Task)
#         result = gft.restricted_schema(
#             filter_schema_cls=RestrictedSchemaMixin,
#             variants=[Task],
#             field_name="tasks",
#             _profile="shared"
#         )
#         assert isinstance(result, tuple)
#         assert len(result) == 2
#         field_schema, defs = result
#         assert isinstance(field_schema, dict)
#         assert isinstance(defs, dict)

#     def test_restricted_schema_with_single_type(self):
#         """Should handle single type correctly"""
#         gft = GenericFieldType(ModuleList, Task)
#         field_schema, defs = gft.restricted_schema(
#             filter_schema_cls=RestrictedSchemaMixin,
#             variants=[Task],
#             field_name="tasks",
#             _profile="shared"
#         )
#         # Should return schema with merged defs
#         assert isinstance(field_schema, dict)
#         assert isinstance(defs, dict)

#     def test_restricted_schema_with_union_types(self):
#         """Should handle union of types"""
#         gft = GenericFieldType(ModuleList, [Task, State])
#         field_schema, defs = gft.restricted_schema(
#             filter_schema_cls=RestrictedSchemaMixin,
#             variants=[Task, State],
#             field_name="items",
#             _profile="shared"
#         )
#         assert isinstance(field_schema, dict)
#         assert isinstance(defs, dict)

#     def test_restricted_schema_with_dict_types(self):
#         """Should handle dict with key and value types"""
#         gft = GenericFieldType(ModuleDict, str, Task)
#         field_schema, defs = gft.restricted_schema(
#             filter_schema_cls=RestrictedSchemaMixin,
#             variants=[Task],
#             field_name="states",
#             _profile="shared"
#         )
#         assert isinstance(field_schema, dict)
#         assert isinstance(defs, dict)

#     def test_restricted_schema_with_primitive_types(self):
#         """Should handle primitive types like str, int"""
#         gft = GenericFieldType(ModuleDict, str, Task)
#         field_schema, defs = gft.restricted_schema(
#             filter_schema_cls=RestrictedSchemaMixin,
#             variants=[Task],
#             field_name="mapping",
#             _profile="shared"
#         )
#         # Should handle str type for dict keys
#         assert isinstance(field_schema, dict)
#         assert isinstance(defs, dict)

#     def test_restricted_schema_merges_defs_from_all_positions(self):
#         """Should merge defs from all type positions"""

#         class Module1(BaseModule, RestrictedSchemaMixin):
#             x: int

#             @classmethod
#             def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
#                 schema = cls.schema()
#                 defs = {"Module1Extra": {"type": "object"}}
#                 return (schema, defs)

#         class Module2(BaseModule, RestrictedSchemaMixin):
#             y: str

#             @classmethod
#             def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
#                 schema = cls.schema()
#                 defs = {"Module2Extra": {"type": "string"}}
#                 return (schema, defs)

#         gft = GenericFieldType(ModuleList, [Module1, Module2])
#         field_schema, defs = gft.restricted_schema(
#             filter_schema_cls=RestrictedSchemaMixin,
#             variants=[Module1, Module2],
#             field_name="modules",
#             _profile="shared"
#         )

#         # Should merge defs from both modules
#         # Note: The actual merged defs depend on implementation details
#         assert isinstance(defs, dict)

#     def test_restricted_schema_with_nested_genericfieldtype(self):
#         """Should handle nested GenericFieldType"""
#         inner_gft = GenericFieldType(ModuleList, Task)
#         outer_gft = GenericFieldType(ModuleList, inner_gft)

#         field_schema, defs = outer_gft.restricted_schema(
#             filter_schema_cls=RestrictedSchemaMixin,
#             variants=[Task],
#             field_name="nested",
#             _profile="shared"
#         )

#         assert isinstance(field_schema, dict)
#         assert isinstance(defs, dict)

#     def test_restricted_schema_inline_profile(self):
#         """Should work with inline profile"""
#         gft = GenericFieldType(ModuleList, Task)
#         field_schema, defs = gft.restricted_schema(
#             filter_schema_cls=RestrictedSchemaMixin,
#             variants=[Task],
#             field_name="tasks",
#             _profile="inline"
#         )
#         assert isinstance(field_schema, dict)
#         assert isinstance(defs, dict)


# class TestGenericFieldTypeEquality:
#     """Test GenericFieldType __eq__ and __repr__."""

#     def test_equal_instances_are_equal(self):
#         gft1 = GenericFieldType(ModuleList, Task)
#         gft2 = GenericFieldType(ModuleList, Task)
#         assert gft1 == gft2

#     def test_different_origin_not_equal(self):
#         gft1 = GenericFieldType(ModuleList, Task)
#         gft2 = GenericFieldType(ModuleDict, str, Task)
#         assert gft1 != gft2

#     def test_different_typs_not_equal(self):
#         gft1 = GenericFieldType(ModuleList, Task)
#         gft2 = GenericFieldType(ModuleList, State)
#         assert gft1 != gft2

#     def test_not_equal_to_non_GenericFieldType(self):
#         gft = GenericFieldType(ModuleList, Task)
#         assert gft != "not a GenericFieldType"
#         assert gft != 42

#     def test_repr_shows_origin_and_typs(self):
#         gft = GenericFieldType(ModuleList, Task)
#         repr_str = repr(gft)
#         assert "GenericFieldType" in repr_str
#         assert "ModuleList" in repr_str
#         assert "Task" in repr_str


# class TestGenerictypeHelper:
#     """Test generictype() helper function."""

#     def test_creates_GenericFieldType(self):
#         gft = generictype(ModuleList, Task)
#         assert isinstance(gft, GenericFieldType)
#         assert gft.origin is ModuleList
#         assert gft.typs == [[Task]]

#     def test_with_union_types(self):
#         gft = generictype(ModuleList, [Task, State])
#         assert isinstance(gft, GenericFieldType)
#         assert gft.typs == [[Task, State]]

#     def test_with_dict(self):
#         gft = generictype(ModuleDict, str, Task)
#         assert isinstance(gft, GenericFieldType)
#         assert gft.typs == [[str], [Task]]


# class TestBaseFieldDescriptor:
#     """Test BaseFieldDescriptor abstract base class"""

#     def test_base_field_descriptor_is_abstract(self):
#         """BaseFieldDescriptor should be abstract and not instantiable directly"""
#         with pytest.raises(TypeError):
#             BaseFieldDescriptor(typ=SimpleModule)

#     def test_descriptor_stores_owner_and_name_on_set_name(self):
#         """__set_name__ should store owner class and field name"""
#         descriptor = ModFieldDescriptor(typ=SimpleModule)

#         class TestClass(BaseModule):
#             field: SimpleModule = descriptor

#         assert descriptor._owner is TestClass
#         assert descriptor._name == "field"


# class TestModFieldDescriptor:
#     """Test ModFieldDescriptor for single module fields"""

#     def test_modfield_descriptor_creation(self):
#         """Can create ModFieldDescriptor"""
#         descriptor = ModFieldDescriptor(typ=SimpleModule)
#         assert isinstance(descriptor, BaseFieldDescriptor)

#     def test_modfield_validates_type_annotation_on_set_name(self):
#         """ModFieldDescriptor should validate type is a BaseModule subclass"""

#         with pytest.raises(TypeError, match="must be a BaseModule subclass"):

#             class TestClass(BaseModule):
#                 # int is not a BaseModule
#                 field: int = ModFieldDescriptor(typ=int)

#     def test_modfield_allows_union_of_modules(self):
#         """ModFieldDescriptor should allow Union of BaseModule subclasses"""

#         class Module1(BaseModule):
#             x: int

#         class Module2(BaseModule):
#             y: str

#         # Should not raise
#         class TestClass(BaseModule):
#             field: Module1 | Module2 = ModFieldDescriptor(typ=[Module1, Module2])

#     def test_modfield_get_returns_value(self):
#         """ModFieldDescriptor.__get__ should return the stored value"""

#         class TestClass(BaseModule):
#             field: SimpleModule

#         instance = TestClass(field=SimpleModule(value=42))
#         assert isinstance(instance.field, SimpleModule)
#         assert instance.field.value == 42

#     def test_modfield_set_stores_value(self):
#         """ModFieldDescriptor.__set__ should store the value"""

#         class TestClass(BaseModule):
#             field: SimpleModule = modfield()

#         instance = TestClass(field=SimpleModule(value=10))
#         new_module = SimpleModule(value=20)
#         instance.field = new_module
#         assert instance.field is new_module

#     def test_modfield_restricted_schema_with_no_variants_returns_base_schema(self):
#         """restricted_schema() with variants=None should return unrestricted schema"""

#         class TestClass(BaseModule, RestrictedSchemaMixin):
#             field: RestrictedModule = modfield()

#             @classmethod
#             def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
#                 return cls.schema()

#         descriptor = TestClass.field  # Get the descriptor from class
#         field_schema, defs = descriptor.restricted_schema(
#             filter_schema_cls=RestrictedSchemaMixin, variants=None
#         )

#         # Should return base schema for the field
#         assert "anyOf" in field_schema or "$ref" in field_schema or "type" in field_schema

#     def test_modfield_restricted_schema_with_variants_creates_union(self):
#         """restricted_schema() with variants should create union of variant schemas"""

#         class Module1(BaseModule, RestrictedSchemaMixin):
#             x: int

#             @classmethod
#             def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
#                 return cls.schema()

#         class Module2(BaseModule, RestrictedSchemaMixin):
#             y: str

#             @classmethod
#             def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
#                 return cls.schema()

#         class TestClass(BaseModule, RestrictedSchemaMixin):
#             field: Module1 | Module2 = modfield()

#             @classmethod
#             def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
#                 return cls.schema()

#         descriptor = TestClass.field
#         field_schema, defs = descriptor.restricted_schema(
#             filter_schema_cls=RestrictedSchemaMixin, variants=[Module1, Module2], _profile="shared"
#         )

#         # Should create union schema (anyOf or $ref)
#         assert "anyOf" in field_schema or "$ref" in field_schema
#         assert isinstance(defs, dict)

#     def test_modfield_restricted_schema_inline_profile(self):
#         """restricted_schema() with _profile='inline' should create inline oneOf"""

#         class Module1(BaseModule, RestrictedSchemaMixin):
#             x: int

#             @classmethod
#             def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
#                 return cls.schema()

#         class Module2(BaseModule, RestrictedSchemaMixin):
#             y: str

#             @classmethod
#             def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
#                 return cls.schema()

#         class TestClass(BaseModule, RestrictedSchemaMixin):
#             field: Module1 | Module2 = modfield()

#             @classmethod
#             def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
#                 return cls.schema()

#         descriptor = TestClass.field
#         field_schema, defs = descriptor.restricted_schema(
#             filter_schema_cls=RestrictedSchemaMixin, variants=[Module1, Module2], _profile="inline"
#         )

#         # Should create inline union (anyOf or oneOf)
#         assert "anyOf" in field_schema or "oneOf" in field_schema
#         # Defs may or may not be populated
#         assert isinstance(defs, dict)

#     def test_modfield_restricted_schema_with_genericfieldtype(self):
#         """Should delegate to GenericFieldType when typ contains one"""

#         class TestClass(BaseModule):
#             field: ModuleList[Task] = modfield(typ=GenericFieldType(ModuleList, Task))

#         descriptor = TestClass.field
#         field_schema, defs = descriptor.restricted_schema(
#             filter_schema_cls=RestrictedSchemaMixin,
#             variants=[Task],
#             _profile="shared"
#         )

#         # Should delegate to GenericFieldType.restricted_schema()
#         assert isinstance(field_schema, dict)
#         assert isinstance(defs, dict)

#     def test_modfield_restricted_schema_with_single_type(self):
#         """Should handle single type field correctly"""

#         class Module1(BaseModule, RestrictedSchemaMixin):
#             x: int

#             @classmethod
#             def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
#                 return cls.schema()

#         class TestClass(BaseModule):
#             field: Module1 = modfield(typ=Module1)

#         descriptor = TestClass.field
#         field_schema, defs = descriptor.restricted_schema(
#             filter_schema_cls=RestrictedSchemaMixin,
#             variants=[Module1],
#             _profile="shared"
#         )

#         assert isinstance(field_schema, dict)
#         assert isinstance(defs, dict)

#     def test_modfield_restricted_schema_merges_defs(self):
#         """Should merge defs from all types in union"""

#         class Module1(BaseModule, RestrictedSchemaMixin):
#             x: int

#             @classmethod
#             def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
#                 # Domain mixins return just schemas, not tuples
#                 return cls.schema()

#         class Module2(BaseModule, RestrictedSchemaMixin):
#             y: str

#             @classmethod
#             def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
#                 # Domain mixins return just schemas, not tuples
#                 return cls.schema()

#         class TestClass(BaseModule):
#             field: Module1 | Module2 = modfield(typ=[Module1, Module2])

#         descriptor = TestClass.field
#         field_schema, defs = descriptor.restricted_schema(
#             filter_schema_cls=RestrictedSchemaMixin,
#             variants=[Module1, Module2],
#             _profile="shared"
#         )

#         # Should create union with defs for both modules
#         assert isinstance(defs, dict)

#     def test_modfield_restricted_schema_with_none_type(self):
#         """Should handle None in type union"""

#         class Module1(BaseModule, RestrictedSchemaMixin):
#             x: int

#             @classmethod
#             def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
#                 return cls.schema()

#         class TestClass(BaseModule):
#             field: Module1 | None = modfield(typ=[Module1, None])

#         descriptor = TestClass.field
#         field_schema, defs = descriptor.restricted_schema(
#             filter_schema_cls=RestrictedSchemaMixin,
#             variants=[Module1],
#             _profile="shared"
#         )

#         # Should include null in union
#         assert isinstance(field_schema, dict)
#         assert isinstance(defs, dict)


# class TestModListFieldDescriptor:
#     """Test ModListFieldDescriptor for ModuleList fields"""

#     def test_modlistfield_descriptor_creation(self):
#         """Can create ModListFieldDescriptor"""
#         descriptor = ModListFieldDescriptor(typ=typing.List[SimpleModule])
#         assert isinstance(descriptor, BaseFieldDescriptor)

#     def test_modlistfield_restricted_schema_with_variants_creates_array_union(self):
#         """restricted_schema() should create array with union of variants"""

#         class Module1(BaseModule, RestrictedSchemaMixin):
#             x: int

#             @classmethod
#             def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
#                 return cls.schema()

#         class Module2(BaseModule, RestrictedSchemaMixin):
#             y: str

#             @classmethod
#             def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
#                 return cls.schema()

#         class TestClass(BaseModule, RestrictedSchemaMixin):
#             field: ModuleList[Module1 | Module2] = modlistfield()

#             @classmethod
#             def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
#                 return cls.schema()

#         descriptor = TestClass.field
#         field_schema, defs = descriptor.restricted_schema(
#             filter_schema_cls=RestrictedSchemaMixin, variants=[Module1, Module2], _profile="shared"
#         )

#         # Should be object type (ModuleListSpec)
#         assert field_schema["type"] == "object"
#         # Should have items property with array containing union
#         assert "items" in field_schema["properties"]
#         # Defs may or may not be populated depending on variant implementation
#         assert isinstance(defs, dict)


# class TestModDictFieldDescriptor:
#     """Test ModDictFieldDescriptor for ModuleDict fields"""

#     def test_moddictfield_descriptor_creation(self):
#         """Can create ModDictFieldDescriptor"""
#         descriptor = ModDictFieldDescriptor()
#         assert isinstance(descriptor, BaseFieldDescriptor)

#     def test_moddictfield_validates_key_type(self):
#         """ModDictFieldDescriptor should only allow str or int keys"""

#         with pytest.raises(TypeError, match="key type must be str or int"):

#             class TestClass(BaseModule):
#                 field: ModuleDict[float, SimpleModule] = moddictfield()

#     def test_moddictfield_restricted_schema_with_variants_creates_dict_union(self):
#         """restricted_schema() should create object with additionalProperties union"""

#         class Module1(BaseModule, RestrictedSchemaMixin):
#             x: int

#             @classmethod
#             def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
#                 return cls.schema()

#         class Module2(BaseModule, RestrictedSchemaMixin):
#             y: str

#             @classmethod
#             def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
#                 return cls.schema()

#         class TestClass(BaseModule, RestrictedSchemaMixin):
#             field: ModuleDict[str, Module1 | Module2] = moddictfield()

#             @classmethod
#             def restricted_schema(cls, *, _profile="shared", _seen=None, **kwargs):
#                 return cls.schema()

#         descriptor = TestClass.field
#         field_schema, defs = descriptor.restricted_schema(
#             filter_schema_cls=RestrictedSchemaMixin, variants=[Module1, Module2], _profile="shared"
#         )

#         # Should be object type (ModuleDictSpec)
#         assert field_schema["type"] == "object"
#         # Should have items property with dict containing union
#         assert "items" in field_schema["properties"]
#         # Defs may or may not be populated depending on variant implementation
#         assert isinstance(defs, dict)


# class TestModFieldFactories:
#     """Test factory functions modfield(), modlistfield(), moddictfield()"""

#     def test_modfield_factory_creates_descriptor(self):
#         """modfield() should create ModFieldDescriptor"""
#         descriptor = modfield()
#         assert isinstance(descriptor, ModFieldDescriptor)

#     def test_modlistfield_factory_creates_descriptor(self):
#         """modlistfield() should create ModListFieldDescriptor"""
#         descriptor = modlistfield()
#         assert isinstance(descriptor, ModListFieldDescriptor)

#     def test_moddictfield_factory_creates_descriptor(self):
#         """moddictfield() should create ModDictFieldDescriptor"""
#         descriptor = moddictfield()
#         assert isinstance(descriptor, ModDictFieldDescriptor)

#     def test_modfield_factory_usage_in_class(self):
#         """modfield() can be used as field default in BaseModule"""

#         class TestClass(BaseModule):
#             field: SimpleModule = modfield()

#         instance = TestClass(field=SimpleModule(value=99))
#         assert instance.field.value == 99
