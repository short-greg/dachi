from dachi import utils
from dachi.core import model_to_text, struct_template
from pydantic import BaseModel
import pytest


class SimpleStruct(BaseModel):

    x: str


class NestedStruct(BaseModel):

    simple: SimpleStruct


class TestIsUndefined(object):

    def test_is_undefined(self):

        assert utils.is_undefined(
            utils.UNDEFINED
        )

    def test_not_is_undefined(self):

        assert not utils.is_undefined(
            1
        )


class TestIsNestedModel:

    def test_is_nested_model_returns_true_for_nested(self):

        assert utils.is_nested_model(NestedStruct) is True

    def test_is_nested_model_returns_false_for_not_nested(self):

        assert utils.is_nested_model(SimpleStruct) is False


class TestStruct(object):

    def test_simple_struct_gets_string(self):

        struct = SimpleStruct(x="2")
        assert struct.x == '2'
    
    def test_template_gives_correct_template(self):

        struct = SimpleStruct(x="2")
        template = struct_template(struct)
        print(template)
        assert template['x'].is_required is True
        assert template['x'].type_ == type('text')

    def test_template_gives_correct_template_with_nested(self):

        struct = NestedStruct(simple=SimpleStruct(x="2"))
        template = struct_template(struct)
        assert template['simple']['x'].is_required is True
        assert template['simple']['x'].type_ == type('text')

    def test_to_text_converts_to_text(self):
        struct = SimpleStruct(x="2")
        text = model_to_text(struct)
        assert "2" in text

    def test_to_text_doubles_the_braces(self):
        struct = SimpleStruct(x="2")
        text = model_to_text(struct, True)
        assert "{{" in text
        assert "}}" in text

    def test_to_text_works_for_nested(self):
        struct = NestedStruct(simple=SimpleStruct(x="2"))
        text = model_to_text(struct, True)
        assert text.count('{{') == 2
        assert text.count("}}") == 2

    def test_to_dict_converts_to_a_dict(self):
        struct = SimpleStruct(x="2")
        d = struct.model_dump()
        assert d['x'] == "2"


class TestStrFormatter(object):

    def test_formatter_formats_positional_variables(self):

        assert utils.str_formatter(
            '{} {}', 1, 2
        ) == '1 2'

    def test_formatter_formats_positional_variables(self):

        assert utils.str_formatter(
            '{0} {1}', 1, 2
        ) == '1 2'

    def test_formatter_formats_named_variables(self):

        assert utils.str_formatter(
            '{x} {y}', x=1, y=2
        ) == '1 2'

    def test_formatter_raises_error_if_positional_and_named_variables(self):

        with pytest.raises(ValueError):
            utils.str_formatter(
                '{0} {y}', 1, y=2
            )

    def test_get_variables_gets_all_pos_variables(self):

        assert utils.get_str_variables(
            '{0} {1}'
        ) == [0, 1]

    def test_get_variables_gets_all_named_variables(self):

        assert utils.get_str_variables(
            '{x} {y}'
        ) == ['x', 'y']


class TestGetMember(object):

    def test_get_member_gets_immediate_child(self):

        class X:
            y = 2

        x = X()

        assert utils.get_member(
            x, 'y'
        ) == 2

    def test_get_member_gets_sub_child(self):

        class X:
            y = 2

            def __getattr__(self, key):

                o = X()
                object.__setattr__(self, key, o)
                return o

        x = X()

        assert utils.get_member(
            x, 'z.y'
        ) == 2


class TestGetOrSpawn(object):

    def test_get_or_spawn_doesnt_spawn_new_state(self):

        state = {'child': {}}
        target = state['child']
        child = utils.get_or_spawn(state, 'child')
        assert child is target

    def test_get_or_spawn_spawns_new_state(self):

        state = {'child': {}}
        target = state['child']
        child = utils.get_or_spawn(state, 'other')
        assert not child is target


class TestGetOrSet(object):

    def test_get_or_set_doesnt_set_new_value(self):

        state = {'val': 2}
        target = state['val']
        child = utils.get_or_set(state, 'val', 3)
        assert child is target

    def test_get_or_spawn_sets_a_new_value(self):

        state = {}
        child = utils.get_or_set(state, 'val', 3)
        assert child == 3


class TestPythonTypeToJsonSchema:

    def test_converts_int(self):
        result = utils.python_type_to_json_schema(int)
        assert result == {'type': 'integer'}

    def test_converts_str(self):
        result = utils.python_type_to_json_schema(str)
        assert result == {'type': 'string'}

    def test_converts_float(self):
        result = utils.python_type_to_json_schema(float)
        assert result == {'type': 'number'}

    def test_converts_bool(self):
        result = utils.python_type_to_json_schema(bool)
        assert result == {'type': 'boolean'}

    def test_converts_list(self):
        result = utils.python_type_to_json_schema(list)
        assert result == {'type': 'array'}

    def test_converts_dict(self):
        result = utils.python_type_to_json_schema(dict)
        assert result == {'type': 'object'}

    def test_converts_none_type(self):
        result = utils.python_type_to_json_schema(type(None))
        assert result == {'type': 'null'}

    def test_converts_list_of_int(self):
        from typing import List
        result = utils.python_type_to_json_schema(List[int])
        assert result == {'type': 'array', 'items': {'type': 'integer'}}

    def test_converts_list_of_str(self):
        from typing import List
        result = utils.python_type_to_json_schema(List[str])
        assert result == {'type': 'array', 'items': {'type': 'string'}}

    def test_converts_dict_with_value_type(self):
        from typing import Dict
        result = utils.python_type_to_json_schema(Dict[str, int])
        assert result == {'type': 'object', 'additionalProperties': {'type': 'integer'}}

    def test_converts_union_types(self):
        from typing import Union
        result = utils.python_type_to_json_schema(Union[int, str])
        assert result == {'oneOf': [{'type': 'integer'}, {'type': 'string'}]}

    def test_converts_optional_int(self):
        from typing import Optional
        result = utils.python_type_to_json_schema(Optional[int])
        assert result == {'oneOf': [{'type': 'integer'}, {'type': 'null'}]}

    def test_converts_nested_list(self):
        from typing import List
        result = utils.python_type_to_json_schema(List[List[str]])
        expected = {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'string'}}}
        assert result == expected

    def test_unknown_type_defaults_to_string(self):
        class CustomType:
            pass
        result = utils.python_type_to_json_schema(CustomType)
        assert result == {'type': 'string'}


class TestIsGenericType:
    """Test is_generic_type() utility function."""

    def test_plain_int_returns_false(self):
        assert utils.is_generic_type(int) is False

    def test_plain_str_returns_false(self):
        assert utils.is_generic_type(str) is False

    def test_plain_bool_returns_false(self):
        assert utils.is_generic_type(bool) is False

    def test_plain_float_returns_false(self):
        assert utils.is_generic_type(float) is False

    def test_custom_class_returns_false(self):
        class CustomClass:
            pass
        assert utils.is_generic_type(CustomClass) is False

    def test_list_of_int_returns_true(self):
        assert utils.is_generic_type(list[int]) is True

    def test_dict_of_str_int_returns_true(self):
        assert utils.is_generic_type(dict[str, int]) is True

    def test_tuple_of_int_str_returns_true(self):
        assert utils.is_generic_type(tuple[int, str]) is True

    def test_List_of_int_returns_true(self):
        from typing import List
        assert utils.is_generic_type(List[int]) is True

    def test_Dict_of_str_int_returns_true(self):
        from typing import Dict
        assert utils.is_generic_type(Dict[str, int]) is True

    def test_Tuple_of_int_str_returns_true(self):
        from typing import Tuple
        assert utils.is_generic_type(Tuple[int, str]) is True

    def test_Union_of_int_str_returns_true(self):
        from typing import Union
        assert utils.is_generic_type(Union[int, str]) is True

    def test_pep604_union_returns_true(self):
        assert utils.is_generic_type(int | str) is True

    def test_Optional_returns_true(self):
        from typing import Optional
        assert utils.is_generic_type(Optional[int]) is True

    def test_string_plain_type_returns_false(self):
        assert utils.is_generic_type("int") is False

    def test_string_List_int_returns_true(self):
        assert utils.is_generic_type("List[int]") is True

    def test_string_custom_generic_returns_true(self):
        assert utils.is_generic_type("ModuleList[Task]") is True

    def test_string_Union_returns_true(self):
        assert utils.is_generic_type("Union[int, str]") is True

    def test_ForwardRef_plain_type_returns_false(self):
        from typing import ForwardRef
        assert utils.is_generic_type(ForwardRef("int")) is False

    def test_ForwardRef_List_int_returns_true(self):
        from typing import ForwardRef
        assert utils.is_generic_type(ForwardRef("List[int]")) is True

    def test_ForwardRef_custom_generic_returns_true(self):
        from typing import ForwardRef
        assert utils.is_generic_type(ForwardRef("Dict[str, Any]")) is True

    def test_list_class_returns_false(self):
        assert utils.is_generic_type(list) is False

    def test_dict_class_returns_false(self):
        assert utils.is_generic_type(dict) is False

    def test_tuple_class_returns_false(self):
        assert utils.is_generic_type(tuple) is False

    def test_None_returns_false(self):
        assert utils.is_generic_type(None) is False

    def test_type_None_returns_false(self):
        assert utils.is_generic_type(type(None)) is False

    def test_custom_generic_subscripted_returns_true(self):
        from typing import Generic, TypeVar
        T = TypeVar('T')

        class CustomGeneric(Generic[T]):
            pass

        assert utils.is_generic_type(CustomGeneric[int]) is True

    def test_custom_generic_unsubscripted_returns_false(self):
        from typing import Generic, TypeVar
        T = TypeVar('T')

        class CustomGeneric(Generic[T]):
            pass

        assert utils.is_generic_type(CustomGeneric) is False

    def test_nested_generic_returns_true(self):
        from typing import List, Dict
        assert utils.is_generic_type(List[Dict[str, int]]) is True
