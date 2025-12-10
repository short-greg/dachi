from dachi.utils._internal import _cls_tools as utils
from typing import List, Union, Optional, ForwardRef


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
        result = utils.python_type_to_json_schema(List[int])
        assert result == {'type': 'array', 'items': {'type': 'integer'}}

    def test_converts_list_of_str(self):
        result = utils.python_type_to_json_schema(List[str])
        assert result == {'type': 'array', 'items': {'type': 'string'}}

    def test_converts_dict_with_value_type(self):
        from typing import Dict
        result = utils.python_type_to_json_schema(Dict[str, int])
        assert result == {'type': 'object', 'additionalProperties': {'type': 'integer'}}

    def test_converts_union_types(self):
        result = utils.python_type_to_json_schema(Union[int, str])
        assert result == {'oneOf': [{'type': 'integer'}, {'type': 'string'}]}

    def test_converts_optional_int(self):
        result = utils.python_type_to_json_schema(Optional[int])
        assert result == {'oneOf': [{'type': 'integer'}, {'type': 'null'}]}

    def test_converts_nested_list(self):
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
        assert utils.is_generic_type(ForwardRef("int")) is False

    def test_ForwardRef_List_int_returns_true(self):
        assert utils.is_generic_type(ForwardRef("List[int]")) is True

    def test_ForwardRef_custom_generic_returns_true(self):
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
