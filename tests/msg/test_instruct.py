
from dachi.msg._instruct import style_formatter, bullet
import pytest
from dachi.msg._instruct import generate_numbered_list, numbered


class TestGenerateNumberedList:

    def test_arabic_numbering(self):
        """Tests arabic numbering."""
        result = generate_numbered_list(5, numbering_type='arabic')
        expected = ['1', '2', '3', '4', '5']
        assert result == expected

    def test_roman_numbering(self):
        """Tests roman numeral numbering."""
        result = generate_numbered_list(5, numbering_type='roman')
        expected = ['i', 'ii', 'iii', 'iv', 'v']
        assert result == expected

    def test_alphabet_numbering(self):
        """Tests alphabetic numbering."""
        result = generate_numbered_list(5, numbering_type='alphabet')
        expected = ['A', 'B', 'C', 'D', 'E']
        assert result == expected

    def test_invalid_numbering_type(self):
        """Tests invalid numbering type."""
        with pytest.raises(ValueError):
            generate_numbered_list(5, numbering_type='invalid')

    def test_alphabet_numbering_exceeds_limit(self):
        """Tests alphabetic numbering exceeding the limit."""
        with pytest.raises(ValueError, match="Alphabetic numbering can only handle up to 26 items"):
            generate_numbered_list(27, numbering_type='alphabet')

    def test_zero_items(self):
        """Tests generating a list with zero items."""
        result = generate_numbered_list(0, numbering_type='arabic')
        assert result == []

    def test_negative_items(self):
        """Tests generating a list with negative items."""
        with pytest.raises(ValueError):
            generate_numbered_list(-5, numbering_type='arabic')

    def test_default_numbering_type(self):
        """Tests default numbering type (arabic)."""
        result = generate_numbered_list(3)
        expected = ['1', '2', '3']
        assert result == expected


class TestNumbered:

    def test_arabic_numbering(self):
        """Tests arabic numbering with default indent."""
        result = numbered(["Apple", "Banana", "Cherry"], numbering="arabic")
        expected = "1. Apple\n2. Banana\n3. Cherry"
        assert result == expected

    def test_roman_numbering(self):
        """Tests roman numeral numbering."""
        result = numbered(["Apple", "Banana", "Cherry"], numbering="roman")
        expected = "i. Apple\nii. Banana\niii. Cherry"
        assert result == expected

    def test_alphabet_numbering(self):
        """Tests alphabetic numbering."""
        result = numbered(["Apple", "Banana", "Cherry"], numbering="alphabet")
        expected = "A. Apple\nB. Banana\nC. Cherry"
        assert result == expected

    def test_with_indent(self):
        """Tests numbering with indentation."""
        result = numbered(["Apple", "Banana", "Cherry"], indent=4, numbering="arabic")
        expected = "    1. Apple\n    2. Banana\n    3. Cherry"
        assert result == expected

    def test_empty_list(self):
        """Tests numbering with an empty list."""
        result = numbered([], numbering="arabic")
        assert result == ""

    def test_invalid_numbering_type(self):
        """Tests invalid numbering type."""
        with pytest.raises(ValueError):
            numbered(["Apple", "Banana"], numbering="invalid")

    def test_single_item(self):
        """Tests numbering with a single item."""
        result = numbered(["Apple"], numbering="arabic")
        expected = "1. Apple"
        assert result == expected

    def test_negative_indent(self):
        """Tests numbering with a negative indent."""
        result = numbered(["Apple", "Banana"], indent=-2, numbering="arabic")
        expected = "1. Apple\n2. Banana"
        assert result == expected  # Negative indent should be treated as no indent

    def test_large_list(self):
        """Tests numbering with a large list."""
        with pytest.raises(ValueError):
            items = [f"Item {i}" for i in range(1, 28)]
            numbered(items, numbering="alphabet")
            # expected = "\n".join([f"{chr(64 + i)}. Item {i}" for i in range(1, 27)])
            # assert result == expected

class TestBullet:

    def test_basic_bullet_list(self):
        """Tests basic bullet list generation."""
        result = bullet(["Apple", "Banana", "Cherry"])
        expected = "- Apple\n- Banana\n- Cherry"
        assert result == expected

    def test_custom_bullet_character(self):
        """Tests bullet list with a custom bullet character."""
        result = bullet(["Apple", "Banana", "Cherry"], bullets="*")
        expected = "* Apple\n* Banana\n* Cherry"
        assert result == expected

    def test_with_indent(self):
        """Tests bullet list with indentation."""
        result = bullet(["Apple", "Banana", "Cherry"], indent=4)
        expected = "    - Apple\n    - Banana\n    - Cherry"
        assert result == expected

    def test_empty_list(self):
        """Tests bullet list with an empty list."""
        result = bullet([])
        assert result == ""

    def test_single_item(self):
        """Tests bullet list with a single item."""
        result = bullet(["Apple"])
        expected = "- Apple"
        assert result == expected

    def test_negative_indent(self):
        """Tests bullet list with a negative indent."""
        result = bullet(["Apple", "Banana"], indent=-2)
        expected = "- Apple\n- Banana"  # Negative indent should be treated as no indent
        assert result == expected

    def test_numeric_items(self):
        """Tests bullet list with numeric items."""
        result = bullet([1, 2, 3])
        expected = "- 1\n- 2\n- 3"
        assert result == expected

    def test_mixed_type_items(self):
        """Tests bullet list with mixed type items."""
        result = bullet(["Apple", 42, 3.14])
        expected = "- Apple\n- 42\n- 3.14"
        assert result == expected

    def test_custom_bullet_and_indent(self):
        """Tests bullet list with custom bullet and indentation."""
        result = bullet(["Apple", "Banana"], bullets=">", indent=2)
        expected = "  > Apple\n  > Banana"
        assert result == expected


class TestStyleFormat:

    def test_basic_string_formatting(self):
        """Tests standard Python formatting without special styling."""
        result = style_formatter("Hello, {name}!", name="Alice")
        assert result == "Hello, Alice!"

    def test_bullet_list_formatting(self):
        """Tests whether lists are correctly formatted as bullet points."""
        result = style_formatter("Shopping List:\n\n{x:bullet}", x=["Milk", "Eggs", "Bread"])
        expected = "Shopping List:\n\n- Milk\n- Eggs\n- Bread"
        print(result, expected)
        assert result == expected

    def test_bold_formatting(self):
        """Tests bold formatting."""
        result = style_formatter("Total: {total:bold}", total=100)
        expected = "Total: **100**"
        assert result == expected

    def test_italic_formatting(self):
        """Tests italic formatting."""
        result = style_formatter("Style: {text_:italic}", text_="Fancy")
        expected = "Style: *Fancy*"
        assert result == expected

    def test_mixed_formatting(self):
        """Tests multiple styles in one string."""
        result = style_formatter(
            "Receipt:\nItems:\n{x:bullet}\nTotal: {total:bold}",
            x=["Apple", "Banana"],
            total=12.50
        )
        expected = "Receipt:\nItems:\n- Apple\n- Banana\nTotal: **12.5**"
        assert result == expected

    def test_no_style_fallback(self):
        """Tests that standard formatting still works if no styles are applied."""
        result = style_formatter("Hello, {name}", name="World")
        assert result == "Hello, World"

    def test_empty_list_bullet(self):
        """Tests bullet formatting with an empty list."""
        result = style_formatter("Items:\n{x:bullet}", x=[])
        expected = "Items:\n"
        assert result == expected

    def test_numeric_formatting(self):
        """Tests if numbers are formatted properly."""
        result = style_formatter("The value is: {value}", value=42)
        assert result == "The value is: 42"

    def test_style_with_empty_string(self):
        """Tests that styling works even when the string is empty."""
        result = style_formatter("{x:bold}", x="")
        expected = "****"  # Bold empty string should still be two asterisks
        assert result == expected

    def test_positional_arguments(self):
        """Tests positional arguments instead of named arguments."""
        result = style_formatter("Values:\n{0:bullet}", ["A", "B", "C"])
        expected = "Values:\n- A\n- B\n- C"
        assert result == expected

    def test_function_call_style(self):
        """Tests positional arguments instead of named arguments."""
        result = style_formatter("Values:\n{0:bullet('-', 1)}", ["A", "B", "C"])
        expected = "Values:\n - A\n - B\n - C"
        assert result == expected

# class TestStyleFormat:

#     def test_extract_style_var(self):

#         result = _instruct.extract_styles(
#             """
#             {<x: bullet>}
#             """
#         )
#         print(result)
#         assert result[0][0] == 'x'
#         assert result[0][1] == 'bullet'
#         assert result[0][2] is None
#         assert result[0][3] is True

#     def test_extract_style_var_with_args(self):

#         result = _instruct.extract_styles(
#             """
#             {<x: bullet(1)>}
#             """
#         )
#         print(result)
#         assert result[0][0] == 'x'
#         assert result[0][1] == 'bullet'
#         assert result[0][2] == ['1']
#         assert result[0][3] is True

#     def test_extract_style_var_with_default(self):

#         result = _instruct.extract_styles(
#             """
#             {<y::>}
#             """
#         )
#         assert result[0][0] == 'y'
#         assert result[0][1] == 'DEFAULT'
#         assert result[0][2] is None
#         assert result[0][3] is True

#     def test_extract_style_with_bullet(self):

#         result = _instruct.extract_styles(
#             """
#             {<bullet>}
#             """
#         )
#         assert result[0][0] == 0
#         assert result[0][1] == 'bullet'
#         assert result[0][2] is None
#         assert result[0][3] is False

#     def test_extract_style_with_bullet_with_args(self):

#         result = _instruct.extract_styles(
#             """
#             {<bullet(1)>}
#             """
#         )
#         assert result[0][0] == 0
#         assert result[0][1] == 'bullet'
#         assert result[0][2] == ['1']
#         assert result[0][3] is False

#     def test_extract_style_with_bullet_with_two_args(self):

#         result = _instruct.extract_styles(
#             """
#             {<bullet(1, 2)>}
#             """
#         )
#         assert result[0][0] == 0
#         assert result[0][1] == 'bullet'
#         assert result[0][2] == ['1', '2']
#         assert result[0][3] is False

#     def test_that_the_pos_is_correct(self):

#         result = _instruct.extract_styles(
#             """
#             {} {<x>}
#             """
#         )
#         assert result[0][0] == 1
#         assert result[0][1] == 'x'
#         assert result[0][2] == None
#         assert result[0][3] is False

#     def test_that_the_pos_is_correct_with_pos(self):

#         result = _instruct.extract_styles(
#             """
#             {} {<2::>}
#             """
#         )
#         assert result[0][0] == 2
#         assert result[0][1] == 'DEFAULT'
#         assert result[0][2] == None
#         assert result[0][3] is True

#     def test_replace_style_formatting_with_var(self):

#         result = _instruct.replace_style_formatting(
#             """{} {<x::>}"""
#         )
#         assert result == """{} {x}"""

#     def test_replace_style_formatting_with_only_style(self):

#         result = _instruct.replace_style_formatting(
#             """{} {<x>}"""
#         )
#         assert result == """{} {}"""


#     def test_replace_style_formatting_with_only_pos(self):

#         result = _instruct.replace_style_formatting(
#             """{} {<1::>}"""
#         )
#         assert result == """{} {1}"""

#     def test_replace_style_formatting_with_only_style_and_pos(self):

#         result = _instruct.replace_style_formatting(
#             """{} {<2:bullet(2)>}"""
#         )
#         assert result == """{} {2}"""

#     def test_replace_style_formatting_with_only_style_and_pos_and_no_args(self):

#         result = _instruct.replace_style_formatting(
#             """{} {<2:bullet>}"""
#         )
#         assert result == """{} {2}"""


#     def test_process_style_args_with_int(self):

#         result = _instruct.process_style_args(
#             ['1']
#         )
#         assert result == [1]

#     def test_process_style_args_with_float(self):

#         result = _instruct.process_style_args(
#             ['1.']
#         )
#         assert result == [1.0]

#     def test_process_style_args_with_str(self):

#         result = _instruct.process_style_args(
#             ['"1."']
#         )
#         assert result == ["1."]


#     def test_process_style_args_with_invalid_arg(self):

#         with pytest.raises(ValueError):
#             _instruct.process_style_args(
#                 ['x']
#             )
        
#     # TODO: Next I need to add proper styling
#     # functions and define how they work
#     def test_style_format_formats_a_list(self):

#         data = [1,2,3]
#         res = _instruct.style_format(
#             '{<data: bullet>}', data=data
#         )
#         print(res)
#         assert False

    # def test_extract_retrieves_style(self):

    #     result = _instruct.extract_styles(
    #         """
    #         {<bullet>}
    #         """
    #     )
    #     print(result)
    #     assert result[0][0] == 0
    #     assert result[0][1] == 'bullet'
    #     assert result[0][-1] is None

    # def test_extract_retrieves_regular_var(self):

    #     result = _instruct.extract_styles(
    #         """
    #         {x:2%}
    #         """
    #     )
    #     print(result)
    #     assert result[0][0] == "x:2%"
    #     assert result[0][1] is None
    #     assert result[0][-1] is None

    # def test_extract_retrieves_style_with_default(self):

    #     result = _instruct.extract_styles(
    #         """
    #         {<y::>}
    #         """
    #     )
    #     print(result)
    #     assert result[0][0] == 'y'
    #     assert result[0][1] == 'DEFAULT'
    #     assert result[0][-1] is None
    
    # def test_extract_style_var_with_args(self):

    #     result = _instruct.extract_styles(
    #         """
    #         {<x: bullet(1)>}
    #         """
    #     )
    #     print(result)
    #     assert result[0][0] == 'x'
    #     assert result[0][1] == 'bullet'
    #     assert result[0][-1] == ['1']
    
    # def test_extract_style_var_with_two_args(self):

    #     result = _instruct.extract_styles(
    #         """
    #         {<x: bullet(1, 2)>}
    #         """
    #     )
    #     print(result)
    #     assert result[0][0] == 'x'
    #     assert result[0][1] == 'bullet'
    #     assert result[0][-1] == ['1', '2']