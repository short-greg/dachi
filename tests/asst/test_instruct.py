from dachi.asst import _instruct_core as core
from dachi.asst import Cue
from dachi.asst import _instruct
from dachi.asst._instruct import style_formatter
from ..utils.test_core import SimpleStruct
from .test_ai import DummyAIModel
import pytest

def dummy_dec(f):
    """Use to ensure signaturemethod works 
    even if decorated
    """
    
    def _(*args, **kwargs):
        return f(*args, **kwargs)
    return _


class TestStyleFormat:

    def test_basic_string_formatting(self):
        """Tests standard Python formatting without special styling."""
        result = style_formatter("Hello, {name}!", name="Alice")
        assert result == "Hello, Alice!"

    def test_bullet_list_formatting(self):
        """Tests whether lists are correctly formatted as bullet points."""
        result = style_formatter("Shopping List:\n{x:bullet}", x=["Milk", "Eggs", "Bread"])
        expected = "Shopping List:\n- Milk\n- Eggs\n- Bread"
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
        result = style_formatter("Values:\n{0:bullet(1)}", ["A", "B", "C"])
        expected = "Values:\n- A\n- B\n- C"
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


class TestSignatureF:

    def test_inserts_into_docstring(self):

        @core.signaturefunc()
        def signaturep(x: str) -> SimpleStruct:
            """Output the value of x
            
            x: {x}

            Args:
                x (str): The input

            Returns:
                SimpleStruct: The value of x
            """
            pass

        result = signaturep.i(2)
        assert 'x: 2' in result.text

    def test_inserts_into_docstring_with_method(self):

        class X(object):
            @core.signaturemethod()
            def signaturep(self, x: str) -> SimpleStruct:
                """Output the value of x
                
                x: {x}

                Args:
                    x (str): The input

                Returns:
                    SimpleStruct: The value of x
                """
                pass

        x = X()
        result = x.signaturep.i(2)

        assert 'x: 2' in result.text

    def test_signature_executes_model(self):

        class X(object):
            
            @core.signaturemethod(engine=DummyAIModel())
            def signaturep(self, x: str) -> str:
                """Output the value of x
                
                x: {x}

                Args:
                    x (str): The input

                Returns:
                    SimpleStruct: The value of x
                """
                pass

        x = X()
        result = x.signaturep(2)

        assert result == 'Great!'

    def test_inserts_into_docstring_with_method_when_decorated(self):

        class X(object):
            
            @dummy_dec
            @core.signaturemethod(engine=DummyAIModel())
            def signaturep(self, x: str) -> str:
                """Output the value of x
                
                x: {x}

                Args:
                    x (str): The input

                Returns:
                    str: The value of x
                """
                pass

        x = X()
        result = x.signaturep(2)

        assert result == 'Great!'

    def test_signature_streams_the_output(self):

        class X(object):

            @core.signaturemethod(engine=DummyAIModel(), to_stream=True)
            def signaturep(self, x: str) -> str:
                """Output the value of x
                
                x: {x}

                Args:
                    x (str): The input

                Returns:
                    str: The value of x
                """
                pass

        x = X()
        for d in x.signaturep(2):
            pass
        assert d == '!'

#     def test_signature_uses_the_correct_model(self):

#         class X(object):

#             def __init__(self, model):
#                 super().__init__()
#                 self.model = model
            
#             @core.signaturemethod(engine='model')
#             def signaturep(self, x: str) -> str:
#                 """Output the value of x
                
#                 x: {x}

#                 Args:
#                     x (str): The input

#                 Returns:
#                     SimpleStruct: The value of x
#                 """
#                 pass

#         x = X(DummyAIModel('Awesome'))
#         x2 = X(DummyAIModel('Fabulous'))
#         result = x.signaturep(2)
#         result2 = x2.signaturep(2)

#         assert result == 'Awesome'
#         assert result2 == 'Fabulous'



# class TestCue(object):

#     def test_instruction_renders_with_text(self):

#         cue = Cue(
#             text='x'
#         )
#         assert cue.render() == 'x'

#     def test_instruction_text_is_correct(self):

#         text = 'Evaluate the quality of the CSV'
#         cue = Cue(
#             name='Evaluate',
#             text=text
#         )
#         assert cue.text == text

#     def test_render_returns_the_instruction_text(self):

#         text = 'Evaluate the quality of the CSV'
#         cue = Cue(
#             name='Evaluate',
#             text=text
#         )
#         assert cue.render() == text

#     def test_i_returns_the_instruction(self):

#         text = 'Evaluate the quality of the CSV'
#         cue = Cue(
#             name='Evaluate',
#             text=text
#         )
#         assert cue.i() is cue



# class TestInstructF:

#     def test_instruct(self):

#         @core.instructfunc(engine=DummyAIModel())
#         def instructrep(x: str) -> str:
#             """Output the value of x
            
#             x: {x}

#             Args:
#                 x (str): The input

#             Returns:
#                 SimpleStruct: The value of x
#             """
#             return core.Cue(f'Do {x}')

#         result = instructrep.i(2)

#         assert 'Do 2' == result.text

#     def test_inserts_into_instruction_with_method(self):

#         class X(object):

#             @core.instructmethod(engine=DummyAIModel())
#             def instructrep(self, x: str) -> str:
#                 """
#                 """
#                 return core.Cue(f'Do {x}')

#         x = X()
#         result = x.instructrep.i(2)

#         assert 'Do 2' == result.text

#     def test_x_has_different_instance_for_instruct_rep(self):

#         class X(object):

#             @core.instructmethod(engine=DummyAIModel())
#             def instructrep(self, x: str) -> str:
#                 """
#                 """
#                 return core.Cue(f'Do {x}')

#         x = X()
#         x2 = X()
#         assert x.instructrep is not x2.instructrep

#     def test_x_has_same_value_for_instruct_rep(self):

#         class X(object):

#             @core.instructmethod(engine=DummyAIModel())
#             def instructrep(self, x: str) -> str:
#                 """
#                 """
#                 return core.Cue(f'Do {x}')

#         x = X()
#         assert x.instructrep is x.instructrep

#     def test_signature_uses_the_correct_model(self):

#         class X(object):

#             def __init__(self, model):
#                 super().__init__()
#                 self.model = model
            
#             @dummy_dec
#             @core.instructmethod(engine=DummyAIModel('Awesome'))
#             def instructrep(self, x: str) -> str:
#                 """
#                 """
#                 return core.Cue(f'Do {x}')

#         x = X(DummyAIModel('Awesome'))
#         result = x.instructrep(2)

#         assert result == 'Awesome'
