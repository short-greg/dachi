from dachi._core import _instruct as core
from .test_core import Role
from dachi._core._core import Struct, str_formatter, Ref, Out
from .test_core import SimpleStruct
import pytest


class TestOp:

    def test_op_outputs_an_instruction(self):

        role = Role(name='Assistant', duty='You are a helpful assistant')

        text = 'Evaluate the user'
        instruction = core.op(
            [role], text
        )
        assert 'Assistant' in instruction.text

    def test_op_outputs_an_instruction_with_reef(self):

        role = Role(name='Assistant', duty='You are a helpful assistant')
        ref = Ref(desc=role)
        text = 'Evaluate the user'
        instruction = core.op(
            [ref], text
        )
        assert 'Assistant' in instruction.text


class TestBullet(object):

    def test_bullet_outputs_x(self):
        out = core.bullet(
            ['x', 'y']
        )
        assert '-x' in out.render()

    def test_bullet_outputs_y(self):
        out = core.bullet(
            ['x', 'y']
        )
        assert '-y' in out.render()


class TestNumbered(object):

    def test_numbered_outputs_x(self):
        out = core.numbered(
            ['x', 'y']
        )
        assert '2. y' in out.render()

    def test_numbered_outputs_x(self):
        out = core.numbered(
            ['x', 'y'], numbering='roman'
        )
        assert 'i. y' in out.render()


class TestFill(object):

    def test_fill_updates_text(self):

        out = core.fill(
            '{x}', x=2
        )
        assert out.text == '2'

    def test_fill_updates_first_item(self):

        out = core.fill(
            '{x} {x2}', x=2
        )
        assert out.text == '2 {x2}'

    def test_fill_updates_second_item(self):

        out = core.fill(
            '{x} {x2}', x2=2
        )
        assert out.text == '{x} 2'

    def test_fill_updates_output_of_struct(self):

        struct = SimpleStruct(
            x='{x}'
        )

        with pytest.raises(ValueError): 
            core.fill(
                struct, x2=2
            )

class TestHead(object):

    def test_head_adds_a_heading(self):

        out = core.head(
            'Title', 1
        )
        assert out.text == '# Title'

    def test_head_adds_a_heading_with_two(self):

        out = core.head(
            'Title', 2
        )
        assert out.text == '## Title'


class TestSection(object):

    def test_seciton_adds_a_section(self):

        out = core.section(
            'Title', 'A bunch of details'
        )
        assert '# Title' in out.text
        assert 'A bunch of details' in out.text

    def test_head_adds_a_heading_with_size_of_two(self):

        out = core.section(
            'Title', 'A bunch of details', 2, 2
        )
        assert '## Title' in out.text
        assert 'A bunch of details' in out.text


class TestCat(object):

    def test_cat_concatenates_text(self):

        out = core.cat(
            ['Title', 'The Earth Abides']
        )
        assert out.text == 'Title The Earth Abides'


    def test_cat_concatenates_text_with_a_colon(self):

        out = core.cat(
            ['Title', 'The Earth Abides'], ': '
        )
        assert out.text == 'Title: The Earth Abides'


class TestJoin(object):

    def test_join_joins_text(self):

        out = core.join(
            'Title', 'The Earth Abides'
        )
        print(out.text)
        assert out.text == 'Title The Earth Abides'

    def test_join_concatenates_text_with_a_colon(self):

        out = core.join(
            'Title', 'The Earth Abides', ': '
        )
        assert out.text == 'Title: The Earth Abides'


class TestOperation(object):

    def test_operation_alters_teh_text(self):

        operation = core.Operation(
            'Translate', 'Translate the input {}'
        )
        assert operation.instruction.text == 'Translate the input {}'

    def test_operation_renders_the_text(self):

        operation = core.Operation(
            'Translate', 'Translate the input {}'
        )
        assert operation('x').text == 'Translate the input x'

    def test_operation_renders_the_text_with_named_var(self):

        operation = core.Operation(
            'Translate', 'Translate the input {x}'
        )
        assert operation(x='x').text == 'Translate the input x'


class TestInstructF:

    def test_inserts_into_docstring(self):

        @core.instructf(True)
        def signaturep(x: str) -> Out[SimpleStruct]:
            """Output the value of x
            
            x: {x}

            Args:
                x (str): The input

            Returns:
                Out[SimpleStruct]: The value of x
            """
            pass

        result = signaturep(2)

        assert 'x: 2' in result.text


    def test_inserts_into_docstring_with_method(self):

        class X(object):
            @core.instructf(True)
            def signaturep(self, x: str) -> Out[SimpleStruct]:
                """Output the value of x
                
                x: {x}

                Args:
                    x (str): The input

                Returns:
                    Out[SimpleStruct]: The value of x
                """
                pass

        x = X()
        result = x.signaturep(2)

        assert 'x: 2' in result.text
