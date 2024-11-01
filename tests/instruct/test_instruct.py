from dachi.op import _instruct as core
from dachi.op._data import Ref
from .._structs import Role, SimpleStruct
import pytest


class TestOp:

    def test_op_outputs_an_instruction(self):

        role = Role(name='Assistant', duty='You are a helpful assistant')

        text = 'Evaluate the user'
        cue = core.op(
            [role], text
        )
        assert 'Assistant' in cue.text

    def test_op_outputs_an_instruction_with_reef(self):

        role = Role(name='Assistant', duty='You are a helpful assistant')
        ref = Ref(desc=role)
        text = 'Evaluate the user'
        cue = core.op(
            [ref], text
        )
        assert 'Assistant' in cue.text


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

    def test_operation_alters_the_text(self):

        operation = core.Op(
            'Translate', 'Translate the input {}'
        )
        assert operation.cue.render() == 'Translate the input {}'

    def test_operation_renders_the_text(self):

        operation = core.Op(
            'Translate', 'Translate the input {}'
        )
        assert operation('x').text == 'Translate the input x'

    def test_operation_renders_the_text_with_named_var(self):

        operation = core.Op(
            'Translate', 'Translate the input {x}'
        )
        assert operation(x='x').text == 'Translate the input x'
