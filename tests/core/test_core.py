from dachi._core import _core
# from dachi._core import _instruct as core
from dachi._core import Struct, str_formatter
import pytest
from pydantic import Field


class SimpleStruct(_core.Struct):

    x: str


class NestedStruct(_core.Struct):

    simple: SimpleStruct


class TestStruct(object):

    def test_simple_struct_gets_string(self):

        struct = SimpleStruct(x="2")
        assert struct.x == '2'
    
    def test_template_gives_correct_template(self):

        struct = SimpleStruct(x="2")
        template = struct.template()
        print(template)
        assert template['x']['is_required'] is True
        assert template['x']['type'] == type('text')

    def test_template_gives_correct_template_with_nested(self):

        struct = NestedStruct(simple=SimpleStruct(x="2"))
        template = struct.template()
        assert template['simple']['x']['is_required'] is True
        assert template['simple']['x']['type'] == type('text')

    def test_to_text_converts_to_text(self):
        struct = SimpleStruct(x="2")
        text = struct.to_text()
        assert "2" in text

    def test_to_text_doubles_the_braces(self):
        struct = SimpleStruct(x="2")
        text = struct.to_text(True)
        print(text)
        assert "{{" in text
        assert "}}" in text

    def test_to_text_works_for_nested(self):
        struct = NestedStruct(simple=SimpleStruct(x="2"))
        text = struct.to_text(True)
        assert text.count('{{') == 2
        assert text.count("}}") == 2

    def test_render_works_for_nested(self):
        struct = NestedStruct(simple=SimpleStruct(x="2"))
        text = struct.render()
        assert text.count('{{') == 2
        assert text.count("}}") == 2

    def test_to_dict_converts_to_a_dict(self):
        struct = SimpleStruct(x="2")
        d = struct.to_dict()
        assert d['x'] == "2"


class TestIsNestedModel:

    def test_is_nested_model_returns_true_for_nested(self):

        assert _core.is_nested_model(NestedStruct) is True

    def test_is_nested_model_returns_false_for_not_nested(self):

        assert _core.is_nested_model(SimpleStruct) is False


class TestStructList:

    def test_struct_list_retrieves_item(self):

        l = _core.StructList[SimpleStruct](
            [SimpleStruct(x='2'), SimpleStruct(x='3')]
        )
        assert l[0].x == '2'
        assert l[1].x == '3'

    def test_struct_sets_the_item(self):

        l = _core.StructList[SimpleStruct](
            [SimpleStruct(x='4'), SimpleStruct(x='5')]
        )
        l[1] = SimpleStruct(x='8')
        assert l[1].x == '8'
    
    def test_struct_sets_the_item_with_none(self):

        l = _core.StructList[SimpleStruct](
            [SimpleStruct(x='4'), SimpleStruct(x='5')]
        )
        l[None] = SimpleStruct(x='8')
        assert l[2].x == '8'
    


class Evaluation(Struct):

    text: str
    score: float



class TestInstruction:

    def test_instruction_text_is_correct(self):

        text = 'Evaluate the quality of the CSV'
        instruction = _core.Instruction(
            name='Evaluate',
            text=text
        )
        assert instruction.text == text


class TestOut:

    def test_out_creates_out_class(self):

        out = _core.StructFormatter(
            name='F1',
            out_cls=SimpleStruct
            # name='Simple', signature='...',
        )
        simple = SimpleStruct(x='hi')
        d = simple.to_text()
        simple2 = out.read(d)
        assert simple.x == simple2.x

    def test_out_creates_out_class_with_string(self):

        out = _core.StructFormatter(
            name='F1',
            out_cls=SimpleStruct
        )
        simple = SimpleStruct(x='hi')
        d = simple.to_text()
        simple2 = out.read(d)
        assert simple.x == simple2.x
    
    def test_out_template(self):

        out = _core.StructFormatter(
            name='F1',
            out_cls=SimpleStruct
        )
        simple2 = out.template()
        assert 'x' in simple2

    def test_read_reads_in_the_class(self):

        out = _core.StructFormatter(
            name='F1',
            out_cls=SimpleStruct
        )
        s = SimpleStruct(x='2').to_text()
        simple2 = out.read(s)
        assert simple2.x == '2'


class TestIsUndefined(object):

    def test_is_undefined(self):

        assert _core.is_undefined(
            _core.UNDEFINED
        )

    def test_not_is_undefined(self):

        assert not _core.is_undefined(
            1
        )


class TestInstruction(object):

    def test_instruction_renders_with_text(self):

        instruction = _core.Instruction(
            text='x'
        )
        assert instruction.render() == 'x'

    def test_read_(self):

        instruction = _core.Instruction(
            text='x', out=_core.StructFormatter(
                name='F1',
                out_cls=SimpleStruct
            )
        )
        simple = SimpleStruct(x='2')
        assert instruction.read_out(simple.to_text()).x == '2'


class TestParam(object):

    def test_get_x_from_param(self):

        instruction = _core.Param(
            name='X', instruction='x'
        )
        assert instruction.render() == 'x'

    def test_param_with_instruction_passed_in(self):

        instruction = _core.Instruction(
            text='x', out=_core.StructFormatter(
                name='F1',
                out_cls=SimpleStruct
            )
        )

        param = _core.Param(
            name='X', instruction=instruction
        )
        assert param.render() == 'x'

    def test_read_reads_the_object(self):

        instruction = _core.Instruction(
            text='x', out=_core.StructFormatter(
                name='F1',
                out_cls=SimpleStruct
            )
        )
        param = _core.Param(
            name='X', instruction=instruction
        )
        simple = SimpleStruct(x='2')
        assert param.reads(simple.to_text()).x == '2'


class TestRender:

    def test_render_renders_a_primitive(self):
        assert _core.render(1) == '1'

    def test_render_renders_a_primitive(self):
        struct = SimpleStruct(x="2")

        assert '2' in _core.render(struct)


class TestRenderMulti:

    def test_render_renders_a_primitive(self):
        assert _core.render_multi([1, 2])[0] == '1'
        assert _core.render_multi([1, 2])[1] == '2'

    def test_render_renders_a_primitive(self):
        struct = SimpleStruct(x="2")
        struct2 = SimpleStruct(x="4")

        rendered = _core.render_multi([struct, struct2])
        assert '2' in rendered[0]
        assert '4' in rendered[1]


class TestStreamer:

    def test_streamer_gets_next_item(self):

        streamer = _core.Streamer(
            iter([(1, 0), (2, 2), (3, 2)])
        )
        partial = streamer()
        assert partial.cur == 1
        assert partial.complete is False

    def test_streamer_gets_final_item(self):

        streamer = _core.Streamer(
            iter([(1, 0), (2, 2), (3, 2)])
        )
        partial = streamer()
        partial = streamer()
        partial = streamer()
        partial = streamer()
        assert partial.cur == 3
        assert partial.complete is True


class TestInstruction:
    pass


class TestParam:

    def test_param_renders_the_instruction(self):
        pass

        # streamer = _core.Streamer(
        #     iter([1, 2, 3])
        # )
        # partial = streamer()
        # assert partial.cur == 1
        # assert partial.complete is False
