from dachi._core import _core
# from dachi._core import _instruct as core
from dachi._core._core import Struct, str_formatter
import pytest
from pydantic import Field


class SimpleStruct(_core.Struct):

    x: str


class NestedStruct(_core.Struct):

    simple: SimpleStruct


class TestStruct(object):

    def test_simple_struct_gets_string(self):

        struct = SimpleStruct(name='x', x="2")
        assert struct.x == '2'
    
    def test_template_gives_correct_template(self):

        struct = SimpleStruct(name='x', x="2")
        template = struct.template()
        print(template)
        assert template['x']['is_required'] is True
        assert template['x']['type'] == type('text')

    def test_template_gives_correct_template_with_nested(self):

        struct = NestedStruct(name='x', simple=SimpleStruct(name='x', x="2"))
        template = struct.template()
        assert template['simple']['x']['is_required'] is True
        assert template['simple']['x']['type'] == type('text')

    def test_to_text_converts_to_text(self):
        struct = SimpleStruct(name='x', x="2")
        text = struct.to_text()
        assert "2" in text

    def test_to_text_doubles_the_braces(self):
        struct = SimpleStruct(name='x', x="2")
        text = struct.to_text(True)
        print(text)
        assert "{{" in text
        assert "}}" in text

    def test_to_text_works_for_nested(self):
        struct = NestedStruct(simple=SimpleStruct(name='x', x="2"))
        text = struct.to_text(True)
        assert text.count('{{') == 2
        assert text.count("}}") == 2


class Role(_core.Description):

    duty: str = Field(description='The duty of the role')

    def render(self) -> str:

        return f"""
        # Role {self.name}

        {self.duty}
        """
    
    def update(self, **kwargs) -> _core.Description:
        return Role(name=self.name, duty=str_formatter(self.duty, **kwargs))


class Evaluation(Struct):

    text: str
    score: float


class TestDescription:

    def test_text_for_description_is_correct(self):
        
        role = Role(name='Assistant', duty='You are a helpful assistant')
        text = role.render()

        assert text == f"""
        # Role Assistant

        {role.duty}
        """

    def test_text_for_description_is_correct_after_updating(self):
        
        role = Role(name='Assistant', duty='You are a helpful {role}')
        
        role = role.update(role='Sales Assistant')
        text = role.render()
        assert 'Sales Assistant' in text


class TestRef:

    def test_ref_does_not_output_text(self):

        role = Role(name='Assistant', duty='You are a helpful Helpful Assistant')
        ref = _core.Ref(desc=role)
        assert 'Helpful Assistant' in ref.desc.render()

    def test_name_returns_name_of_reference(self):

        role = Role(name='Assistant', duty='You are a helpful Helpful Assistant')
        ref = _core.Ref(desc=role)
        # ref = ref.update(role='Helpful Assistant')
        assert ref.name == 'Assistant'

    def test_text_is_empty_string(self):

        role = Role(name='Assistant', duty='You are a helpful Helpful Assistant')
        ref = _core.Ref(desc=role)
        # ref = ref.update(role='Helpful Assistant')
        assert ref.render() == role.name


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

        out = _core.Out(
            name='F1',
            out_cls=SimpleStruct
            # name='Simple', signature='...',
        )
        simple = SimpleStruct(x='hi')
        d = simple.to_text()
        simple2 = out.read(d)
        assert simple.x == simple2.x

    def test_out_creates_out_class_with_string(self):

        out = _core.Out(
            name='F1',
            out_cls=SimpleStruct
        )
        simple = SimpleStruct(x='hi')
        d = simple.to_text()
        simple2 = out.read(d)
        assert simple.x == simple2.x
    
    def test_out_template(self):

        out = _core.Out(
            name='F1',
            out_cls=SimpleStruct
        )
        simple2 = out.out_template()
        assert 'x' in simple2


class TestStrFormatter(object):

    def test_formatter_formats_positional_variables(self):

        assert _core.str_formatter(
            '{} {}', 1, 2
        ) == '1 2'

    def test_formatter_formats_positional_variables(self):

        assert _core.str_formatter(
            '{0} {1}', 1, 2
        ) == '1 2'

    def test_formatter_formats_named_variables(self):

        assert _core.str_formatter(
            '{x} {y}', x=1, y=2
        ) == '1 2'

    def test_formatter_raises_error_if_positional_and_named_variables(self):

        with pytest.raises(ValueError):
            _core.str_formatter(
                '{0} {y}', 1, y=2
            )

    def test_get_variables_gets_all_pos_variables(self):

        assert _core.get_str_variables(
            '{0} {1}'
        ) == [0, 1]

    def test_get_variables_gets_all_named_variables(self):

        assert _core.get_str_variables(
            '{x} {y}'
        ) == ['x', 'y']


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
            text='x', out=_core.Out(
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
            text='x', out=_core.Out(
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
            text='x', out=_core.Out(
                name='F1',
                out_cls=SimpleStruct
            )
        )
        param = _core.Param(
            name='X', instruction=instruction
        )
        simple = SimpleStruct(x='2')
        assert param.reads(simple.to_text()).x == '2'


class TestListOut(object):

    def test_out_reads_in_the_class(self):

        struct_list = _core.StructList(
            structs=[
                SimpleStruct(x='2'),
                SimpleStruct(x='3')
            ]
        )

        out = _core.ListOut(
            name='F1',
            out_cls=SimpleStruct
        )

        assert out.read(struct_list.to_text())[0].x == '2'

    def test_out_reads_in_the_class_with_str(self):

        out = _core.Out(
            name='F1',
            out_cls=SimpleStruct
        )

        simple = SimpleStruct(x='2')
        assert out.stream_read(simple.to_text()).x == '2'


class TestListOut(object):

    def test_out_reads_in_the_class(self):

        struct_list = _core.StructList(
            structs=[
                SimpleStruct(x='2'),
                SimpleStruct(x='3')
            ]
        )

        out = _core.ListOut(
            name='F1',
            out_cls=SimpleStruct
        )

        assert out.read(struct_list.to_text())[0].x == '2'

    def test_out_reads_in_the_class_with_str(self):

        out = _core.Out(
            name='F1',
            out_cls=SimpleStruct
        )

        simple = SimpleStruct(x='2')
        assert out.stream_read(simple.to_text()).x == '2'


class TestMultiOut(object):

    def test_out_writes_in_the_class(self):

        struct_list = [
            SimpleStruct(x='2'),
            SimpleStruct(x='3')
        ]

        out = _core.MultiOut(
            outs=[_core.Out(
                name='F1',
                out_cls=SimpleStruct
            ), _core.Out(
                name='F2',
                out_cls=SimpleStruct
            )]
        )

        text = out.write(struct_list)
        assert 'x' in text
        assert 'F2' in text

    def test_out_reads_in_the_class(self):

        struct_list = [
            SimpleStruct(x='2'),
            SimpleStruct(x='3')
        ]

        out = _core.MultiOut(
            outs=[_core.Out(
                name='F1',
                out_cls=SimpleStruct
            ), _core.Out(
                name='F2',
                out_cls=SimpleStruct
            )]
        )

        text = out.write(struct_list)
        structs = out.read(text)
        assert structs[0].x == struct_list[0].x

    def test_out_stream_read_in_the_class(self):

        struct_list = [
            SimpleStruct(x='2'),
            SimpleStruct(x='3')
        ]

        out = _core.MultiOut(
            outs=[_core.Out(
                name='F1',
                out_cls=SimpleStruct
            ), _core.Out(
                name='F2',
                out_cls=SimpleStruct
            )]
        )

        text = out.write(struct_list)
        structs, failed_on = out.stream_read(text)
        assert structs[0].x == struct_list[0].x
        assert failed_on is None
