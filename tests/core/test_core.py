from dachi._core import _core
# from dachi._core import _instruct as core
from dachi._core._core import Struct, str_formatter


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
        assert template['x']['is_required'] is True
        assert template['x']['type'] == type('text')

    def test_template_gives_correct_template_with_nested(self):

        struct = NestedStruct(name='x', simple=SimpleStruct(name='x', x="2"))
        template = struct.template()
        assert template['simple']['x']['is_required'] is True
        assert template['simple']['x']['type'] == type('text')


class Role(_core.Description):

    duty: str

    def render(self) -> str:

        return f"""
        # Role

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
        # Role

        {role.duty}
        """

    def test_text_for_description_is_correct_after_updating(self):
        
        role = Role(name='Assistant', duty='You are a helpful {role}')
        
        role = role.update(role='Sales Assistant')
        text = role.render()
        assert 'Sales Assistant' in text


class TestRef:

    def test_ref_does_not_output_text(self):

        role = Role(name='Assistant', duty='You are a helpful {role}')
        ref = _core.Ref(reference=role)
        ref = ref.update(role='Helpful Assistant')
        assert 'Helpful Assistant' not in ref.reference.render()

    def test_name_returns_name_of_reference(self):

        role = Role(name='Assistant', duty='You are a helpful {role}')
        ref = _core.Ref(reference=role)
        ref = ref.update(role='Helpful Assistant')
        assert ref.name == 'Assistant'

    def test_text_is_empty_string(self):

        role = Role(name='Assistant', duty='You are a helpful {role}')
        ref = _core.Ref(reference=role)
        ref = ref.update(role='Helpful Assistant')
        assert ref.render() == ''


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
            name='Simple', signature='...',
            out_cls=SimpleStruct
        )
        simple = SimpleStruct(x='hi')
        d = simple.dump()
        simple2 = out.read(d)
        assert simple.x == simple2.x

    def test_out_creates_out_class_with_string(self):

        out = _core.Out(
            name='Simple', signature='...',
            out_cls=SimpleStruct
        )
        simple = SimpleStruct(x='hi')
        d = simple.dumps()
        simple2 = out.reads(d)
        assert simple.x == simple2.x
    
    def test_out_template(self):

        out = _core.Out(
            name='Simple', signature='...',
            out_cls=SimpleStruct
        )
        simple2 = out.out_template()
        assert 'x' in simple2
