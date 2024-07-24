from dachi._core import _instruct as core
from dachi._core._struct import Struct, str_formatter
from .test_struct import SimpleStruct


class Role(core.Description):

    duty: str

    def render(self) -> str:

        return f"""
        # Role

        {self.duty}
        """
    
    def update(self, **kwargs) -> core.Description:
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
        ref = core.Ref(reference=role)
        ref = ref.update(role='Helpful Assistant')
        assert 'Helpful Assistant' not in ref.reference.render()

    def test_name_returns_name_of_reference(self):

        role = Role(name='Assistant', duty='You are a helpful {role}')
        ref = core.Ref(reference=role)
        ref = ref.update(role='Helpful Assistant')
        assert ref.name == 'Assistant'

    def test_text_is_empty_string(self):

        role = Role(name='Assistant', duty='You are a helpful {role}')
        ref = core.Ref(reference=role)
        ref = ref.update(role='Helpful Assistant')
        assert ref.render() == ''


class TestInstruction:

    def test_instruction_text_is_correct(self):

        text = 'Evaluate the quality of the CSV'
        instruction = core.Instruction(
            name='Evaluate',
            text=text
        )
        assert instruction.text == text



class TestOp:

    def test_op_outputs_an_instruction(self):

        role = Role(name='Assistant', duty='You are a helpful assistant')

        text = 'Evaluate the user'
        instruction = core.op(
            [role], text, 'Evaluate'
        )
        assert 'Assistant' in instruction.text


class TestOut:

    def test_out_creates_out_class(self):

        out = core.Out(
            name='Simple', signature='...',
            out_cls=SimpleStruct
        )
        simple = SimpleStruct(x='hi')
        d = simple.dump()
        simple2 = out.read(d)
        assert simple.x == simple2.x

    def test_out_creates_out_class_with_string(self):

        out = core.Out(
            name='Simple', signature='...',
            out_cls=SimpleStruct
        )
        simple = SimpleStruct(x='hi')
        d = simple.dumps()
        simple2 = out.reads(d)
        assert simple.x == simple2.x
    
    def test_out_template(self):

        out = core.Out(
            name='Simple', signature='...',
            out_cls=SimpleStruct
        )
        simple2 = out.out_template()
        assert 'x' in simple2


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

        out = core.fill(
            struct, x2=2
        )
        assert out.text == '2'

# class TestOutput:

#     def test_output_text_has_reference(self):

#         text = 'Evaluate the user'
#         role = Role(
#             name='Assistant', 
#             duty='You are a helpful assistant'
#         )

#         instruction = core.op(
#             [role], text, 'Evaluate'
#         )
#         output = core.Output[Evaluation](
#             instruction=instruction.
#             name='Evaluation',
#         )
#         assert 'Assistant' in output.text

#     def test_read_reads_the_json(self):

#         text = 'Evaluate the user'
#         role = Role(
#             name='Assistant', 
#             duty='You are a helpful assistant'
#         )

#         instruction = core.op(
#             [role], text, 'Evaluate'
#         )

#         evaluation = Evaluation(text='the user did well', score=1.0)
#         d = evaluation.model_dump_json()

#         output = core.Output[Evaluation](
#             instruction=instruction,
#             name='Evaluation',
#         )
#         reconstructed = output.read(d)
#         assert reconstructed.score == evaluation.score


# class TestOutputList:

#     def test_output_list_outputs_multiple_text_has_reference(self):

#         text = 'Evaluate the user'
#         role = Role(
#             name='Assistant', 
#             duty='You are a helpful assistant'
#         )

#         instruction = core.op(
#             [role], text, 'Evaluate'
#         )
#         output = core.Output[Evaluation](
#             instruction=instruction,
#             name='Evaluation',
#         )

#         output_list = core.OutputList(
#             outputs=[output, output]
#         )

#         print(output_list.text)
#         assert 'Assistant' in output_list.text

#     def test_output_list_outputs_multiple_text_with_header(self):

#         text = 'Evaluate the user'
#         role = Role(
#             name='Assistant', 
#             duty='You are a helpful assistant'
#         )

#         instruction = core.op(
#             [role], text, 'Evaluate'
#         )
#         output = core.Output[Evaluation](
#             instruction=instruction,
#             name='Evaluation',
#         )

#         output_list = core.OutputList(
#             outputs=[output, output]
#         )

#         print(output_list.text)
#         assert 'Assistant' in output_list.text

#     # def test_read_reads_the_json(self):

#     #     text = 'Evaluate the user'
#     #     role = Role(
#     #         name='Assistant', 
#     #         duty='You are a helpful assistant'
#     #     )

#     #     instruction = core.op(
#     #         [role], text, 'Evaluate'
#     #     )

#     #     evaluation1 = Evaluation(text='the user did well', score=1.0)
#     #     evaluation2 = Evaluation(text='the user did poorly', score=0.0)

#     #     d1 = evaluation1.model_dump_json()
#     #     d2 = evaluation2.model_dump_json()

#     #     d = f"""
#     #     ::OUT::Evaluation::

#     #     {d1}

#     #     ::OUT::Evaluation2::

#     #     {d2}
#     #     """

#     #     output = core.Output[Evaluation](
#     #         instruction=instruction,
#     #         name='Evaluation',
#     #     )
#     #     output2 = core.Output[Evaluation](
#     #         instruction=instruction,
#     #         name='Evaluation2',
#     #     )
#     #     output_list = core.OutputList(
#     #         outputs=[output, output2]
#     #     )
#     #     results = output_list.read(
#     #         d
#     #     )

#     #     assert results[0].score == 1.0
#     #     assert results[1].score == 0.0


# # Instruction

#     def test_iter_gets_all_incoming_instructions(self):

#         text = 'Evaluate the quality of the CSV'

#         instruction1 = core.Instruction(
#             name='Evaluate1', 
#             text=text
#         )

#         instruction2 = core.Instruction(
#             name='Evaluate2', 
#             text=text
#         )

#         instruction = core.Instruction(
#             name='Evaluate',
#             text=text,
#             incoming=[instruction1, instruction2]
#         )
#         assert instruction1 in instruction.incoming
#         assert instruction2 in instruction.incoming

#     def test_traverse_gets_all_instructions(self):

#         text = 'Evaluate the quality of the CSV'

#         instruction1 = core.Instruction(
#             name='Evaluate1', 
#             text=text
#         )

#         instruction2 = core.Instruction(
#             name='Evaluate2', 
#             text=text
#         )

#         instruction3 = core.Instruction(
#             name='Evaluate',
#             text=text,
#             incoming=[instruction1, instruction2]
#         )

#         instruction4 = core.Instruction(
#             name='Evaluate',
#             text=text,
#             incoming=[instruction3]
#         )
#         instructions = list(instruction4.traverse())
#         assert instruction1 in instructions
#         assert instruction2 in instructions