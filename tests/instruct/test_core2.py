from dachi.instruct import _core2 as core
from dachi._core._struct import Struct, Str


class Role(core.Description):

    duty: Str

    def text(self) -> str:

        return f"""
        # Role

        {self.duty.text}
        """
    
    def update(self, **kwargs) -> core.Description:
        return Role(name=self.name, duty=self.duty(**kwargs))


class TestDescription:

    def test_text_for_description_is_correct(self):
        
        role = Role(name='Assistant', duty='You are a helpful assistant')
        text = role.text()

        print(text)
        assert text == f"""
        # Role

        {role.duty.text}
        """

    def test_text_for_description_is_correct_after_updating(self):
        
        role = Role(name='Assistant', duty=Str(text='You are a helpful {role}', vars=['role']))
        
        role = role.update(role='Sales Assistant')
        text = role.text()
        assert 'Sales Assistant' in text


class TestRef:

    def test_ref_does_not_output_text(self):

        role = Role(name='Assistant', duty=Str(text='You are a helpful {role}', vars=['role']))
        ref = core.Ref(reference=role)
        ref = ref.update(role='Helpful Assistant')
        assert 'Helpful Assistant' not in ref.reference.text()

    def test_name_returns_name_of_reference(self):

        role = Role(name='Assistant', duty=Str(text='You are a helpful {role}', vars=['role']))
        ref = core.Ref(reference=role)
        ref = ref.update(role='Helpful Assistant')
        assert ref.name == 'Assistant'

    def test_text_is_empty_string(self):

        role = Role(name='Assistant', duty=Str(text='You are a helpful {role}', vars=['role']))
        ref = core.Ref(reference=role)
        ref = ref.update(role='Helpful Assistant')
        assert ref.text == ''


class TestInstruction:

    def test_instruction_text_is_correct(self):

        text = 'Evaluate the quality of the CSV'
        instruction = core.Instruction(
            name='Evaluate', 
            instr=text
        )
        assert instruction.text == text

    def test_iter_gets_all_incoming_instructions(self):

        text = 'Evaluate the quality of the CSV'

        instruction1 = core.Instruction(
            name='Evaluate1', 
            instr=text
        )

        instruction2 = core.Instruction(
            name='Evaluate2', 
            instr=text
        )

        instruction = core.Instruction(
            name='Evaluate',
            instr=text,
            incoming=[instruction1, instruction2]
        )
        assert instruction1 in instruction.incoming
        assert instruction2 in instruction.incoming

    def test_traverse_gets_all_instructions(self):

        text = 'Evaluate the quality of the CSV'

        instruction1 = core.Instruction(
            name='Evaluate1', 
            instr=text
        )

        instruction2 = core.Instruction(
            name='Evaluate2', 
            instr=text
        )

        instruction3 = core.Instruction(
            name='Evaluate',
            instr=text,
            incoming=[instruction1, instruction2]
        )

        instruction4 = core.Instruction(
            name='Evaluate',
            instr=text,
            incoming=[instruction3]
        )
        instructions = list(instruction4.traverse())
        assert instruction1 in instructions
        assert instruction2 in instructions


class TestOp:

    def test_op_outputs_an_instruction(self):

        role = Role(name='Assistant', duty='You are a helpful assistant')

        text = 'Evaluate the user'
        instruction = core.op(
            [role], text, 'Evaluate'
        )
        assert role in instruction.incoming


class Evaluation(Struct):

    text: str
    score: float


class TestOutput:

    def test_output_text_has_reference(self):

        text = 'Evaluate the user'
        role = Role(
            name='Assistant', 
            duty='You are a helpful assistant'
        )

        instruction = core.op(
            [role], text, 'Evaluate'
        )
        output = core.Output[Evaluation](
            instruction=instruction,
            name='Evaluation',
        )
        assert 'Assistant' in output.text

    def test_read_reads_the_json(self):

        text = 'Evaluate the user'
        role = Role(
            name='Assistant', 
            duty='You are a helpful assistant'
        )

        instruction = core.op(
            [role], text, 'Evaluate'
        )

        evaluation = Evaluation(text='the user did well', score=1.0)
        d = evaluation.model_dump_json()

        output = core.Output[Evaluation](
            instruction=instruction,
            name='Evaluation',
        )
        reconstructed = output.read(d)
        assert reconstructed.score == evaluation.score


class TestOutputList:

    def test_output_list_outputs_multiple_text_has_reference(self):

        text = 'Evaluate the user'
        role = Role(
            name='Assistant', 
            duty='You are a helpful assistant'
        )

        instruction = core.op(
            [role], text, 'Evaluate'
        )
        output = core.Output[Evaluation](
            instruction=instruction,
            name='Evaluation',
        )

        output_list = core.OutputList(
            outputs=[output, output]
        )

        print(output_list.text)
        assert 'Assistant' in output_list.text

    def test_output_list_outputs_multiple_text_with_header(self):

        text = 'Evaluate the user'
        role = Role(
            name='Assistant', 
            duty='You are a helpful assistant'
        )

        instruction = core.op(
            [role], text, 'Evaluate'
        )
        output = core.Output[Evaluation](
            instruction=instruction,
            name='Evaluation',
        )

        output_list = core.OutputList(
            outputs=[output, output]
        )

        print(output_list.text)
        assert 'Assistant' in output_list.text

    def test_read_reads_the_json(self):

        text = 'Evaluate the user'
        role = Role(
            name='Assistant', 
            duty='You are a helpful assistant'
        )

        instruction = core.op(
            [role], text, 'Evaluate'
        )

        evaluation1 = Evaluation(text='the user did well', score=1.0)
        evaluation2 = Evaluation(text='the user did poorly', score=0.0)

        d1 = evaluation1.model_dump_json()
        d2 = evaluation2.model_dump_json()

        d = f"""
        ::OUT::Evaluation::

        {d1}

        ::OUT::Evaluation2::

        {d2}
        """

        output = core.Output[Evaluation](
            instruction=instruction,
            name='Evaluation',
        )
        output2 = core.Output[Evaluation](
            instruction=instruction,
            name='Evaluation2',
        )
        output_list = core.OutputList(
            outputs=[output, output2]
        )
        results = output_list.read(
            d
        )

        assert results[0].score == 1.0
        assert results[1].score == 0.0
