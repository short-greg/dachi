from dachi._core import _core
# from dachi._core import _instruct as core
from dachi._core import Struct, str_formatter
import pytest
from pydantic import Field

import asyncio
from typing import Any, Iterator, Tuple
import pytest
from dachi._core import _process as p
from dachi._core._core import Module
import typing


class SimpleStruct(_core.Struct):

    x: str


class SimpleStruct2(_core.Struct):

    x: str
    y: int


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



class Evaluation(Struct):

    text: str
    score: float



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
            text='x', out=_core.StructRead(
                name='F1',
                out_cls=SimpleStruct
            )
        )
        simple = SimpleStruct(x='2')
        assert instruction.read(simple.to_text()).x == '2'

    def test_instruction_text_is_correct(self):

        text = 'Evaluate the quality of the CSV'
        instruction = _core.Instruction(
            name='Evaluate',
            text=text
        )
        assert instruction.text == text

    def test_render_returns_the_instruction_text(self):

        text = 'Evaluate the quality of the CSV'
        instruction = _core.Instruction(
            name='Evaluate',
            text=text
        )
        assert instruction.render() == text

    def test_i_returns_the_instruction(self):

        text = 'Evaluate the quality of the CSV'
        instruction = _core.Instruction(
            name='Evaluate',
            text=text
        )
        assert instruction.i() is instruction


class TestParam(object):

    def test_get_x_from_param(self):

        instruction = _core.Param(
            name='X', instruction='x'
        )
        assert instruction.render() == 'x'

    def test_param_with_instruction_passed_in(self):

        instruction = _core.Instruction(
            text='x', out=_core.StructRead(
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
            text='x', out=_core.StructRead(
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


class TestInstruction:
    pass



class Append(Module):

    def __init__(self, append: str):
        super().__init__()
        self._append = append

    def forward(self, name: str='') -> Any:
        return name + self._append


class NestedModule(Module):

    def __init__(self, child: Module):
        super().__init__()
        self.child = child
        self.p = _core.Param(
            name='p',
            instruction=_core.Instruction(text='Do this')
        )

    def forward(self) -> Any:
        return None


class WriteOut(Module):

    def forward(self, x: str) -> str:

        return x

    def stream_forward(self, x: str) -> Iterator[Tuple[Any, Any]]:
        
        out = ''
        for c in x:
            out = out + c
            yield out, c


class TestModule:

    def test_module_forward_outputs_correct_value(self):

        append = Append('_t')
        assert append.forward('x') == 'x_t'

    def test_async_runs_the_model_asynchronously(self):
        
        module = Append('t')

        async def run_model(data: typing.List[str]):

            tasks = []
            async with asyncio.TaskGroup() as tg:
                for data_i in data:
                    tasks.append(
                        tg.create_task(module.async_forward(data_i))
                    )

            return list(task.result() for task in tasks)

        with asyncio.Runner() as runner:
            results = runner.run(run_model(['hi', 'bye']))
        
        assert results[0] == 'hit'
        assert results[1] == 'byet'

    def test_stream_forward_returns_the_results(self):
        
        module = Append('t')

        res = ''
        for x, dx in module.stream_forward('xyz'):
            res += dx
        assert res == 'xyzt'

    def test_children_returns_no_children(self):
        
        module = Append('t')

        children = list(module.children())
        assert len(children) == 0

    def test_children_returns_two_children_with_nested(self):
        
        module = NestedModule(NestedModule(Append('a')))

        children = list(module.children())
        assert len(children) == 2

    def test_parameters_returns_all_parameters(self):
        
        module = NestedModule(NestedModule(Append('a')))

        children = list(module.parameters())
        assert len(children) == 2

    def test_streamable_streams_characters(self):

        writer = WriteOut()
        results = []
        for x, dx in writer.stream_forward('xyz'):
            results.append(dx)
        assert results == list('xyz')


class TestParam:

    def test_param_renders_the_instruction(self):

        param = _core.Param(
            name='p',
            instruction=_core.Instruction(
                text='simple instruction'
            )
        )
        target = param.instruction.render()
        assert param.render() == target

    def test_param_updates_the_instruction(self):

        param = _core.Param(
            name='p',
            instruction=_core.Instruction(
                text='simple instruction'
            ),
            training=True
        )
        target = 'basic instruction'
        param.update(target)
        assert param.render() == target
