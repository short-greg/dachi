from dachi._core import _core
import asyncio
import pytest
from pydantic import BaseModel

from typing import Any, Iterator, Tuple

from dachi._core._core import Module
import typing


class SimpleStruct(BaseModel):

    x: str


class SimpleStruct2(BaseModel):

    x: str
    y: int


class NestedStruct(BaseModel):

    simple: SimpleStruct


class WriteOut(Module):

    def forward(self, x: str) -> str:

        return x

    def stream(self, x: str) -> Iterator[Tuple[Any, Any]]:
        
        out = ''
        for c in x:
            out = out + c
            yield out, c


class Evaluation(BaseModel):

    text: str
    score: float



# Test Storable


class TestCue(object):

    def test_instruction_renders_with_text(self):

        cue = _core.Cue(
            text='x'
        )
        assert cue.render() == 'x'

    # def test_read_(self):

    #     instruction = _core.Cue(
    #         text='x', out=_core.StructRead(
    #             name='F1',
    #             out_cls=SimpleStruct
    #         )
    #     )
    #     simple = SimpleStruct(x='2')
    #     assert instruction.read(simple.to_text()).x == '2'

    def test_instruction_text_is_correct(self):

        text = 'Evaluate the quality of the CSV'
        cue = _core.Cue(
            name='Evaluate',
            text=text
        )
        assert cue.text == text

    def test_render_returns_the_instruction_text(self):

        text = 'Evaluate the quality of the CSV'
        cue = _core.Cue(
            name='Evaluate',
            text=text
        )
        assert cue.render() == text

    def test_i_returns_the_instruction(self):

        text = 'Evaluate the quality of the CSV'
        cue = _core.Cue(
            name='Evaluate',
            text=text
        )
        assert cue.i() is cue


class TestParam(object):

    def test_get_x_from_param(self):

        cue = _core.Param(
            name='X', cue='x'
        )
        assert cue.render() == 'x'

    # def test_param_with_instruction_passed_in(self):

    #     instruction = _core.Cue(
    #         text='x', out=_core.StructRead(
    #             name='F1',
    #             out_cls=SimpleStruct
    #         )
    #     )

    #     param = _core.Param(
    #         name='X', instruction=instruction
    #     )
    #     assert param.render() == 'x'

#     def test_read_reads_the_object(self):

#         instruction = _core.Cue(
#             text='x', out=_core.StructRead(
#                 name='F1',
#                 out_cls=SimpleStruct
#             )
#         )
#         param = _core.Param(
#             name='X', instruction=instruction
#         )
#         simple = SimpleStruct(x='2')
#         assert param.reads(simple.to_text()).x == '2'



# Test render: Add more tests - Two few


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
            cue=_core.Cue(text='Do this')
        )

    def forward(self) -> Any:
        return None


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
                        tg.create_task(module.aforward(data_i))
                    )

            return list(task.result() for task in tasks)

        with asyncio.Runner() as runner:
            results = runner.run(run_model(['hi', 'bye']))
        
        assert results[0] == 'hit'
        assert results[1] == 'byet'

    def test_stream_forward_returns_the_results(self):
        
        module = Append('t')

        res = ''
        for x, dx in module.stream('xyz'):
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

        children = list(module._parameters())
        assert len(children) == 2

    def test_streamable_streams_characters(self):

        writer = WriteOut()
        results = []
        for x, dx in writer.stream('xyz'):
            results.append(dx)
        assert results == list('xyz')


class TestParam:

    def test_param_renders_the_instruction(self):

        param = _core.Param(
            name='p',
            cue=_core.Cue(
                text='simple instruction'
            )
        )
        target = param.cue.render()
        assert param.render() == target

    def test_param_updates_the_instruction(self):

        param = _core.Param(
            name='p',
            cue=_core.Cue(
                text='simple instruction'
            ),
            training=True
        )
        target = 'basic instruction'
        param.update(target)
        assert param.render() == target
