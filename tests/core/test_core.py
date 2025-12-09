from dachi.core import render, render_multi
from pydantic import BaseModel
from typing import Any, Iterator, Tuple
from dachi.proc._process import Process


class SimpleStruct(BaseModel):

    x: str


class SimpleStruct2(BaseModel):

    x: str
    y: int


class NestedStruct(BaseModel):

    simple: SimpleStruct


class WriteOut(Process):

    def delta(self, x: str) -> str:

        return x

    def stream(self, x: str) -> Iterator[Tuple[Any, Any]]:
        
        out = ''
        for c in x:
            out = out + c
            yield out, c


class Evaluation(BaseModel):

    text: str
    score: float

# Test render: Add more tests - Two few


class TestRender:

    def test_render_renders_a_primitive(self):
        assert render(1) == '1'

    def test_render_renders_a_primitive(self):
        struct = SimpleStruct(x="2")

        assert '2' in render(struct)


class TestRenderMulti:

    def test_render_renders_a_primitive(self):
        assert render_multi([1, 2])[0] == '1'
        assert render_multi([1, 2])[1] == '2'

    def test_render_renders_a_primitive(self):
        struct = SimpleStruct(x="2")
        struct2 = SimpleStruct(x="4")

        rendered = render_multi([struct, struct2])
        assert '2' in rendered[0]
        assert '4' in rendered[1]


class Append(Process):

    def __init__(self, append: str):
        super().__init__()
        self._append = append

    def delta(self, name: str='') -> Any:
        return name + self._append
