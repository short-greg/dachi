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


class Append(Process):

    def __init__(self, append: str):
        super().__init__()
        self._append = append

    def delta(self, name: str='') -> Any:
        return name + self._append
