from dachi.store import _struct
from dachi.instruct import _core as instruct
import json
import typing


class SimpleStruct(_struct.Struct):

    x: _struct.Str


class T(_struct.Struct):

    s: str


class Y(_struct.Struct):

    ts: typing.List[T]


class JSONStyle(instruct.ReversibleStyle[SimpleStruct]):

    def reverse(self, text: str) -> SimpleStruct:

        return SimpleStruct.model_validate(
            json.loads(text)
        )
    
    def forward(self, struct: SimpleStruct) -> str:

        return struct.model_dump_json()

import pydantic

class Z(_struct.Struct):

    y: _struct.Str


class TestStyle(object):

    def test_style_converts_to_string(self):

        style = JSONStyle()
        struct = SimpleStruct(x="2")
        json_ = style(struct)
        d = json.loads(json_)
        assert d["x"]["text"] == "2"

    def test_reverse_creates_struct(self):

        style = JSONStyle()
        
        struct = SimpleStruct(x='hi!')
        json_ = style(struct)

        struct = style.reverse(json_)
        assert struct.x.text == "hi!"
