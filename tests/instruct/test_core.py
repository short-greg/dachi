from dachi._core import _struct
from dachi.instruct import _core as instruct
import json
import typing
import pydantic


class SimpleStruct(_struct.Struct):

    x: _struct.Str


class T(_struct.Struct):

    s: str


class Y(_struct.Struct):

    ts: typing.List[T]


class JSONStyle(instruct.RStyle[SimpleStruct]):

    def reverse(self, text: str) -> SimpleStruct:

        return SimpleStruct.model_validate(
            json.loads(text)
        )
    
    def forward(self, struct: SimpleStruct) -> str:

        return struct.model_dump_json()


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


class TestIVar(object):

    def test_name_equals_name(self):

        name = 'I'
        text = 'Text for var'
        var = instruct.IVar(
            name=name, text=text
        )
        assert name == var.name
        assert text == var.text


class TestInstruction(object):

    def test_instruction(self):

        name = 'X1'
        style = JSONStyle()

        instruction = instruct.Instruction(
            name=name, style=style
        )
        assert instruction.name == 'X1'
        assert isinstance(
            instruction.style,
            JSONStyle
        )


class TestMaterial(object):

    def test_material(self):

        name = 'X1'
        style = JSONStyle()

        instruction = instruct.Instruction(
            name=name, style=style
        )
        assert instruction.name == 'X1'
        assert isinstance(
            instruction.style,
            JSONStyle
        )


# class TestOp(object):

#     def test_op_forward_outputs_string(self):

#         name = 'X1'
#         ivar = instruct.IVar(
#             name=name, text='hi'
#         )

#         instruction = instruct.Op(
#             [ivar], 'list_outputs', 'out'
#         )
#         assert instruction.name == 'out'
