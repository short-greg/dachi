from .test_ai import DummyAIModel
from dachi.asst import LineParser, Op, NullOutConv, ToText
from dachi.asst import MR

lines = """hi
this is
some lines
"""

class TestOp:

    def test_op_parses_and_returns_line_values(self):

        op = Op(
            DummyAIModel(lines, proc=[LineParser('data'), NullOutConv('out', 'data')]), ToText(), 
            out='out'
        )
        r = op.forward("Message")
        assert r[0] == 'hi'
        assert r[1] == 'this is'
        assert r[2] == 'some lines'

    def test_op_parses_and_returns_line_values_with_streaming(self):

        op = Op(
            DummyAIModel(lines, proc=[LineParser('data'), NullOutConv('out', 'data')]), ToText(),
            out='out'
        )
        rs = []
        for r_i in op.stream("Message"):
            rs.append(r_i)
        rs[0] == 'hi'
        rs[1] == 'this is'
        rs[2] == 'some lines'
