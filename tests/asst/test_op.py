from .test_ai import DummyAIModel
from dachi.asst import LineParser, Op, NullOutConv, ToText
from dachi.asst import MR
import pytest

lines = """hi
this is
some lines
"""

class TestOp:

    def test_op_handles_empty_input(self):
        """Test Op with empty input to ensure it handles edge cases gracefully."""
        op = Op(
            DummyAIModel("", proc=[NullOutConv('out', 'content')]), ToText(),
            out='out'
        )
        r = op.forward("")
        assert r == ''

    def test_op_handles_large_input(self):
        """Test Op with a very large input to ensure it processes correctly."""
        large_input = "\n".join([f"line {i}" for i in range(1000)])
        op = Op(
            DummyAIModel(
                large_input, 
                proc=[LineParser('data'), NullOutConv('out', 'data')]
            ), ToText(),
            out='out'
        )
        r = op.forward("Message")
        assert len(r) == 1000
        assert r[0] == "line 0"
        assert r[-1] == "line 999"

    def test_op_stream_handles_empty_input(self):
        """Test Op's stream method with empty input."""
        op = Op(
            DummyAIModel("", proc=[NullOutConv('out', 'content')]), ToText(),
            out='out'
        )
        rs = list(op.stream(""))
        assert rs == []

    def test_op_stream_handles_large_input(self):
        """Test Op's stream method with a very large input."""
        large_input = "\n".join([f"line {i}" for i in range(1000)])
        op = Op(
            DummyAIModel(
                large_input, proc=[LineParser('data', 'content'), NullOutConv('out', 'data')]
            ), ToText(), out='out', filter_undefined=True
        )
        rs = []
        for val in op.stream("Message"):
            rs.extend(val)

        assert len(rs) == 1000
        assert rs[0] == "line 0"
        assert rs[-1] == "line 999"

    # def test_op_handles_non_string_input(self):
    #     """Test Op with non-string input to ensure type robustness."""
    #     op = Op(
    #         DummyAIModel(lines, proc=[LineParser('data'), NullOutConv('out', 'data')]), ToText(),
    #         out='out'
    #     )
    #     with pytest.raises(TypeError):
    #         op.forward(12345)

    # def test_op_stream_handles_non_string_input(self):
    #     """Test Op's stream method with non-string input."""
    #     op = Op(
    #         DummyAIModel(lines, proc=[LineParser('data'), NullOutConv('out', 'data')]), ToText(),
    #         out='out'
    #     )
    #     with pytest.raises(TypeError):
    #         list(op.stream(12345))

    def test_op_handles_special_characters(self):
        """Test Op with input containing special characters."""
        special_input = "hello\nworld\n!@#$%^&*()"
        op = Op(
            DummyAIModel(special_input, proc=[LineParser('data'), NullOutConv('out', 'data')]), ToText(),
            out='out'
        )
        r = op.forward("Message")
        assert r == ["hello", "world", "!@#$%^&*()"]

    def test_op_stream_handles_special_characters(self):
        """Test Op's stream method with input containing special characters."""
        special_input = "hello\nworld\n!@#$%^&*()"
        op = Op(
            DummyAIModel(special_input, proc=[LineParser('data'), NullOutConv('out', 'data')]), ToText(),
            out='out'
        )
        rs = []
        for res in op.stream("Message"):
            rs.extend(res)
        assert rs == ["hello", "world", "!@#$%^&*()"]

    def test_op_handles_partial_streaming(self):
        """Test Op's stream method to ensure partial results are returned."""
        partial_input = "line 1\nline 2\nline 3"
        op = Op(
            DummyAIModel(partial_input, proc=[LineParser('data'), NullOutConv('out', 'data')]), ToText(),
            out='out'
        )
        stream = op.stream("Message")
        assert next(stream) == ["line 1"]
        assert next(stream) == ["line 2"]
        assert next(stream) == ["line 3"]

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
            rs.extend(r_i)
        rs[0] == 'hi'
        rs[1] == 'this is'
        rs[2] == 'some lines'
