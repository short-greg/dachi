from .test_ai import DummyAIModel
from dachi.asst import LineParser, Op, Threaded, NullOut, ToText
from dachi.asst import KeyRet
from dachi.utils import Args
import pytest

lines = """hi
this is
some lines
"""

class TestOp:

    def test_op_handles_empty_input(self):
        """Test Op with empty input to ensure it handles edge cases gracefully."""
        op = Op(
            DummyAIModel("", proc=[NullOut('out', 'content')]), ToText(),
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
                proc=[LineParser('data'), NullOut('out', 'data')]
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
            DummyAIModel("", proc=[NullOut('out', 'content')]), ToText(),
            out='out'
        )
        rs = list(op.stream(""))
        assert rs == []

    def test_op_stream_handles_large_input(self):
        """Test Op's stream method with a very large input."""
        large_input = "\n".join([f"line {i}" for i in range(1000)])
        op = Op(
            DummyAIModel(
                large_input, proc=[LineParser('data', 'content'), NullOut('out', 'data')]
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
            DummyAIModel(special_input, proc=[LineParser('data'), NullOut('out', 'data')]), ToText(),
            out='out'
        )
        r = op.forward("Message")
        assert r == ["hello", "world", "!@#$%^&*()"]

    def test_op_stream_handles_special_characters(self):
        """Test Op's stream method with input containing special characters."""
        special_input = "hello\nworld\n!@#$%^&*()"
        op = Op(
            DummyAIModel(special_input, proc=[LineParser('data'), NullOut('out', 'data')]), ToText(),
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
            DummyAIModel(partial_input, proc=[LineParser('data'), NullOut('out', 'data')]), ToText(),
            out='out'
        )
        stream = op.stream("Message")
        assert next(stream) == ["line 1"]
        assert next(stream) == ["line 2"]
        assert next(stream) == ["line 3"]

    def test_op_parses_and_returns_line_values(self):

        op = Op(
            DummyAIModel(lines, proc=[LineParser('data'), NullOut('out', 'data')]), ToText(), 
            out='out'
        )
        r = op.forward("Message")
        assert r[0] == 'hi'
        assert r[1] == 'this is'
        assert r[2] == 'some lines'

    def test_op_parses_and_returns_line_values_with_streaming(self):

        op = Op(
            DummyAIModel(lines, proc=[LineParser('data'), NullOut('out', 'data')]), ToText(),
            out='out'
        )
        rs = []
        for r_i in op.stream("Message"):
            rs.extend(r_i)
        rs[0] == 'hi'
        rs[1] == 'this is'
        rs[2] == 'some lines'


class TestThreaded:

    def test_threaded_handles_empty_dialog(self):
        """Test Threaded with an empty dialog to ensure it initializes correctly."""
        threaded = Threaded(
            DummyAIModel("", proc=[NullOut('out', 'content')]),
            router={"user": ToText("user"), "system": ToText("system")},
            out='out'
        )
        result = threaded.forward("user", "")
        assert result == ""

    def test_threaded_handles_large_dialog(self):
        """Test Threaded with a large dialog to ensure it processes correctly."""
        large_dialog = "\n".join([f"line {i}" for i in range(1000)])
        threaded = Threaded(
            DummyAIModel(large_dialog, proc=[LineParser('data'), NullOut('out', 'data')]),
            router={"user": ToText("user"), "system": ToText("system")},
            out='out'
        )
        result = threaded.forward("user", "Message")
        assert len(result) == 1000
        assert result[0] == "line 0"
        assert result[-1] == "line 999"

    def test_threaded_stream_handles_empty_dialog(self):
        """Test Threaded's stream method with an empty dialog."""
        threaded = Threaded(
            DummyAIModel("", proc=[NullOut('out', 'content')]),
            router={"user": ToText("user"), "system": ToText("system")},
            out='out'
        )
        results = list(threaded.stream("user", ""))
        assert results == []

    def test_threaded_stream_handles_large_dialog(self):
        """Test Threaded's stream method with a large dialog."""
        large_dialog = "\n".join([f"line {i}" for i in range(1000)])
        threaded = Threaded(
            DummyAIModel(large_dialog, proc=[LineParser('data'), NullOut('out', 'data')]),
            router={"user": ToText("user"), "system": ToText("system")},
            out='out',
            filter_undefined=True
        )
        results = []
        for val in threaded.stream("user", "Message"):
            results.extend(val)

        assert len(results) == 1000
        assert results[0] == "line 0"
        assert results[-1] == "line 999"

    def test_threaded_handles_special_characters(self):
        """Test Threaded with input containing special characters."""
        special_input = "hello\nworld\n!@#$%^&*()"
        threaded = Threaded(
            DummyAIModel(special_input, proc=[LineParser('data'), NullOut('out', 'data')]),
            router={"user": ToText("user"), "system": ToText("system")},
            out='out'
        )
        result = threaded.forward("user", "Message")
        assert result == ["hello", "world", "!@#$%^&*()"]

    def test_threaded_stream_handles_special_characters(self):
        """Test Threaded's stream method with input containing special characters."""
        special_input = "hello\nworld\n!@#$%^&*()"
        threaded = Threaded(
            DummyAIModel(special_input, proc=[LineParser('data'), NullOut('out', 'data')]),
            router={"user": ToText("user"), "system": ToText("system")},
            out='out'
        )
        results = []
        for res in threaded.stream("user", "Message"):
            results.extend(res)
        assert results == ["hello", "world", "!@#$%^&*()"]

    def test_threaded_handles_multiple_roles(self):
        """Test Threaded with multiple roles in the router."""
        threaded = Threaded(
            DummyAIModel("Hello", proc=[NullOut('out', 'content')]),
            router={"user": ToText("user"), "system": ToText("system")},
            out='out'
        )
        user_result = threaded.forward("user", "Hello")
        system_result = threaded.forward("system", "Hi there")
        assert user_result == "Hello"
        assert system_result == "Hello"

    def test_threaded_stream_handles_partial_streaming(self):
        """Test Threaded's stream method to ensure partial results are returned."""
        partial_input = "line 1\nline 2\nline 3"
        threaded = Threaded(
            DummyAIModel(partial_input, proc=[LineParser('data'), NullOut('out', 'data')]),
            router={"user": ToText("user"), "system": ToText("system")},
            out='out'
        )
        stream = threaded.stream("user", "Message")
        assert next(stream) == ["line 1"]
        assert next(stream) == ["line 2"]
        assert next(stream) == ["line 3"]
