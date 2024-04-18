from dachi import process

from .test_core import DummyNode, DummyNode2


class TestOutput:

    def test_output_value_returns_1(self):

        output = process.TOut("X", 1)
        assert output.value == 1

    def test_output_probe_returns_1(self):

        output = process.TOut("X", 1)
        assert output.__call__() == 1

    def test_output_probe_returns_1_after_probe(self):

        output = process.TOut("Y", 1)
        output = output.clone()
        assert output.__call__() == 1


class TestProcess:

    def test_process_value_returns_2(self):

        node = DummyNode("D")
        p = process.TMid(node, [1])

        assert p.value == 2

    def test_process_value_returns_correct_values(self):

        node = DummyNode2("D")
        p = process.TMid(node, [1])

        result = p.value
        assert result[0] == 2
        assert result[1] == 3

    def test_process_value_returns_3_with_var(self):

        var = process.TIn(
            "X", default=2
        )
        node = DummyNode2("D")
        p = process.TMid(node, [var])

        result = p.value
        assert result[0] == 3
        assert result[1] == 4

    def test_process_value_returns_3_with_var_after_clone(self):

        var = process.TIn(
            "X", default=2
        )
        node = DummyNode2("D")
        p = process.TMid(node, [var])
        p = p.clone()

        result = p.value
        assert result[0] == 3
        assert result[1] == 4