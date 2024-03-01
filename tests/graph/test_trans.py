from dachi import graph

from .test_core import DummyNode, DummyNode2


class TestOutput:

    def test_output_value_returns_1(self):

        output = graph.Output("X", 1)
        assert output.value == 1

    def test_output_probe_returns_1(self):

        output = graph.Output("X", 1)
        assert output.__call__() == 1

    def test_output_probe_returns_1_after_probe(self):

        output = graph.Output("Y", 1)
        output = output.clone()
        assert output.__call__() == 1


class TestProcess:

    def test_process_value_returns_1(self):

        node = DummyNode("D")
        process = graph.Process(node, [1])

        assert process.value == 2

    def test_process_value_returns_1(self):

        node = DummyNode2("D")
        process = graph.Process(node, [1])

        result = process.value
        assert result[0] == 2
        assert result[1] == 3

    def test_process_value_returns_3_with_var(self):

        var = graph.Var(
            "X", default=2
        )
        node = DummyNode2("D")
        process = graph.Process(node, [var])

        result = process.value
        assert result[0] == 3
        assert result[1] == 4

    def test_process_value_returns_3_with_var_after_clone(self):

        var = graph.Var(
            "X", default=2
        )
        node = DummyNode2("D")
        process = graph.Process(node, [var])
        process = process.clone()

        result = process.value
        assert result[0] == 3
        assert result[1] == 4