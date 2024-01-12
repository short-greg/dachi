from dachi import graph


class DummyNode(graph.Node):

    def op(self, x: int=2):
        return x + 1


class DummyNode2(graph.Node):

    def op(self, x: int=2):
        return x + 1, x + 2


class TestVar:

    def test_var_returns_default_when_not_in_by(self):
        var = graph.Var(
            "X", default=2
        )
        assert var.value == 2

    def test_var_returns_result_of_function(self):
        var = graph.Var(
            "X", default=graph.F(lambda: 2)
        )
        assert var.value == 2

    def test_var_returns_result_of_by_if_defined(self):
        var = graph.Var(
            "X", default=graph.F(lambda: 2)
        )
        assert var.probe(by={var: 3}) == 3

    def test_clone_returns_var_with_value(self):
        var = graph.Var("X", default=graph.F(lambda: 2))
        var = var.clone()
        assert var.probe(by={var: 3}) == 3


class TestOutput:

    def test_output_value_returns_1(self):

        output = graph.Output("X", 1)
        assert output.value == 1

    def test_output_probe_returns_1(self):

        output = graph.Output("X", 1)
        assert output.probe() == 1

    def test_output_probe_returns_1_after_probe(self):

        output = graph.Output("Y", 1)
        output = output.clone()
        assert output.probe() == 1


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


class TestNode:

    def test_node_outputs_correct_value_in_chain(self):

        node1 = DummyNode("Node1")
        node2 = DummyNode2("Node2")
        
        var = graph.Var("X", default=1)

        y = node1.link(var)
        y = node2.link(y)
        
        r1, r2 = y.probe()
        assert r1 == 3
        assert r2 == 4

    def test_node_outputs_correct_value_in_chain_with_no_vars(self):

        node1 = DummyNode("Node1")
        node2 = DummyNode2("Node2")
        y1 = node1()
        r1, r2 = node2.link(y1)
        assert r1.value == 4
        assert r2.value == 5


class TestAdapter:

    def test_adapter_outputs_correct_value_in_chain(self):

        node1 = DummyNode("Node1")
        node2 = DummyNode2("Node2")
        
        var = graph.Var("X", default=1)

        y1 = node1.link(var)
        y2 = node2.link(y1)
        adapter = graph.Adapter(
            "Adapt", [var], [y1, (y2, 0)]
        )
        
        r1, r2 = adapter()
        assert r1 == 2
        assert r2 == 3
