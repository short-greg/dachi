from dachi import graph


class DummyNode(graph.Node):

    def op(self, x: int=2):
        return x + 1


class DummyNode2(graph.Node):

    def op(self, x: int=2):
        return x + 1, x + 2


class DummyNode3(graph.Node):

    def op(self, x: int=2, y: int=3):
        return x + 1, y + 2


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
        assert var.__call__(by={var: 3}) == 3

    def test_clone_returns_var_with_value(self):
        var = graph.Var("X", default=graph.F(lambda: 2))
        var = var.clone()
        assert var.__call__(by={var: 3}) == 3

    def test_incoming_returns_incoming_nodes(self):

        node1 = DummyNode("Node1")
        node2 = DummyNode3("Node2")
        var = graph.Var(
            "X", default=graph.F(lambda: 2)
        )
        var = graph.Var("X", default=1)

        y = node1.link(var)
        out = node2.link(y, var)
        incoming = [inc.val for inc in out.incoming]
        assert y in incoming
        assert var in incoming


class TestNode:

    def test_node_outputs_correct_value_in_chain(self):

        node1 = DummyNode("Node1")
        node2 = DummyNode2("Node2")
        
        var = graph.Var("X", default=1)

        y = node1.link(var)
        y = node2.link(y)
        
        r1, r2 = y.__call__()
        assert r1 == 3
        assert r2 == 4

    def test_node_outputs_correct_value_in_chain_with_no_vars(self):

        node1 = DummyNode("Node1")
        node2 = DummyNode2("Node2")
        y1 = node1()
        r1, r2 = node2.link(y1)
        assert r1.value == 4
        assert r2.value == 5

