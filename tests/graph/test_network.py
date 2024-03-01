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


class TestNetwork:

    def test_node_outputs_correct_value_in_chain(self):

        node1 = DummyNode("Node1")
        node2 = DummyNode2("Node2")
        
        var = graph.Var("X", default=1)

        y = node1.link(var)
        y = node2.link(y)
        
        network = graph.Network(y)

        result = network.exec(by={var: 2})
        print(result)
        assert False
