from dachi import process
from ..depracated.core import DummyNode, DummyNode2


class TestNetwork:

    def test_network_outputs_correct_value_in_chain(self):

        node1 = DummyNode("Node1")
        node2 = DummyNode2("Node2")
        
        var = process.TIn("X", default=1)

        y = node1.link(var)
        y = node2.link(y)
        
        network = process.Network((y, 0))

        result = network.exec(by={var: 2})
        assert result == 4

    def test_node_outputs_correct_value_with_two_outputs(self):

        node1 = DummyNode("Node1")
        node2 = DummyNode2("Node2")
        
        var = process.TIn("X", default=1)

        y1 = node1.link(var)
        y = node2.link(y1)
        
        network = process.Network([(y, 0), y1])

        result = network.exec(by={var: 2})
        assert result[0] == 4
        assert result[1] == 3


    def test_node_outputs_correct_value_with_two_outputs(self):

        node1 = DummyNode("Node1")
        node2 = DummyNode2("Node2")
        
        var = process.TIn("X", default=1)

        y = node2.link(var)
        y = node1.link(y[0])
        
        network = process.Network(y)

        result = network.exec(by={var: 2})
        assert result == 4
