from dachi import graph

from .test_core import DummyNode, DummyNode2


class NodeMethodTester(object):

    @graph.nodemethod(['x'], ['y'])
    def meth1(self, x):
        return x + 1

    @graph.nodemethod(['x'], ['y'])
    def meth2(self, x):
        return x + 2


@graph.nodefunc(['x'], ['y'])
def meth1(x):
    return x + 1

@graph.nodefunc(['x'], ['y'])
def meth2(x):
    return x + 2


class TestNodeMethod:

    def test_node_method_works_as_regular_node(self):

        tester = NodeMethodTester()
        var = graph.Var("X", default=1)

        y1 = tester.meth1.link(var)
        y2 = tester.meth2.link(y1)

        result = y2()
        assert result == 4


class TestNodeFunc:

    def test_node_func_works_as_regular_node(self):

        var = graph.Var("X", default=1)

        y1 = meth1.link(var)
        y2 = meth2.link(y1)

        result = y2()
        assert result == 4


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
