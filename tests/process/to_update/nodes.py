from dachi import process

from ..depracated.core import DummyNode, DummyNode2


class NodeMethodTester(object):

    @process.nodedef(['x'], ['y'])
    def meth1(self, x):
        return x + 1

    @process.nodedef(['x'], ['y'])
    def meth2(self, x):
        return x + 2


@process.nodedef(['x'], ['y'])
def meth1(x):
    return x + 1

@process.nodedef(['x'], ['y'])
def meth2(x):
    return x + 2


class TestNodeFunc:

    def test_node_func_works_as_regular_node(self):

        var = process.TIn("X", default=1)

        y1 = process.linkf(meth1, var)
        y2 = process.linkf(meth2, y1)

        result = y2()
        assert result == 4


class TestAdapter:

    def test_adapter_outputs_correct_value_in_chain(self):

        node1 = DummyNode("Node1")
        node2 = DummyNode2("Node2")
        
        var = process.TIn("X", default=1)

        y1 = node1.link(var)
        y2 = node2.link(y1)
        adapter = process.Adapter(
            "Adapt", [var], [y1, (y2, 0)]
        )
        r1, r2 = adapter()
        assert r1 == 2
        assert r2 == 3
