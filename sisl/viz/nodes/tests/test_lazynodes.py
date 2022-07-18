import pytest

from sisl.viz.nodes import lazy_context
from sisl.viz.nodes.node import GetItemNode, Node


@pytest.fixture(scope='module', params=["explicit_class", "from_func"])
def sum_node(request):
    if request.param == "explicit_class":
        class SumNode(Node):
            def __init__(self, input1, input2, **kwargs):
                super().__init__(input1=input1, input2=input2, **kwargs)

            def _get(self, input1, input2):
                return input1 + input2
    else:
        @Node.from_func
        def SumNode(input1, input2):
            return input1 + input2

    return SumNode

@lazy_context(nodes=True)
def test_node_runs(sum_node):
    node = sum_node(1,2)
    res = node.get()
    assert res == 3

@lazy_context(nodes=True)
def test_node_not_updated(sum_node):
    """Checks that the node only runs when it needs to."""
    node = sum_node(1,2)
    res = node.get()
    assert res == 3
    assert node._nupdates == 1

    res = node.get()
    assert res == 3
    assert node._nupdates == 1

@lazy_context(nodes=True)
def test_node_links():

    @Node.from_func
    def my_node(a: int = 2):
        return a

    node1 = my_node()
    node2 = my_node(a=node1)

    # Check that node1 knows its output is being used by
    # node2
    assert len(node1._output_links) == 1
    assert node1._output_links[0] is node2

    # And that node2 knows it's using node1 as an input.
    assert len(node2._input_nodes) == 1
    assert 'a' in node2._input_nodes
    assert node2._input_nodes['a'] is node1

    # Now check that if we update node2, the connections
    # will be removed.
    node2.update_inputs(a="other value")

    assert len(node1._output_links) == 0
    assert len(node2._input_nodes) == 0

    # Finally check that connections are properly built when
    # updating inputs with a value containing a node.
    node3 = my_node()
    node2.update_inputs(a=node3)

    # Check that node3 knows its output is being used by
    # node2
    assert len(node3._output_links) == 1
    assert node3._output_links[0] is node2

    # And that node2 knows it's using node3 as an input.
    assert len(node2._input_nodes) == 1
    assert 'a' in node2._input_nodes
    assert node2._input_nodes['a'] is node3

@lazy_context(nodes=True)
def test_node_tree(sum_node):
    node1 = sum_node(1,2)
    node2 = sum_node(node1, 3)

    res = node2.get()
    assert res == 6
    assert node1._nupdates == 1

    node2.update_inputs(input1=node1, input2=4)

    res = node2.get()
    assert res == 7
    assert node1._nupdates == 1

    node1.update_inputs(input1=5, input2=6)

    res = node2.get()
    assert res == 15
    assert node1._nupdates == 2

@lazy_context(nodes=True)
def test_automatic_recalculation(sum_node):

    # Set the first node automatic recalculation on
    node1 = sum_node(1, 2, automatic_recalc=True)
    node2 = sum_node(node1, 3)

    node2.get()

    assert node1._output == 3
    node1.update_inputs(input1=1, input2=3)
    assert node1._output == 4

    # However, node2 should not be automatically recalculated
    assert node2._output == 6

    # If we now get the result, it should get recalculated
    assert node2.get() == 7
    assert node2._output == 7

    # We now check the opposite. Node2 will have automatic recalculation,
    # so everything should get updated when we update node1, if the outdated
    # signal is properly transferred
    node1.update_inputs(automatic_recalc=False)
    node2.update_inputs(automatic_recalc=True)
    node2.get()

    assert node1._output == 4
    node1.update_inputs(input1=1, input2=4)
    assert node1._output == 5
    assert node2._output == 8


def test_getitem():

    @Node.from_func
    def some_tuple():
        return (3, 4)
    
    with lazy_context(nodes=True):
        my_tuple = some_tuple()

    val = my_tuple[0]

    assert val == 3

    with lazy_context(nodes=True):
        val = my_tuple[0]
    
    assert isinstance(val, GetItemNode)
    assert val.get() == 3

@lazy_context(nodes=True)
def test_args():
    """Checks that functions with *args are correctly handled by Node."""

    @Node.from_func
    def reduce_(*nums, factor: int = 1):

        val = 0
        for num in nums:
            val += num
        return val * factor

    val = reduce_(1, 2, 3, factor=2)

    assert val.get() == 12

    val2 = reduce_(val, 4, factor=1)

    assert val2.get() == 16
    assert val._nupdates == 1