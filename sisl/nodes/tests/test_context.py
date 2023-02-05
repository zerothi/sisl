import pytest

from sisl.nodes import Node, Workflow
from sisl.nodes.context import lazy_context

def test_node():

    @Node.from_func
    def sum_node(a, b):
        return a + b

    with lazy_context(nodes=True):
        val = sum_node(a=2, b=3)
        assert isinstance(val, sum_node)
        assert val.get() == 5

    with lazy_context(nodes=False):
        val = sum_node(a=2, b=3)
        assert val == 5

def test_node_inside_node():
    """When a node class is called inside another node, it should never be lazy in its computation.
    
    That is, calling a node within another node is like calling a function.
    """
    @Node.from_func
    def shift(a):
        return a + 1

    @Node.from_func
    def sum_node(a, b):
        a = shift(a)
        return a + b

    with lazy_context(nodes=True):
        val = sum_node(a=2, b=3)
        assert isinstance(val, sum_node)
        assert val.get() == 6

    with lazy_context(nodes=False):
        val = sum_node(a=2, b=3)
        assert val == 6

@pytest.mark.parametrize("nodes_lazy", [True, False])
def test_workflow(nodes_lazy):
    
    def sum_node(a, b):
        return a + b

    @Workflow.from_func
    def my_workflow(a, b, c):
        first_sum = sum_node(a, b)
        return sum_node(first_sum, c)

    #It shouldn't matter whether nodes have lazy computation on or off for the working of the workflow
    with lazy_context(nodes=nodes_lazy, workflows=True):
        val = my_workflow(a=2, b=3, c=4)
        assert isinstance(val, my_workflow)
        assert val.get() == 9
    
    with lazy_context(nodes=nodes_lazy, workflows=False):
        val = my_workflow(a=2, b=3, c=4)
        assert val == 9
