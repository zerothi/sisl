import pytest

from sisl.nodes import Node, Workflow
from sisl.nodes.context import temporal_context


def test_node():

    @Node.from_func
    def sum_node(a, b):
        return a + b

    with temporal_context(lazy=True):
        val = sum_node(a=2, b=3)
        assert isinstance(val, sum_node)
        assert val.get() == 5

    with temporal_context(lazy=False):
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

    with temporal_context(lazy=True):
        val = sum_node(a=2, b=3)
        assert isinstance(val, sum_node)
        assert val.get() == 6

    with temporal_context(lazy=False):
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
    
    with temporal_context(context=Node.context, lazy=nodes_lazy):
        #It shouldn't matter whether nodes have lazy computation on or off for the working of the workflow
        with temporal_context(context=Workflow.context, lazy=True):
            val = my_workflow(a=2, b=3, c=4)
            assert isinstance(val, my_workflow)
            assert val.get() == 9
        
        with temporal_context(context=Workflow.context, lazy=False):
            val = my_workflow(a=2, b=3, c=4)
            assert val == 9

def test_instance_context():

    @Node.from_func
    def sum_node(a, b):
        return a + b
    
    sum_node.context.update(lazy=True)
    
    # By default, an instance should behave as the class context specifies,
    # so in this case the node should not automatically recalculate
    val = sum_node(a=2, b=3)
    assert isinstance(val, sum_node)
    assert val.get() == 5

    val.update_inputs(a=8)
    assert val._nupdates == 1

    # However, we can set a specific context for the instance.
    val2 = sum_node(a=2, b=3)
    assert isinstance(val2, sum_node)
    assert val2.get() == 5

    val2.context.update(lazy=False)

    val2.update_inputs(a=8)
    assert val2._nupdates == 2

    # And it shouldn't affect the other instance
    val.update_inputs(a=7)
    assert val._nupdates == 1


@pytest.mark.parametrize("lazy_init", [True, False])
def test_default_context(lazy_init):
    """Test that the default context is set correctly for a node class."""

    @Node.from_func
    def calc(val: int):
        return val
    
    @Node.from_func(context={"lazy": False, "lazy_init": lazy_init})
    def alert_change(val: int):
        ...

    val = calc(1)

    init_nupdates = 0 if lazy_init else 1
        
    # We feed the node that produces the intermediate value into our alert node 
    my_alert = alert_change(val=val)

    val.get()
    assert my_alert._nupdates == init_nupdates
    val.update_inputs(val=2)
    assert my_alert._nupdates == init_nupdates + 1