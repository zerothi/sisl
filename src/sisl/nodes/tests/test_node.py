import pytest

from sisl.nodes import Node, temporal_context
from sisl.nodes.node import GetItemNode


@pytest.fixture(scope='module', params=["explicit_class", "from_func"])
def sum_node(request):
    if request.param == "explicit_class":
        class SumNode(Node):
            @staticmethod
            def function(input1, input2):
                return input1 + input2
    else:
        @Node.from_func
        def SumNode(input1, input2):
            return input1 + input2

    return SumNode

def test_node_classes_reused():
    def a():
        pass

    x = Node.from_func(a)
    y = Node.from_func(a)

    assert x is y

def test_node_runs(sum_node):
    node = sum_node(1,2)
    res = node.get()
    assert res == 3

@temporal_context(lazy=False)
def test_nonlazy_node(sum_node):
    node = sum_node(1,2)
    
    assert node._nupdates == 1
    assert node._output == 3

@temporal_context(lazy=True)
def test_node_not_updated(sum_node):
    """Checks that the node only runs when it needs to."""
    node = sum_node(1,2)

    assert node._nupdates == 0

    res = node.get()
    assert res == 3
    assert node._nupdates == 1

    res = node.get()
    assert res == 3
    assert node._nupdates == 1

@temporal_context(lazy=True)
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

@temporal_context(lazy=True)
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

def test_automatic_recalculation(sum_node):

    # Set the first node automatic recalculation on
    node1 = sum_node(1, 2)
    assert node1._nupdates == 0

    node1.context.update(lazy=False)

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
    node1.context.update(lazy=True)
    node2.context.update(lazy=False)
    node2.get()

    assert node1._output == 4
    node1.update_inputs(input1=1, input2=4)
    assert node1._output == 5
    assert node2._output == 8

def test_getitem():

    @Node.from_func
    def some_tuple():
        return (3, 4)
    
    my_tuple = some_tuple()

    val = my_tuple.get()[0]
    assert val == 3

    item = my_tuple[0]
    
    assert isinstance(item, GetItemNode)
    assert item.get() == 3

@temporal_context(lazy=True)
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

@temporal_context(lazy=True)
def test_node_links_args():

    @Node.from_func
    def my_node(*some_args):
        return some_args

    node1 = my_node()
    node2 = my_node(4, 1, node1)

    # Check that node1 knows that node2 uses its output
    assert len(node1._output_links) == 1
    assert node1._output_links[0] is node2

    # And that node2 knows it's using node1 as an input.
    assert len(node2._input_nodes) == 1
    assert 'some_args[2]' in node2._input_nodes
    assert node2._input_nodes['some_args[2]'] is node1


@temporal_context(lazy=True)
def test_kwargs():
    """Checks that functions with **kwargs are correctly handled by Node."""

    @Node.from_func
    def my_dict(**some_kwargs):
        return some_kwargs

    val = my_dict(a=2, b=4)

    assert val.get() == {"a": 2, "b": 4}

    val2 = my_dict(old=val)

    assert val2.get() == {"old": {"a": 2, "b": 4}}

@temporal_context(lazy=True)
def test_update_kwargs():
    """Checks that functions with **kwargs are correctly handled by Node."""

    @Node.from_func
    def my_dict(**some_kwargs):
        return some_kwargs

    val = my_dict(a=2, b=4)
    assert val.get() == {"a": 2, "b": 4}

    val.update_inputs(a=3)
    assert val.get() == {"a": 3, "b": 4}

    val.update_inputs(c=5)
    assert val.get() == {"a": 3, "b": 4, "c": 5}

    val.update_inputs(a=Node.DELETE_KWARG)
    assert val.get() == {"b": 4, "c": 5}

@temporal_context(lazy=True)
def test_node_links_kwargs():

    @Node.from_func
    def my_node(**some_kwargs):
        return some_kwargs

    node1 = my_node()
    node2 = my_node(a=node1)

    # Check that node1 knows its output is being used by
    # node2
    assert len(node1._output_links) == 1
    assert node1._output_links[0] is node2

    # And that node2 knows it's using node1 as an input.
    assert len(node2._input_nodes) == 1
    assert 'some_kwargs[a]' in node2._input_nodes
    assert node2._input_nodes['some_kwargs[a]'] is node1

    # Test that kwargs that no longer exist are delinked.

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
    assert 'some_kwargs[a]' in node2._input_nodes
    assert node2._input_nodes['some_kwargs[a]'] is node3

def test_ufunc(sum_node):

    node = sum_node(1, 3)

    assert node.get() == 4

    node2 = node + 6

    assert node2.get() == 10