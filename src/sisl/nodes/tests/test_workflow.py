from typing import Type

import pytest

from sisl.nodes import Node, Workflow
from sisl.nodes.utils import traverse_tree_forward


@pytest.fixture(scope='module', params=["from_func", "explicit_class", "input_operations"])
def triple_sum(request) -> Type[Workflow]:
    """Returns a workflow that computes a triple sum.
    
    The workflow might have been obtained in different ways, but they all
    should be equivalent in functionality.
    """

    def my_sum(a, b):
        return a + b
            
    if request.param == "from_func":
        # A triple sum 
        @Workflow.from_func
        def triple_sum(a, b, c):
            first_sum = my_sum(a, b)
            return my_sum(first_sum, c)
        
        triple_sum._sum_key = "my_sum"
    elif request.param == "explicit_class":
        class triple_sum(Workflow):

            @staticmethod
            def function(a, b, c):
                first_sum = my_sum(a, b)
                return my_sum(first_sum, c)
        
        triple_sum._sum_key = "my_sum"
    elif request.param == "input_operations":
        @Workflow.from_func
        def triple_sum(a, b, c):
            first_sum = a + b
            return first_sum + c

        triple_sum._sum_key = "UfuncNode"

    return triple_sum

def test_named_vars(triple_sum):
    # Check that the first_sum variable has been detected correctly.
    assert set(triple_sum.dryrun_nodes.named_vars) == {'first_sum'}

    # And check that it maps to the correct node.
    assert triple_sum.dryrun_nodes.first_sum is triple_sum.dryrun_nodes.workers[triple_sum._sum_key]

def test_workflow_instantiation(triple_sum):
    # Create an instance of the workflow.
    flow = triple_sum(2, 3, 5)

    # Check that the workflow nodes have been instantiated.
    assert hasattr(flow, 'nodes')
    for k, wf_node in flow.dryrun_nodes.items():
        assert k in flow.nodes._all_nodes
        new_node = flow.nodes[k]
        assert wf_node is not new_node

    # Traverse all the workflow and make sure that there are no references
    # to the workflow dryrun nodes.
    old_ids = [id(n) for n in flow.dryrun_nodes.values()]

    def check_not_old_id(node):
        assert id(node) not in old_ids

    traverse_tree_forward(flow.nodes.inputs.values(), check_not_old_id)

def test_right_result(triple_sum):
    assert triple_sum(a=2, b=3, c=5).get() == 10

def test_updatable_inputs(triple_sum):

    val = triple_sum(a=2, b=3, c=5)
    
    assert val.get() == 10

    val.update_inputs(b=4)

    assert val.get() == 11

def test_recalc_necessary_only(triple_sum):

    val = triple_sum(a=2, b=3, c=5)
    
    assert val.get() == 10

    val.update_inputs(c=4)

    assert val.get() == 9

    assert val.nodes[triple_sum._sum_key]._nupdates == 1
    assert val.nodes[f'{triple_sum._sum_key}_1']._nupdates == 2

def test_positional_arguments(triple_sum):

    val = triple_sum(2, 3, 5)

    assert val.get() == 10

# *args and **kwargs are not supported for now in workflows. 
# def test_kwargs_not_overriden():

#     @Node.from_func
#     def some_node(**kwargs):
#         return kwargs

#     @Workflow.from_func
#     def some_workflow(**kwargs):
#         return some_node(a=2, b=3, **kwargs)
    
#     # Here we check that passing **kwargs to the node inside the workflow
#     # does not interfere with the other keyword arguments that are explicitly
#     # passed to the node (and accepted by the node as **kwargs)
#     assert some_workflow().get() == {'a': 2, 'b': 3}
#     assert some_workflow(c=4).get() == {'a': 2, 'b': 3, 'c': 4}

def test_args_nodes_registered():

    def some_node(*args):
        return args

    @Workflow.from_func
    def some_workflow():
        a = some_node(1, 2, 3)
        return some_node(2, a, 4)
    
    # Check that the workflow knows about the first instanced node.
    wf = some_workflow()
    assert len(wf.nodes.workers) == 2

def test_kwargs_nodes_registered():

    def some_node(**kwargs):
        return kwargs

    @Workflow.from_func
    def some_workflow():
        a = some_node(a=1, b=2, c=3)
        return some_node(b=2, a=a, c=4)
    
    # Check that the workflow knows about the first instanced node.
    wf = some_workflow()
    assert len(wf.nodes.workers) == 2

def test_workflow_inside_workflow(triple_sum):

    def multiply(a, b):
        return a * b

    @Workflow.from_func
    def some_multiplication(a, b, c, d, e, f):
        """ Workflow that computes (a + b + c) * (d + e + f)"""
        return multiply(triple_sum(a,b,c), triple_sum(d, e, f))

    val = some_multiplication(1, 2, 3, 1, 2, 1)

    assert val.get() == (1 + 2 + 3) * (1 + 2 + 1)

    first_triple_sum = val.nodes['triple_sum']

    assert first_triple_sum.nodes[triple_sum._sum_key]._nupdates == 1
    assert first_triple_sum.nodes[f'{triple_sum._sum_key}_1']._nupdates == 1

    val.update_inputs(c=2)

    assert first_triple_sum.nodes[triple_sum._sum_key]._nupdates == 1
    assert first_triple_sum.nodes[f'{triple_sum._sum_key}_1']._nupdates == 1

    assert val.get() == (1 + 2 + 2) * (1 + 2 + 1)

    assert first_triple_sum.nodes[triple_sum._sum_key]._nupdates == 1
    assert first_triple_sum.nodes[f'{triple_sum._sum_key}_1']._nupdates == 2