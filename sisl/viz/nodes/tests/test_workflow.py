from sisl.viz.nodes.node import Node
from sisl.viz.nodes.workflow import Workflow

import pytest

@pytest.fixture(scope='module', params=["from_func", "explicit_class"])
def triple_sum(request):
    """Returns a workflow that computes a triple sum.
    
    The workflow might have been obtained in different ways, but they all
    should be equivalent in functionality.
    """

    @Node.from_func
    def sum(a, b):
        return a + b
            
    if request.param == "from_func":
        # A triple sum 
        @Workflow.from_func
        def triple_sum(a, b, c):
            first_sum = sum(a, b)
            return sum(first_sum, c)
    else:
        class triple_sum(Workflow):

            @staticmethod
            def _workflow(a, b, c):
                first_sum = sum(a, b)
                return sum(first_sum, c)

    return triple_sum

def test_linked_inputs(triple_sum):

    linked_inputs = triple_sum._linked_inputs
    
    assert "sum" in linked_inputs
    assert linked_inputs["sum"].get("a") == "a"
    assert linked_inputs["sum"].get("b") == "b"

    assert "sum_1" in linked_inputs
    assert "a" not in linked_inputs["sum_1"]
    assert linked_inputs["sum_1"].get("b") == "c"

def test_init(triple_sum):
    triple_sum(a=2, b=3, c=5)

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

    assert val._nodes['sum']._nupdates == 1
    assert val._nodes['sum_1']._nupdates == 2

def test_positional_arguments(triple_sum):

    val = triple_sum(2, 3, 5)

    assert val.get() == 10