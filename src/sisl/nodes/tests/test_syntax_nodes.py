from sisl.nodes.node import ConstantNode
from sisl.nodes.syntax_nodes import (
    CompareSyntaxNode,
    ConditionalExpressionSyntaxNode,
    DictSyntaxNode,
    ListSyntaxNode,
    TupleSyntaxNode,
)
from sisl.nodes.workflow import Workflow


def test_list_syntax_node():
    assert ListSyntaxNode("a", "b", "c").get() == ["a", "b", "c"]


def test_tuple_syntax_node():
    assert TupleSyntaxNode("a", "b", "c").get() == ("a", "b", "c")


def test_dict_syntax_node():
    assert DictSyntaxNode(a="b", c="d", e="f").get() == {"a": "b", "c": "d", "e": "f"}


def test_cond_expr_node():
    node = ConditionalExpressionSyntaxNode(test=True, true=1, false=2)

    assert node.get() == 1
    node.update_inputs(test=False)

    assert node._outdated
    assert node.get() == 2

    node.update_inputs(true=3)
    assert not node._outdated

    # Check that only one path is evaluated.
    input1 = ConstantNode(1)
    input2 = ConstantNode(2)

    node = ConditionalExpressionSyntaxNode(test=True, true=input1, false=input2)

    assert node.get() == 1
    assert input1._nupdates == 1
    assert input2._nupdates == 0


def test_compare_syntax_node():
    assert CompareSyntaxNode(1, "eq", 2).get() == False
    assert CompareSyntaxNode(1, "ne", 2).get() == True
    assert CompareSyntaxNode(1, "gt", 2).get() == False
    assert CompareSyntaxNode(1, "lt", 2).get() == True
    assert CompareSyntaxNode(1, "ge", 2).get() == False
    assert CompareSyntaxNode(1, "le", 2).get() == True


def test_workflow_with_syntax():
    def f(a):
        return [a]

    assert Workflow.from_func(f)(2).get() == [2]

    def f(a):
        return (a,)

    assert Workflow.from_func(f)(2).get() == (2,)

    def f(a):
        return {"a": a}

    assert Workflow.from_func(f)(2).get() == {"a": 2}

    def f(a, b, c):
        return b if a else c

    assert Workflow.from_func(f)(False, 1, 2).get() == 2

    def f(a, b):
        return a != b

    assert Workflow.from_func(f)(1, 2).get() == True
