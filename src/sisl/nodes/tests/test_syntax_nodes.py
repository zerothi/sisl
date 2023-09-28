from sisl.nodes.syntax_nodes import DictSyntaxNode, ListSyntaxNode, TupleSyntaxNode
from sisl.nodes.workflow import Workflow


def test_list_syntax_node():
    assert ListSyntaxNode("a", "b", "c").get() == ["a", "b", "c"]

def test_tuple_syntax_node():
    assert TupleSyntaxNode("a", "b", "c").get() == ("a", "b", "c")

def test_dict_syntax_node():
    assert DictSyntaxNode(a="b", c="d", e="f").get() == {"a": "b", "c": "d", "e": "f"}

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
