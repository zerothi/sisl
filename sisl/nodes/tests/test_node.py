from sisl.nodes import Node

def test_node_classes_reused():
    def a():
        pass

    x = Node.from_func(a)
    y = Node.from_func(a)

    assert x is y