import pytest

from sisl.nodes import Node
from sisl.nodes.utils import (
    StopTraverse,
    traverse_tree_backward,
    traverse_tree_forward,
    visit_all_connected,
)


@pytest.fixture(scope="module")
def sum_node():
    @Node.from_func
    def sum(a, b):
        return a + b

    return sum


def test_traverse_tree_forward(sum_node):
    initial = sum_node(0, 1)
    second = sum_node(initial, 2)
    final = sum_node(second, 3)

    i = 0

    def count(node):
        nonlocal i
        i += 1

    traverse_tree_forward((final,), func=count)
    assert i == 1

    i = 0
    traverse_tree_forward((second,), func=count)
    assert i == 2

    i = 0
    traverse_tree_forward((initial,), func=count)
    assert i == 3

    def only_first(node):
        nonlocal i
        i += 1
        raise StopTraverse

    i = 0
    traverse_tree_forward((initial,), func=only_first)
    assert i == 1


def test_traverse_tree_backward(sum_node):
    initial = sum_node(0, 1)
    second = sum_node(initial, 2)
    final = sum_node(second, 3)

    i = 0

    def count(node):
        nonlocal i
        i += 1

    traverse_tree_backward((final,), func=count)
    assert i == 3

    i = 0
    traverse_tree_backward((second,), func=count)
    assert i == 2

    i = 0
    traverse_tree_backward((initial,), func=count)
    assert i == 1

    def only_first(node):
        nonlocal i
        i += 1
        raise StopTraverse

    i = 0
    traverse_tree_backward((final,), func=only_first)
    assert i == 1


def test_visit_all_connected(sum_node):
    initial = sum_node(0, 1)
    second = sum_node(initial, 2)
    final = sum_node(second, 3)

    i = 0

    def count(node):
        nonlocal i
        i += 1

    visit_all_connected((initial,), func=count)
    assert i == 3

    i = 0
    visit_all_connected((second,), func=count)
    assert i == 3

    i = 0
    visit_all_connected((final,), func=count)
    assert i == 3

    def only_first(node):
        nonlocal i
        i += 1
        raise StopTraverse

    i = 0
    visit_all_connected((final,), func=only_first)
    assert i == 1
