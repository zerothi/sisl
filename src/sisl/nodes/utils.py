# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from typing import Sequence, Callable, Any

from .node import Node

def traverse_tree_forward(roots: Sequence[Node],  func: Callable[[Node], Any]) -> None:
    """Traverse a tree of nodes in a forward fashion.

    Parameters
    ----------
    roots : Sequence[Node]
        The roots of the tree to traverse.
    func : Callable[[Node], Any]
        The function to apply to each node in the tree.
    """
    for root in roots:
        func(root)
        traverse_tree_forward(root._output_links, func)

def traverse_tree_backward(leaves: Sequence[Node],  func: Callable[[Node], Any]) -> None:
    """Traverse a tree of nodes in a backwards fashion.

    Parameters
    ----------
    leaves : Sequence[Node]
        The leaves of the tree to traverse.
    func : Callable[[Node], Any]
        The function to apply to each node in the tree.
    """
    for leaf in leaves:
        func(leaf)
        leaf.map_inputs(
            leaf.inputs, 
            func=lambda node: traverse_tree_backward((node, ), func=func),
            only_nodes=True
        )

def visit_all_connected(nodes: Sequence[Node], func: Callable[[Node], Any], _seen_nodes=None) -> None:
    """Visit all nodes that are connected to a list of nodes.

    Parameters
    ----------
    nodes : Sequence[Node]
        The nodes to traverse.
    func : Callable[[Node], Any]
        The function to apply to each node in the tree.
    """
    if _seen_nodes is None:
        _seen_nodes = []

    for node in nodes:
        if node in _seen_nodes:
            continue

        _seen_nodes.append(id(node))
        traverse_tree_forward((node, ), func=lambda node: visit_all_connected((node, ), func=func, _seen_nodes=_seen_nodes))
        traverse_tree_backward((node, ), func=lambda node: visit_all_connected((node, ), func=func, _seen_nodes=_seen_nodes))