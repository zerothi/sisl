import operator
from typing import Any, Dict

from .node import Node


class SyntaxNode(Node): ...


class ListSyntaxNode(SyntaxNode):
    @staticmethod
    def function(*items):
        return list(items)


class TupleSyntaxNode(SyntaxNode):
    @staticmethod
    def function(*items):
        return tuple(items)


class DictSyntaxNode(SyntaxNode):
    @staticmethod
    def function(**items):
        return items


class ConditionalExpressionSyntaxNode(SyntaxNode):
    _outdate_due_to_inputs: bool = False

    def _get_evaluated_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the inputs of this node.

        This function overwrites the default implementation in Node, because
        we want to evaluate only the path that we are going to take.

        Parameters
        ----------
        inputs : dict
            The inputs to this node.
        """

        evaluated = {}

        # Get the state of the test input, which determines the path that we are going to take.
        evaluated["test"] = (
            self.evaluate_input_node(inputs["test"])
            if isinstance(inputs["test"], Node)
            else inputs["test"]
        )

        # Evaluate only the path that we are going to take.
        if evaluated["test"]:
            evaluated["true"] = (
                self.evaluate_input_node(inputs["true"])
                if isinstance(inputs["true"], Node)
                else inputs["true"]
            )
            evaluated["false"] = self._prev_evaluated_inputs.get("false")
        else:
            evaluated["false"] = (
                self.evaluate_input_node(inputs["false"])
                if isinstance(inputs["false"], Node)
                else inputs["false"]
            )
            evaluated["true"] = self._prev_evaluated_inputs.get("true")

        return evaluated

    def update_inputs(self, **inputs):
        # This is just a wrapper over the normal update_inputs, which makes
        # sure that the node is only marked as outdated if the input that
        # is being used has changed. Note that here we just create a flag,
        # which is then used in _receive_outdated. (_receive_outdated is
        # called by super().update_inputs())
        current_test = self._prev_evaluated_inputs["test"]

        self._outdate_due_to_inputs = len(inputs) > 0
        if "test" not in inputs:
            if current_test and ("true" not in inputs):
                self._outdate_due_to_inputs = False
            elif not current_test and ("false" not in inputs):
                self._outdate_due_to_inputs = False

        try:
            super().update_inputs(**inputs)
        except:
            self._outdate_due_to_inputs = False
            raise

    def _receive_outdated(self):
        # Relevant inputs have been updated, mark this node as outdated.
        if self._outdate_due_to_inputs:
            return super()._receive_outdated()

        # We avoid marking this node as outdated if the outdated input
        # is not the one being returned.
        for k in self._input_nodes:
            if self._input_nodes[k]._outdated:
                if k == "test":
                    return super()._receive_outdated()
                elif k == "true":
                    if self._prev_evaluated_inputs["test"]:
                        return super()._receive_outdated()
                elif k == "false":
                    if not self._prev_evaluated_inputs["test"]:
                        return super()._receive_outdated()

    @staticmethod
    def function(test, true, false):
        return true if test else false

    def get_diagram_label(self):
        """Returns the label to be used in diagrams when displaying this node."""
        return "if/else"


class CompareSyntaxNode(SyntaxNode):
    _op_to_symbol = {
        "eq": "==",
        "ne": "!=",
        "gt": ">",
        "lt": "<",
        "ge": ">=",
        "le": "<=",
        None: "compare",
    }

    @staticmethod
    def function(left, op: str, right):
        return getattr(operator, op)(left, right)

    def get_diagram_label(self):
        """Returns the label to be used in diagrams when displaying this node."""
        return self._op_to_symbol.get(self._prev_evaluated_inputs.get("op"))
