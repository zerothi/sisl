from typing import Literal

from ..node import Node

@Node.from_func
def get_axis_var(axis: Literal["x", "y"], var: str, var_axis: Literal["x", "y"], other_var: str) -> str:
    """Very simple node to use in workflows for determining what goes on each axis."""
    if axis == var_axis:
        return var
    else:
        return other_var