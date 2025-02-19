# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Optional, Union

import numpy as np

from sisl.utils import direction


def sanitize_axis(ax) -> Union[str, int, np.ndarray]:
    if isinstance(ax, str):
        if re.match("[+-]?[012]", ax):
            ax = ax.replace("0", "a").replace("1", "b").replace("2", "c")
        ax = ax.lower().replace("+", "")
    elif isinstance(ax, int):
        ax = "abc"[ax]
    elif isinstance(ax, (list, tuple)):
        ax = np.array(ax)

    # Now perform some checks
    invalid = True
    if isinstance(ax, str):
        invalid = not re.match("-?[xyzabc]", ax)
    elif isinstance(ax, np.ndarray):
        invalid = ax.shape != (3,)

    if invalid:
        raise ValueError(
            f"Incorrect axis passed. Axes must be one of [+-]('x', 'y', 'z', 'a', 'b', 'c', '0', '1', '2', 0, 1, 2)"
            + " or a numpy array/list/tuple of shape (3, )"
        )

    return ax


def sanitize_axes(
    val: Union[str, Sequence[Union[str, int, np.ndarray]]],
) -> list[Union[str, int, np.ndarray]]:
    if isinstance(val, str):
        val = re.findall("[+-]?[xyzabc012]", val)
    return [sanitize_axis(ax) for ax in val]


def get_ax_title(ax: Union[Axis, Callable], cartesian_units: str = "Ang") -> str:
    """Generates the title for a given axis"""
    if hasattr(ax, "__name__"):
        title = ax.__name__
    elif isinstance(ax, np.ndarray) and ax.shape == (3,):
        title = str(ax)
    elif not isinstance(ax, str):
        title = ""
    elif re.match("[+-]?[xXyYzZ]", ax):
        title = f"{ax.upper()} axis [{cartesian_units}]"
    elif re.match("[+-]?[aAbBcC]", ax):
        title = f"{ax.upper()} lattice vector"
    else:
        title = ax

    return title


def axis_direction(
    ax: Axis, cell: Optional[Union[npt.ArrayLike, Lattice]] = None
) -> npt.NDArray[np.float64]:
    """Returns the vector direction of a given axis.

    Parameters
    ----------
    ax: Axis
        Axis specification for which you want the direction. It supports
        negative signs (e.g. "-x"), which will invert the direction.
    cell: array-like of shape (3, 3) or Lattice, optional
        The cell of the structure, only needed if lattice vectors {"a", "b", "c"}
        are provided for `ax`.

    Returns
    ----------
    np.ndarray of shape (3, )
        The direction of the axis.
    """
    if isinstance(ax, (int, str)):
        sign = 1
        # If the axis contains a -, we need to mirror the direction.
        if isinstance(ax, str) and ax[0] == "-":
            sign = -1
            ax = ax[1]
        ax = sign * direction(ax, abc=cell, xyz=np.diag([1.0, 1.0, 1.0]))

    return ax


def axes_cross_product(
    v1: Axis, v2: Axis, cell: Optional[Union[npt.ArrayLike, Lattice]] = None
):
    """An enhanced version of the cross product.

    It is an enhanced version because both vectors accept strings that represent
    the cartesian axes or the lattice vectors (see `v1`, `v2` below). It has been built
    so that cross product between lattice vectors (-){"a", "b", "c"} follows the same rules
    as (-){"x", "y", "z"}

    Parameters
    ----------
    v1, v2: array-like of shape (3,) or (-){"x", "y", "z", "a", "b", "c"}
        The vectors to take the cross product of.
    cell: array-like of shape (3, 3)
        The cell of the structure, only needed if lattice vectors {"a", "b", "c"}
        are passed for `v1` and `v2`.
    """
    # Make abc follow the same rules as xyz to find the orthogonal direction
    # That is, a X b = c; -a X b = -c and so on.
    if isinstance(v1, str) and isinstance(v2, str):
        if re.match("([+-]?[abc]){2}", v1 + v2):
            v1 = v1.replace("a", "x").replace("b", "y").replace("c", "z")
            v2 = v2.replace("a", "x").replace("b", "y").replace("c", "z")
            ort = axes_cross_product(v1, v2)
            ort_ax = "abc"[np.where(ort != 0)[0][0]]
            if ort.sum() == -1:
                ort_ax = "-" + ort_ax
            return axis_direction(ort_ax, cell)

    # If the vectors are not abc, we just need to take the cross product.
    return np.cross(axis_direction(v1, cell), axis_direction(v2, cell))
