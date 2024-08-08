# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# TODO when forward refs work with annotations
# from __future__ import annotations

from typing import Any, Literal, TypedDict

import numpy as np
from xarray import Dataset

from sisl._core.lattice import Lattice
from sisl.typing import LatticeLike

from .coords import CoordsDataset

CellDataset = CoordsDataset


def is_cartesian_unordered(cell: LatticeLike, tol: float = 1e-3) -> bool:
    """Whether a cell has cartesian axes as lattice vectors, regardless of their order.

    Parameters
    -----------
    cell: np.array of shape (3, 3)
        The cell that you want to check.
    tol: float, optional
        Threshold value to consider a component of the cell nonzero.
    """
    cell = Lattice.new(cell).cell

    bigger_than_tol = abs(cell) > tol
    return (
        bigger_than_tol.sum() == 3
        and bigger_than_tol.any(axis=0).all()
        and bigger_than_tol.any(axis=1).all()
    )


def is_1D_cartesian(
    cell: LatticeLike, coord_ax: Literal["x", "y", "z"], tol: float = 1e-3
) -> bool:
    """Whether a cell contains only one vector that contributes only to a given coordinate.

    That is, one vector follows the direction of the cartesian axis and the other vectors don't
    have any component in that direction.

    Parameters
    -----------
    cell: np.array of shape (3, 3)
        The cell that you want to check.
    coord_ax: {"x", "y", "z"}
        The cartesian axis that you are looking for in the cell.
    tol: float, optional
        Threshold value to consider a component of the cell nonzero.
    """
    cell = Lattice.new(cell).cell

    coord_index = "xyz".index(coord_ax)
    lattice_vecs = np.where(cell[:, coord_index] > tol)[0]

    is_1D_cartesian = lattice_vecs.shape[0] == 1
    return is_1D_cartesian and (cell[lattice_vecs[0]] > tol).sum() == 1


def infer_cell_axes(cell: LatticeLike, axes: list[str], tol: float = 1e-3) -> list[int]:
    """Returns the indices of the lattice vectors that correspond to the given axes."""
    cell = Lattice.new(cell).cell

    grid_axes = []
    for ax in axes:
        if ax in ("x", "y", "z"):
            coord_index = "xyz".index(ax)
            lattice_vecs = np.where(cell[:, coord_index] > tol)[0]
            if lattice_vecs.shape[0] != 1:
                raise ValueError(
                    f"There are {lattice_vecs.shape[0]} lattice vectors that contribute to the {'xyz'[coord_index]} coordinate."
                )
            grid_axes.append(lattice_vecs[0])
        else:
            grid_axes.append("abc".index(ax))

    return grid_axes


def gen_cell_dataset(lattice: LatticeLike) -> CellDataset:
    """Generates a dataset with the vertices of the cell."""
    lattice = Lattice.new(lattice)

    return Dataset(
        {"xyz": (("a", "b", "c", "axis"), lattice.vertices())},
        coords={"a": [0, 1], "b": [0, 1], "c": [0, 1], "axis": [0, 1, 2]},
        attrs={"lattice": lattice},
    )


class CellStyleSpec(TypedDict):
    color: Any
    width: Any
    opacity: Any


class PartialCellStyleSpec(TypedDict, total=False):
    color: Any
    width: Any
    opacity: Any


def cell_to_lines(
    cell_data: CellDataset,
    how: Literal["box", "axes"],
    cell_style: PartialCellStyleSpec = {},
) -> CellDataset:
    """Converts a cell dataset to lines that should be plotted.

    Parameters
    -----------
    cell_data: xr.Dataset
        The cell dataset, containing the vertices of the cell.
    how: {"box", "axes"}
        Whether to draw the cell as a box or as axes.
        This determines how many points are needed to draw the cell
        using lines, and where those points are located.
    cell_style: dict, optional
        Style of the cell lines. A dictionary optionally containing
        the keys "color", "width" and "opacity".
    """
    cell_data = cell_data.reindex(a=[0, 1, 2], b=[0, 1, 2], c=[0, 1, 2])

    if how == "box":
        verts = np.array(
            [
                (0, 0, 0),
                (0, 1, 0),
                (1, 1, 0),
                (1, 1, 1),
                (0, 1, 1),
                (0, 1, 0),
                (2, 2, 2),
                (0, 1, 1),
                (0, 0, 1),
                (0, 0, 0),
                (1, 0, 0),
                (1, 0, 1),
                (0, 0, 1),
                (2, 2, 2),
                (1, 1, 0),
                (1, 0, 0),
                (2, 2, 2),
                (1, 1, 1),
                (1, 0, 1),
            ]
        )

    elif how == "axes":
        verts = np.array(
            [
                (0, 0, 0),
                (1, 0, 0),
                (2, 2, 2),
                (0, 0, 0),
                (0, 1, 0),
                (2, 2, 2),
                (0, 0, 0),
                (0, 0, 1),
                (2, 2, 2),
            ]
        )
    else:
        raise ValueError(
            f"'how' argument must be either 'box' or 'axes', but got {how}"
        )

    xyz = cell_data.xyz.values[verts[:, 0], verts[:, 1], verts[:, 2]]

    cell_data = cell_data.assign(
        {
            "xyz": (("point_index", "axis"), xyz),
            "color": cell_style.get("color"),
            "width": cell_style.get("width"),
            "opacity": cell_style.get("opacity"),
        }
    )

    return cell_data
