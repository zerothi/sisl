# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Literal

from sisl.typing import LatticeLike

from ..processors.cell import cell_to_lines, gen_cell_dataset
from ..processors.coords import project_to_axes
from ..types import Axes
from .xarray import draw_xarray_xy


def get_ndim(axes: Axes) -> int:
    return len(axes)


def get_z(ndim: int) -> Literal["z", False]:
    if ndim == 3:
        z = "z"
    else:
        z = False
    return z


def cell_plot_actions(
    cell: LatticeLike = None,
    show_cell: Literal[False, "box", "axes"] = "box",
    axes=["x", "y", "z"],
    name: str = "Unit cell",
    cell_style={},
    dataaxis_1d=None,
):
    if show_cell == False:
        cell_plottings = []
    else:
        cell_ds = gen_cell_dataset(cell)
        cell_lines = cell_to_lines(cell_ds, show_cell, cell_style)
        projected_cell_lines = project_to_axes(
            cell_lines, axes=axes, dataaxis_1d=dataaxis_1d
        )

        ndim = get_ndim(axes)
        z = get_z(ndim)
        cell_plottings = draw_xarray_xy(
            data=projected_cell_lines,
            x="x",
            y="y",
            z=z,
            set_axequal=ndim > 1,
            name=name,
        )

    return cell_plottings
