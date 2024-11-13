# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Optional

import numpy as np
import xarray as xr

from sisl.viz.processors.grid import get_isos

from . import plot_actions


def draw_grid(
    data,
    isos: list[dict] = [],
    colorscale: Optional[str] = None,
    crange: Optional[tuple[float, float]] = None,
    cmid: Optional[float] = None,
    smooth: bool = False,
    color_pixels_2d: bool = True,
    textformat: Optional[str] = None,
    textfont: dict = {},
    name: Optional[str] = None,
    coloraxis_name: Optional[str] = "grid_color",
    set_equal_axes: bool = True,
):
    to_plot = []

    # If it is a numpy array, convert it to a DataArray
    if not isinstance(data, xr.DataArray):
        # If it's 2D, here we assume that the array is a matrix.
        # Therefore, rows are y and columns are x. Otherwise, we
        # assume that dimensions are cartesian coordinates.
        if data.ndim == 2:
            dims = ["y", "x"]
        else:
            dims = ["x", "y", "z"][: data.ndim]
        data = xr.DataArray(data, dims=dims)

    ndim = data.ndim

    if ndim == 1:
        to_plot.append(plot_actions.draw_line(x=data.x, y=data.values, name=name))
    elif ndim == 2:
        data = data.transpose("y", "x")

        cmin, cmax = crange if crange is not None else (None, None)

        to_plot.append(
            plot_actions.init_coloraxis(
                name=coloraxis_name,
                cmin=cmin,
                cmax=cmax,
                cmid=cmid,
                colorscale=colorscale,
                showscale=color_pixels_2d,
            )
        )

        if not color_pixels_2d:
            textfont = {"color": "black", **textfont}

        to_plot.append(
            plot_actions.draw_heatmap(
                values=data.values,
                x=data.x if "x" in data.coords else None,
                y=data.y if "y" in data.coords else None,
                name=name,
                opacity=1 if color_pixels_2d else 0,
                zsmooth="best" if smooth else False,
                coloraxis=coloraxis_name,
                textformat=textformat,
                textfont=textfont,
            )
        )

        iso_lines = get_isos(data, isos)
        for iso_line in iso_lines:
            iso_line["line"] = {
                "color": iso_line.pop("color", None),
                "opacity": iso_line.pop("opacity", None),
                "width": iso_line.pop("width", None),
                **iso_line.get("line", {}),
            }
            to_plot.append(plot_actions.draw_line(**iso_line))
    elif ndim == 3:
        isosurfaces = get_isos(data, isos)

        for isosurface in isosurfaces:
            to_plot.append(plot_actions.draw_mesh_3D(**isosurface))

    if set_equal_axes and ndim > 1:
        to_plot.append(plot_actions.set_axes_equal())

    return to_plot


def draw_grid_arrows(data, arrows: list[dict]):
    to_plot = []

    # If it is a numpy array, convert it to a DataArray
    if not isinstance(data, xr.DataArray):
        # If it's 2D, here we assume that the array is a matrix.
        # Therefore, rows are y and columns are x. Otherwise, we
        # assume that dimensions are cartesian coordinates.
        if data.ndim == 2:
            dims = ["y", "x"]
        else:
            dims = ["x", "y", "z"][: data.ndim]
        data = xr.DataArray(data, dims=dims)

    ndim = data.ndim

    if ndim == 1:
        return []
    elif ndim == 2:
        coords = np.array(np.meshgrid(data.x, data.y))
        coords = coords.transpose(1, 2, 0)
        flat_coords = coords.reshape(-1, coords.shape[-1])

        for arrow_data in arrows:
            center = arrow_data.get("center", "middle")

            values = (
                arrow_data["data"]
                if "data" in arrow_data
                else np.stack([np.zeros_like(data.values), -data.values], axis=-1)
            )
            arrows_array = xr.DataArray(values, dims=["y", "x", "arrow_coords"])

            arrow_norms = arrows_array.reduce(np.linalg.norm, "arrow_coords")
            arrow_max = np.nanmax(arrow_norms)
            normed_arrows = (
                arrows_array / arrow_max * (1 if center == "middle" else 0.5)
            )

            flat_normed_arrows = normed_arrows.values.reshape(-1, coords.shape[-1])

            x = flat_coords[:, 0]
            y = flat_coords[:, 1]
            if center == "middle":
                x = x - flat_normed_arrows[:, 0] / 2
                y = y - flat_normed_arrows[:, 1] / 2
            elif center == "end":
                x = x - flat_normed_arrows[:, 0]
                y = y - flat_normed_arrows[:, 1]
            elif center != "start":
                raise ValueError(
                    f"Invalid value for 'center' in arrow data: {center}. Must be 'start', 'middle' or 'end'."
                )

            to_plot.append(
                plot_actions.draw_arrows(
                    x=x,
                    y=y,
                    dxy=flat_normed_arrows,
                    name=arrow_data.get("name", None),
                    line=dict(
                        width=arrow_data.get("width", None),
                        color=arrow_data.get("color", None),
                        opacity=arrow_data.get("opacity", None),
                        dash=arrow_data.get("dash", None),
                    ),
                    arrowhead_scale=arrow_data.get("arrowhead_scale", 0.2),
                    arrowhead_angle=arrow_data.get("arrowhead_angle", 20),
                )
            )

    elif ndim == 3:
        return []

    return to_plot
