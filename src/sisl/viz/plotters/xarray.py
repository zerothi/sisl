# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import itertools
import typing

import numpy as np
from xarray import DataArray

import sisl.viz.plotters.plot_actions as plot_actions
from sisl.messages import info

# from sisl.viz.nodes.processors.grid import get_isos


def _process_xarray_data(data, x=None, y=None, z=False, style={}):
    axes = {"x": x, "y": y}
    if z is not False:
        axes["z"] = z

    ndim = len(axes)

    # Normalize data to a Dataset
    if isinstance(data, DataArray):
        if np.all([ax is None for ax in axes.values()]):
            raise ValueError(
                "You have to provide either x or y (or z if it is not False) (one needs to be the fixed variable)."
            )
        axes = {k: v or data.name for k, v in axes.items()}
        data = data.to_dataset(name=data.name)
    else:
        if np.any([ax is None for ax in axes.values()]):
            raise ValueError(
                "Since you provided a Dataset, you have to provide both x and y (and z if it is not False)."
            )

    data_axis = None
    fixed_axes = {}
    # Check, for each axis, if it is uni dimensional (in which case we add it to the fixed axes dictionary)
    # or it contains more than one dimension, in which case we set it as the data axis
    for k in axes:
        if axes[k] in data.coords or (axes[k] in data and data[axes[k]].ndim == 1):
            if len(fixed_axes) < ndim - 1:
                fixed_axes[k] = axes[k]
            else:
                data_axis = k
        else:
            data_axis = k

    # Transpose the data so that the fixed axes are first.
    last_dims = []
    for ax_key, fixed_axis in fixed_axes.items():
        if fixed_axis not in data.dims:
            # This means that the fixed axis is a variable, which should contain only one dimension
            last_dim = data[fixed_axis].dims[-1]
        else:
            last_dim = fixed_axis
        last_dims.append(last_dim)
    last_dims = np.unique(last_dims)
    data = data.transpose(..., *last_dims)

    data_var = axes[data_axis]

    style_dims = set()
    for key, value in style.items():
        if value in data:
            style_dims = style_dims.union(set(data[value].dims))

    extra_style_dims = style_dims - set(data[data_var].dims)
    if extra_style_dims:
        data = data.stack(extra_style_dim=extra_style_dims).transpose(
            "extra_style_dim", ...
        )

    if data[data_var].shape[0] == 0:
        return None, None, None, None, None

    if len(data[data_var].shape) == 1:
        data = data.expand_dims(dim={"fake_dim": [0]}, axis=0)
    # We have to flatten all the dimensions that will not be represented as an axis,
    # since we will just iterate over them.
    dims_to_stack = data[data_var].dims[: -len(last_dims)]
    data = data.stack(iterate_dim=dims_to_stack).transpose("iterate_dim", ...)

    styles = {}
    for key, value in style.items():
        if value in data:
            styles[key] = data[value]
        elif key == "name":
            styles[key] = DataArray(value)
        else:
            styles[key] = None

    plot_data = data[axes[data_axis]]

    fixed_coords = {}
    for ax_key, fixed_axis in fixed_axes.items():
        fixed_coord = data[fixed_axis]
        if "iterate_dim" in fixed_coord.dims:
            # This is if fixed_coord was a variable of the dataset, which possibly has
            # gotten the extra iterate_dim added.
            fixed_coord = fixed_coord.isel(iterate_dim=0)
        fixed_coords[ax_key] = fixed_coord

    # info(f"{self} variables: \n\t- Fixed: {fixed_axes}\n\t- Data axis: {data_axis}\n\t")

    return plot_data, fixed_coords, styles, data_axis, axes


def draw_xarray_xy(
    data,
    x=None,
    y=None,
    z=False,
    color="color",
    width="width",
    dash="dash",
    opacity="opacity",
    name="",
    colorscale=None,
    what: typing.Literal[
        "line", "scatter", "balls", "area_line", "arrows", "none"
    ] = "line",
    dependent_axis: typing.Optional[typing.Literal["x", "y"]] = None,
    set_axrange=False,
    set_axequal=False,
):
    if what == "none":
        return []

    plot_data, fixed_coords, styles, data_axis, axes = _process_xarray_data(
        data,
        x=x,
        y=y,
        z=z,
        style={
            "color": color,
            "width": width,
            "opacity": opacity,
            "dash": dash,
            "name": name,
        },
    )

    if plot_data is None:
        return []

    to_plot = _draw_xarray_lines(
        data=plot_data,
        style=styles,
        fixed_coords=fixed_coords,
        data_axis=data_axis,
        colorscale=colorscale,
        what=what,
        name=name,
        dependent_axis=dependent_axis,
    )

    if set_axequal:
        to_plot.append(plot_actions.set_axes_equal())

    # Set axis range
    for key, coord_key in axes.items():
        if coord_key == getattr(data, "name", None):
            ax = data
        else:
            ax = data[coord_key]
        title = ax.name
        units = ax.attrs.get("units")
        if units:
            title += f" [{units}]"

        axis = {"title": title}

        if set_axrange:
            axis["range"] = (float(ax.min()), float(ax.max()))

        axis.update(ax.attrs.get("axis", {}))

        to_plot.append(plot_actions.set_axis(axis=key, **axis))

    return to_plot


def _draw_xarray_lines(
    data, style, fixed_coords, data_axis, colorscale, what, name="", dependent_axis=None
):
    # Initialize actions list
    to_plot = []

    # Get the lines styles
    lines_style = {}
    extra_style_dims = False
    for key in ("color", "width", "opacity", "dash", "name"):
        lines_style[key] = style.get(key)

        if lines_style[key] is not None:
            extra_style_dims = (
                extra_style_dims or "extra_style_dim" in lines_style[key].dims
            )
        # If some style is constant, just repeat it.
        if lines_style[key] is None or "iterate_dim" not in lines_style[key].dims:
            lines_style[key] = itertools.repeat(lines_style[key])

    # If we have to draw multicolored lines, we have to initialize a color axis and
    # use a special drawing function. If we have to draw lines with multiple widths
    # we also need to use a special function.
    line_kwargs = {}
    if isinstance(lines_style["color"], itertools.repeat):
        color_value = next(lines_style["color"])
    else:
        color_value = lines_style["color"]

    if isinstance(lines_style["width"], itertools.repeat):
        width_value = next(lines_style["width"])
    else:
        width_value = lines_style["width"]

    if isinstance(color_value, DataArray) and (data.dims[-1] in color_value.dims):
        color = color_value
        if color.dtype in (int, float):
            coloraxis_name = f"{color.name}_{name}" if name else color.name
            to_plot.append(
                plot_actions.init_coloraxis(
                    name=coloraxis_name,
                    cmin=color.values.min(),
                    cmax=color.values.max(),
                    colorscale=colorscale,
                )
            )
            line_kwargs = {"coloraxis": coloraxis_name}
        drawing_function_name = f"draw_multicolor_{what}"
    elif isinstance(width_value, DataArray) and (data.dims[-1] in width_value.dims):
        drawing_function_name = f"draw_multisize_{what}"
    else:
        drawing_function_name = f"draw_{what}"

    # Check if we have to use a 3D function
    if len(fixed_coords) == 2:
        to_plot.append(plot_actions.init_3D())
        drawing_function_name += "_3D"

    _drawing_function = getattr(plot_actions, drawing_function_name)
    if what in ("scatter", "balls"):

        def drawing_function(*args, **kwargs):
            marker = kwargs.pop("line")
            marker["size"] = marker.pop("width")

            to_plot.append(_drawing_function(*args, marker=marker, **kwargs))

    elif what == "area_line":

        def drawing_function(*args, **kwargs):
            to_plot.append(
                _drawing_function(*args, dependent_axis=dependent_axis, **kwargs)
            )

    else:

        def drawing_function(*args, **kwargs):
            to_plot.append(_drawing_function(*args, **kwargs))

    # Define the iterator over lines, containing both values and styles
    iterator = zip(
        data,
        lines_style["color"],
        lines_style["width"],
        lines_style["opacity"],
        lines_style["dash"],
        lines_style["name"],
    )

    fixed_coords_values = {k: arr.values for k, arr in fixed_coords.items()}

    # Now just iterate over each line and plot it.
    for values, *styles in iterator:

        parsed_styles = []
        for style in styles:
            if style is not None:
                style = style.values
                if style.ndim == 0:
                    style = style[()]
            parsed_styles.append(style)

        line_color, line_width, line_opacity, line_dash, line_name = parsed_styles
        line_style = {
            "color": line_color,
            "width": line_width,
            "opacity": line_opacity,
            "dash": line_dash,
        }
        line = {**line_style, **line_kwargs}

        coords = {
            data_axis: values,
            **fixed_coords_values,
        }

        if not extra_style_dims:
            drawing_function(**coords, line=line, name=str(line_name))
        else:
            for k, v in line_style.items():
                if v is None or v.ndim == 0:
                    line_style[k] = itertools.repeat(v)

            for l_color, l_width, l_opacity, l_dash in zip(
                line_style["color"],
                line_style["width"],
                line_style["opacity"],
                line_style["dash"],
            ):
                line_style = {
                    "color": l_color,
                    "width": l_width,
                    "opacity": l_opacity,
                    "dash": l_dash,
                }
                drawing_function(**coords, line=line_style, name=str(line_name))

    return to_plot


# class PlotterNodeGrid(PlotterXArray):

#     def draw(self, data, isos=[]):

#         ndim = data.ndim

#         if ndim == 2:
#             transposed = data.transpose("y", "x")

#             self.draw_heatmap(transposed.values, x=data.x, y=data.y, name="HEAT", zsmooth="best")

#             dx = data.x[1] - data.x[0]
#             dy = data.y[1] - data.y[0]

#             iso_lines = get_isos(transposed, isos)
#             for iso_line in iso_lines:
#                 iso_line['line'] = {
#                     "color": iso_line.pop("color", None),
#                     "opacity": iso_line.pop("opacity", None),
#                     "width": iso_line.pop("width", None),
#                     **iso_line.get("line", {})
#                 }
#                 self.draw_line(**iso_line)
#         elif ndim == 3:
#             isosurfaces = get_isos(data, isos)

#             for isosurface in isosurfaces:
#                 self.draw_mesh_3D(**isosurface)


#         self.set_axes_equal()
