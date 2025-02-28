# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import itertools
import typing
from typing import Literal, Optional, Union

import numpy as np
from xarray import DataArray, Dataset

from . import plot_actions

# from sisl.viz.nodes.processors.grid import get_isos


def _process_xarray_data(
    data: Union[DataArray, Dataset],
    x: Union[str, None] = None,
    y: Union[str, None] = None,
    z: Union[str, Literal[False], None] = False,
    style: dict[str, str] = {},
) -> tuple[
    DataArray,
    dict[str, DataArray],
    dict[str, DataArray],
    Literal["x", "y", "z"],
    dict[str, str],
]:
    """Preprocesses a dataarray or dataset with plotting specifications.

    The complexity of this function lies in that it supports that one axis of the plot
    contains multiple specifications. E.g. for a 2D line plot you can have:

        -  A list of values for the X axis.
        -  Multiple lists of values for the Y axis.

    which would indicate that you want to do something like:

    ... code-block:: python

        x = [1, 2, 3]
        y = [[1, 2, 3], [2, 3, 4]]

        plot_line(x, y[0])
        plot_line(x, y[1])

    in other words, multiple lines can have different Y while sharing the same X.
    You can also have different styles for each line. Only one axis is allowed to
    have multiple specifications, we refer to this axis as the "free" axis. The other
    ones are shared/fixed axes.

    In practice, this is handled by looking at the dimensions of the variables in the
    `DataArray`/`Dataset`. Look at the Examples section for a practical example.

    Parameters
    ----------
    data:
        data to process for the plot.
    x:
        name of the x-axis variable. It can be either the name of a coordinate in `data`
        or a the name of a variable in `data` in case `data` is a `Dataset`.

        If it is `None`, it means that the data goes to the X axis. Potentially the X axis
        can then contain multiple specifications. I.e. it is not a shared/fixed axis.
    y:
        name of the y-axis variable. It can be either the name of a coordinate in `data`
        or a the name of a variable in `data` in case `data` is a `Dataset`.

        If it is `None`, it means that the data goes to the Y axis. Potentially the Y axis
        can then contain multiple specifications. I.e. it is not a shared/fixed axis.
    z:
        name of the z-axis variable. It can be either the name of a coordinate in `data`
        or a the name of a variable in `data` in case `data` is a `Dataset`.

        If `z` is `False`, it means that the plot is 2D.

        If it is `None`, it means that the data goes to the Z axis. Potentially the Z axis
        can then contain multiple specifications. I.e. it is not a shared/fixed axis.
    style:
        dictionary containing the styles for the plot. The keys are the names of the styles
        and the values are the names of the variables in `data` that contain the style values.

        The coordinates of styles are matched with the coordinates of the data. E.g. if a style
        depends on the coordinates (`time`, `space`) and the data also contains this coordinates,
        each data point (time, space) will be styled accordingly.

        Different styles can have different dimensions and coordinates. If a style has more
        dimensions than the data, the same data will be plotted multiple times with
        different styles.

    Returns
    -------
    plot_data:
        Contains the values of the "free" axis, i.e. the axis that has not been fixed. If the data
        contains spare dimensions (i.e. dimensions that have not been assigned to a fixed axis),
        this means that the user wants to plot multiple traces. In this returned object, all the
        extra dimensions are stacked/flattened into a single dimension called `iterate_dim`.

        If there was no extra dimensions, i.e. only one trace is to be plotted, the returned object
        contains a dummy dimension called `fake_dim` with a single value.
    fixed_coords:
        Contains the values for the fixed axes. The keys are the names of the axes and the values
        are the values of this axis shared by all the traces in `plot_data`.
    styles:
        The values of each style. The `DataArray` of each style contains coordinates matching the
        coordinates of `plot_data`, including possibly `"iterate_dim"` and `"fake_dim"`.
    data_axis:
        The name of the "free" axis.
    axes:
        Mapping from axis name to the name of the coordinate or variable in `data` that contains
        the values for this axis.

    Examples
    --------
    Let's say that the input data is a `DataArray` with coordinates `"time"` and `"space"`.
    We want to plot lines from it.

    If `x` is set to `"time"`, the time coordinate will go to the X axis.

    Now, there are multiple options:

        -   `y` is set to `None`, `z` is set to `False`. The values of the `DataArray`
            will go to the Y axis. But the data still has one extra dimension (space).
            Multiple lines will be plotted, one for each value of `space`.
            If one of the style attributes contains the `space` coordinate, each line
            will have also a different style.

        -   `y` is set to `None`, `z` is set to `"space"`. The space coordinate will go
            to the Z axis. The values of the `DataArray` will go to the Y axis. Since there
            are no unused coordinates, only one line will be plotted.

        -   `y` is set to `"space"`, `z` is set to `None`. The space coordinate will go
            to the Y axis. The values of the `DataArray` will go to the Z axis. Since there
            are no unused coordinates, only one line will be plotted.
    """
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

    return plot_data, fixed_coords, styles, data_axis, axes


def draw_xarray_xy(
    data: Union[DataArray, Dataset],
    x: Union[str, None] = None,
    y: Union[str, None] = None,
    z: Union[str, Literal[False], None] = False,
    color: Optional[str] = "color",
    width: Optional[str] = "width",
    dash: Optional[str] = "dash",
    opacity: Optional[str] = "opacity",
    border_width: Optional[str] = "border_width",
    border_color: Optional[str] = "border_color",
    name: Optional[str] = "",
    colorscale: Optional[str] = None,
    what: Literal["line", "scatter", "balls", "area_line", "arrows", "none"] = "line",
    dependent_axis: Optional[Literal["x", "y"]] = None,
    set_axrange: bool = False,
    set_axequal: bool = False,
) -> list[dict]:
    """Generates plotting actions to draw lines/scatter or similars from xarray data.

    The complexity of this function lies in that it supports that one axis of the plot
    contains multiple specifications. E.g. for a 2D line plot you can have:

        -  A list of values for the X axis.
        -  Multiple lists of values for the Y axis.

    which would indicate that you want to do something like:

    ... code-block:: python

        x = [1, 2, 3]
        y = [[1, 2, 3], [2, 3, 4]]

        plot_line(x, y[0])
        plot_line(x, y[1])

    in other words, multiple lines can have different Y while sharing the same X.
    You can also have different styles for each line. Only one axis is allowed to
    have multiple specifications, we refer to this axis as the "free" axis. The other
    ones are shared/fixed axes.

    In practice, this is handled by looking at the dimensions of the variables in the
    `DataArray`/`Dataset`. Look at the Examples section for a practical example.

    Parameters
    ----------
    data:
        data to process for the plot.
    x:
        name of the x-axis variable. It can be either the name of a coordinate in `data`
        or a the name of a variable in `data` in case `data` is a `Dataset`.

        If it is `None`, it means that the data goes to the X axis. Potentially the X axis
        can then contain multiple specifications. I.e. it is not a shared/fixed axis.
    y:
        name of the y-axis variable. It can be either the name of a coordinate in `data`
        or a the name of a variable in `data` in case `data` is a `Dataset`.

        If it is `None`, it means that the data goes to the Y axis. Potentially the Y axis
        can then contain multiple specifications. I.e. it is not a shared/fixed axis.
    z:
        name of the z-axis variable. It can be either the name of a coordinate in `data`
        or a the name of a variable in `data` in case `data` is a `Dataset`.

        If `z` is `False`, it means that the plot is 2D.

        If it is `None`, it means that the data goes to the Z axis. Potentially the Z axis
        can then contain multiple specifications. I.e. it is not a shared/fixed axis.
    color:
        name of the variable in `data` that contains the color values.
    width:
        name of the variable in `data` that contains the width values.
    dash:
        name of the variable in `data` that contains the dash values.
    opacity:
        name of the variable in `data` that contains the opacity values.
    border_width:
        name of the variable in `data` that contains the border width values.
    border_color:
        name of the variable in `data` that contains the border color values.
    name:
        name of the variable in `data` that contains the names of each trace.
    colorscale:
        The name of a colorscale that is supported by the plotting backend.
        It will be used if the color values are numerical.
    what:
        Method of plotting the XY(Z) points. Some plotting backends might
        not support some of the methods.
    dependent_axis:
        When using the `area_line` method, this determines whether the area is filled
        vertically or horizontally.
    set_axrange:
        If `True`, the axis range will be set to the minimum and maximum values of the
        corresponding coordinate.
    set_axequal:
        If `True`, the axes will be set to have the same scale. I.e. a distance of 1
        in the X axis will be the same amount of pixels as a distance of 1 in the Y axis.
    Returns
    -------
    plot_actions:
        List of plotting actions that need to be used to generate the requested plot.

    Examples
    --------
    Let's say that the input data is a `DataArray` with coordinates `"time"` and `"space"`.
    We want to plot lines from it.

    If `x` is set to `"time"`, the time coordinate will go to the X axis.

    Now, there are multiple options:

        -   `y` is set to `None`, `z` is set to `False`. The values of the `DataArray`
            will go to the Y axis. But the data still has one extra dimension (space).
            Multiple lines will be plotted, one for each value of `space`.
            If one of the style attributes contains the `space` coordinate, each line
            will have also a different style.

        -   `y` is set to `None`, `z` is set to `"space"`. The space coordinate will go
            to the Z axis. The values of the `DataArray` will go to the Y axis. Since there
            are no unused coordinates, only one line will be plotted.

        -   `y` is set to `"space"`, `z` is set to `None`. The space coordinate will go
            to the Y axis. The values of the `DataArray` will go to the Z axis. Since there
            are no unused coordinates, only one line will be plotted.
    """
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
            "border_width": border_width,
            "border_color": border_color,
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
    data: DataArray,
    style: dict[str, DataArray],
    fixed_coords: dict[str, DataArray],
    data_axis: str,
    colorscale: Optional[str],
    what: Literal["line", "scatter", "balls", "area_line", "arrows", "none"],
    name: str = "",
    dependent_axis: Optional[Literal["x", "y"]] = None,
):
    """Creates the drawing actions from the processed xarray data.

    The values accepted by this function are the ones returned by `_process_xarray_data`.

    Parameters
    ----------
    data:
        The values of the "free" axis.
    style:
        The values of each style.
    fixed_coords:
        The values for the fixed axes.
    data_axis:
        The name of the "free" axis.
    colorscale:
        The name of a colorscale that is supported by the plotting backend.
        It will be used if the color values are numerical.
    what:
        Method of plotting the XY(Z) points.
    name:
        Added to as a suffix to the color axes that are created, so that they don't
        clash with other color axes.
    dependent_axis:
        When using the `area_line` method, this determines whether the area is filled

    """
    # Initialize actions list
    to_plot = []

    # Get the lines styles
    lines_style = {}
    extra_style_dims = False
    style_keys = (
        "color",
        "width",
        "opacity",
        "dash",
        "name",
        "border_width",
        "border_color",
    )
    for key in style_keys:
        lines_style[key] = style.get(key)

        if lines_style[key] is not None:
            extra_style_dims = (
                extra_style_dims or "extra_style_dim" in lines_style[key].dims
            )

            # A negative width does not make sense, just plot the absolute value
            # However we only take the absolute value if the array contains numbers.
            if key == "width" and np.issubdtype(lines_style[key].dtype, np.number):
                lines_style[key] = abs(lines_style[key])

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
            marker = {**kwargs.pop("line"), **kwargs.pop("marker")}
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
        lines_style["border_width"],
        lines_style["border_color"],
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

        (
            line_color,
            line_width,
            line_opacity,
            line_dash,
            line_name,
            border_width,
            border_color,
        ) = parsed_styles
        line_style = {
            "color": line_color,
            "width": line_width,
            "opacity": line_opacity,
            "dash": line_dash,
        }
        line = {**line_style, **line_kwargs}
        marker_style = {"line_width": border_width, "line_color": border_color}
        marker_style = {k: v for k, v in marker_style.items() if v is not None}

        coords = {
            data_axis: values,
            **fixed_coords_values,
        }

        if not extra_style_dims:
            drawing_function(
                **coords, line=line, marker=marker_style, name=str(line_name)
            )
        else:
            for k, v in line_style.items():
                if v is None or v.ndim == 0:
                    line_style[k] = itertools.repeat(v)

            for l_color, l_width, l_opacity, l_dash, b_color, b_width in zip(
                line_style["color"],
                line_style["width"],
                line_style["opacity"],
                line_style["dash"],
                marker_style["line_color"],
                marker_style["line_width"],
            ):
                line_style = {
                    "color": l_color,
                    "width": l_width,
                    "opacity": l_opacity,
                    "dash": l_dash,
                }
                marker_style = {"line_width": b_width, "line_color": b_color}
                marker_style = {k: v for k, v in marker_style.items() if v is not None}
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
