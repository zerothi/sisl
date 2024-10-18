# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections import ChainMap
from collections.abc import Callable, Sequence
from typing import Literal, Optional, Union

from sisl._core import Geometry, Grid

from ..data import EigenstateData
from ..figure import Figure, get_figure
from ..plot import Plot
from ..plotters.cell import cell_plot_actions
from ..plotters.grid import draw_grid
from ..plotters.plot_actions import combined
from ..processors.axes import sanitize_axes
from ..processors.eigenstate import (
    eigenstate_geometry,
    get_eigenstate,
    get_grid_nsc,
    project_wavefunction,
    tile_if_k,
)
from ..processors.grid import (
    apply_transforms,
    get_grid_axes,
    get_grid_representation,
    grid_geometry,
    grid_to_dataarray,
    interpolate_grid,
    orthogonalize_grid_if_needed,
    reduce_grid,
    sub_grid,
    tile_grid,
)
from ..types import Axes, Colorscale
from .geometry import geometry_plot


def _get_structure_plottings(
    plot_geom,
    geometry,
    axes,
    nsc,
    geom_kwargs={},
):
    if plot_geom:
        geom_kwargs = ChainMap(
            geom_kwargs,
            {"axes": axes, "geometry": geometry, "nsc": nsc, "show_cell": False},
        )
        plot_actions = geometry_plot(**geom_kwargs).plot_actions
    else:
        plot_actions = []

    return plot_actions


def grid_plot(
    grid: Optional[Grid] = None,
    axes: Axes = ["z"],
    represent: Literal[
        "real", "imag", "mod", "phase", "deg_phase", "rad_phase"
    ] = "real",
    transforms: Sequence[Union[str, Callable]] = (),
    reduce_method: Literal["average", "sum"] = "average",
    boundary_mode: str = "grid-wrap",
    nsc: tuple[int, int, int] = (1, 1, 1),
    interp: tuple[int, int, int] = (1, 1, 1),
    isos: Sequence[dict] = [],
    smooth: bool = False,
    colorscale: Optional[Colorscale] = None,
    crange: Optional[tuple[float, float]] = None,
    cmid: Optional[float] = None,
    show_cell: Literal["box", "axes", False] = "box",
    cell_style: dict = {},
    x_range: Optional[Sequence[float]] = None,
    y_range: Optional[Sequence[float]] = None,
    z_range: Optional[Sequence[float]] = None,
    plot_geom: bool = False,
    geom_kwargs: dict = {},
    backend: str = "plotly",
) -> Figure:
    """Plots a grid, with plentiful of customization options.

    Parameters
    ----------
    grid:
        The grid to plot.
    axes:
        The axes to project the grid to.
    represent:
        The representation of the grid to plot.
    transforms:
        List of transforms to apply to the grid before plotting.
    reduce_method:
        The method used to reduce the grid axes that are not displayed.
    boundary_mode:
        The method used to deal with the boundary conditions.
        Only used if the grid is to be orthogonalized.
        See scipy docs for more info on the possible values.
    nsc:
        The number of unit cells to display in each direction.
    interp:
        The interpolation factor to use for each axis to make the grid smoother.
    isos:
        List of isosurfaces or isocontours to plot. See the showcase notebooks for examples.
    smooth:
        Whether to ask the plotting backend to make an attempt at smoothing the grid display.
    colorscale:
        Colorscale to use for the grid display in the 2D representation.
        If None, the default colorscale is used for each backend.
    crange:
        Min and max values for the colorscale.
    cmid:
        The value at which the colorscale is centered.
    show_cell:
        Method used to display the unit cell. If False, the cell is not displayed.
    cell_style:
        Style specification for the cell. See the showcase notebooks for examples.
    x_range:
        The range of the x axis to take into account.
        Even if the X axis is not displayed! This is important because the reducing
        operation will only be applied on this range.
    y_range:
        The range of the y axis to take into account.
        Even if the Y axis is not displayed! This is important because the reducing
        operation will only be applied on this range.
    z_range:
        The range of the z axis to take into account.
        Even if the Z axis is not displayed! This is important because the reducing
        operation will only be applied on this range.
    plot_geom:
        Whether to plot the associated geometry (if any).
    geom_kwargs:
        Keyword arguments to pass to the geometry plot of the associated geometry.
    backend:
        The backend to use to generate the figure.

    See also
    --------
    scipy.ndimage.affine_transform : method used to orthogonalize the grid if needed.
    """

    axes = sanitize_axes(axes)

    geometry = grid_geometry(grid, geometry=None)

    grid_repr = get_grid_representation(grid, represent=represent)

    tiled_grid = tile_grid(grid_repr, nsc=nsc)

    ort_grid = orthogonalize_grid_if_needed(tiled_grid, axes=axes, mode=boundary_mode)

    grid_axes = get_grid_axes(ort_grid, axes=axes)

    transformed_grid = apply_transforms(ort_grid, transforms)

    subbed_grid = sub_grid(
        transformed_grid, x_range=x_range, y_range=y_range, z_range=z_range
    )

    reduced_grid = reduce_grid(
        subbed_grid, reduce_method=reduce_method, keep_axes=grid_axes
    )

    interp_grid = interpolate_grid(reduced_grid, interp=interp)

    # Finally, here comes the plotting!
    grid_ds = grid_to_dataarray(interp_grid, axes=axes, grid_axes=grid_axes, nsc=nsc)
    grid_plottings = draw_grid(
        data=grid_ds,
        isos=isos,
        colorscale=colorscale,
        crange=crange,
        cmid=cmid,
        smooth=smooth,
    )

    # Process the cell as well
    cell_plottings = cell_plot_actions(
        cell=grid,
        show_cell=show_cell,
        cell_style=cell_style,
        axes=axes,
    )

    # And maybe plot the strucuture
    geom_plottings = _get_structure_plottings(
        plot_geom=plot_geom,
        geometry=geometry,
        geom_kwargs=geom_kwargs,
        axes=axes,
        nsc=nsc,
    )

    all_plottings = combined(
        grid_plottings, cell_plottings, geom_plottings, composite_method=None
    )

    return get_figure(backend=backend, plot_actions=all_plottings)


def wavefunction_plot(
    eigenstate: EigenstateData,
    i: int = 0,
    geometry: Optional[Geometry] = None,
    grid_prec: float = 0.2,
    # All grid inputs.
    grid: Optional[Grid] = None,
    axes: Axes = ["z"],
    represent: Literal[
        "real", "imag", "mod", "phase", "deg_phase", "rad_phase"
    ] = "real",
    transforms: Sequence[Union[str, Callable]] = (),
    reduce_method: Literal["average", "sum"] = "average",
    boundary_mode: str = "grid-wrap",
    nsc: tuple[int, int, int] = (1, 1, 1),
    interp: tuple[int, int, int] = (1, 1, 1),
    isos: Sequence[dict] = [],
    smooth: bool = False,
    colorscale: Optional[Colorscale] = None,
    crange: Optional[tuple[float, float]] = None,
    cmid: Optional[float] = None,
    show_cell: Literal["box", "axes", False] = "box",
    cell_style: dict = {},
    x_range: Optional[Sequence[float]] = None,
    y_range: Optional[Sequence[float]] = None,
    z_range: Optional[Sequence[float]] = None,
    plot_geom: bool = False,
    geom_kwargs: dict = {},
    backend: str = "plotly",
) -> Figure:
    """Plots a wavefunction in real space.

    Parameters
    ----------
    eigenstate:
        The eigenstate object containing information about eigenstates.
    i:
        The index of the eigenstate to plot.
    geometry:
        Geometry to use to project the eigenstate to real space.
        If None, the geometry associated with the eigenstate is used.
    grid_prec:
        The precision of the grid where the wavefunction is projected.
    grid:
        The grid to plot.
    axes:
        The axes to project the grid to.
    represent:
        The representation of the grid to plot.
    transforms:
        List of transforms to apply to the grid before plotting.
    reduce_method:
        The method used to reduce the grid axes that are not displayed.
    boundary_mode:
        The method used to deal with the boundary conditions.
        Only used if the grid is to be orthogonalized.
        See scipy docs for more info on the possible values.
    nsc:
        The number of unit cells to display in each direction.
    interp:
        The interpolation factor to use for each axis to make the grid smoother.
    isos:
        List of isosurfaces or isocontours to plot. See the showcase notebooks for examples.
    smooth:
        Whether to ask the plotting backend to make an attempt at smoothing the grid display.
    colorscale:
        Colorscale to use for the grid display in the 2D representation.
        If None, the default colorscale is used for each backend.
    crange:
        Min and max values for the colorscale.
    cmid:
        The value at which the colorscale is centered.
    show_cell:
        Method used to display the unit cell. If False, the cell is not displayed.
    cell_style:
        Style specification for the cell. See the showcase notebooks for examples.
    x_range:
        The range of the x axis to take into account.
        Even if the X axis is not displayed! This is important because the reducing
        operation will only be applied on this range.
    y_range:
        The range of the y axis to take into account.
        Even if the Y axis is not displayed! This is important because the reducing
        operation will only be applied on this range.
    z_range:
        The range of the z axis to take into account.
        Even if the Z axis is not displayed! This is important because the reducing
        operation will only be applied on this range.
    plot_geom:
        Whether to plot the associated geometry (if any).
    geom_kwargs:
        Keyword arguments to pass to the geometry plot of the associated geometry.
    backend:
        The backend to use to generate the figure.

    See also
    --------
    scipy.ndimage.affine_transform : method used to orthogonalize the grid if needed.
    """

    # Create a grid with the wavefunction in it.
    i_eigenstate = get_eigenstate(eigenstate, i)
    geometry = eigenstate_geometry(eigenstate, geometry=geometry)

    tiled_geometry = tile_if_k(geometry=geometry, nsc=nsc, eigenstate=i_eigenstate)
    grid_nsc = get_grid_nsc(nsc=nsc, eigenstate=i_eigenstate)
    grid = project_wavefunction(
        eigenstate=i_eigenstate, grid_prec=grid_prec, grid=grid, geometry=tiled_geometry
    )

    # Grid processing
    axes = sanitize_axes(axes)

    grid_repr = get_grid_representation(grid, represent=represent)

    tiled_grid = tile_grid(grid_repr, nsc=grid_nsc)

    ort_grid = orthogonalize_grid_if_needed(tiled_grid, axes=axes, mode=boundary_mode)

    grid_axes = get_grid_axes(ort_grid, axes=axes)

    transformed_grid = apply_transforms(ort_grid, transforms)

    subbed_grid = sub_grid(
        transformed_grid, x_range=x_range, y_range=y_range, z_range=z_range
    )

    reduced_grid = reduce_grid(
        subbed_grid, reduce_method=reduce_method, keep_axes=grid_axes
    )

    interp_grid = interpolate_grid(reduced_grid, interp=interp)

    # Finally, here comes the plotting!
    grid_ds = grid_to_dataarray(
        interp_grid, axes=axes, grid_axes=grid_axes, nsc=grid_nsc
    )
    grid_plottings = draw_grid(
        data=grid_ds,
        isos=isos,
        colorscale=colorscale,
        crange=crange,
        cmid=cmid,
        smooth=smooth,
    )

    # Process the cell as well
    cell_plottings = cell_plot_actions(
        cell=grid,
        show_cell=show_cell,
        cell_style=cell_style,
        axes=axes,
    )

    # And maybe plot the strucuture
    geom_plottings = _get_structure_plottings(
        plot_geom=plot_geom,
        geometry=tiled_geometry,
        geom_kwargs=geom_kwargs,
        axes=axes,
        nsc=grid_nsc,
    )

    all_plottings = combined(
        grid_plottings, cell_plottings, geom_plottings, composite_method=None
    )

    return get_figure(backend=backend, plot_actions=all_plottings)


class GridPlot(Plot):
    function = staticmethod(grid_plot)


class WavefunctionPlot(GridPlot):
    function = staticmethod(wavefunction_plot)


# The following commented code is from the old viz module, where the GridPlot had a scan method.
# It looks very nice, but probably should be reimplemented as a standalone function that plots a grid slice,
# and then merge those grid slices to create a scan.

# def scan(self, along, start=None, stop=None, step=None, num=None, breakpoints=None, mode="moving_slice", animation_kwargs=None, **kwargs):
#         """
#         Returns an animation containing multiple frames scaning along an axis.

#         Parameters
#         -----------
#         along: {"x", "y", "z"}
#             the axis along which the scan is performed. If not provided, it will scan along the axes that are not displayed.
#         start: float, optional
#             the starting value for the scan (in Angstrom).
#             Make sure this value is inside the range of the unit cell, otherwise it will fail.
#         stop: float, optional
#             the last value of the scan (in Angstrom).
#             Make sure this value is inside the range of the unit cell, otherwise it will fail.
#         step: float, optional
#             the distance between steps in Angstrom.

#             If not provided and `num` is also not provided, it will default to 1 Ang.
#         num: int , optional
#             the number of steps that you want the scan to consist of.

#             If `step` is passed, this argument is ignored.

#             Note that the grid is only stored once, so having a big number of steps is not that big of a deal.
#         breakpoints: array-like, optional
#             the discrete points of the scan. To be used if you don't want regular steps.
#             If the last step is exactly the length of the cell, it will be moved one dcell back to avoid errors.

#             Note that if this parameter is passed, both `step` and `num` are ignored.
#         mode: {"moving_slice", "as_is"}, optional
#             the type of scan you want to see.
#             "moving_slice" renders a volumetric scan where a slice moves through the grid.
#             "as_is" renders each part of the scan as an animation frame.
#             (therefore, "as_is" SUPPORTS SCANNING 1D, 2D AND 3D REPRESENTATIONS OF THE GRID, e.g. display the volume data for different ranges of z)
#         animation_kwargs: dict, optional
#             dictionary whose keys and values are directly passed to the animated method as kwargs and therefore
#             end up being passed to animation initialization.
#         **kwargs:
#             the rest of settings that you want to apply to overwrite the existing ones.

#             This settings apply to each plot and go directly to their initialization.

#         Returns
#         -------
#         sisl.viz.plotly.Animation
#             An animation representation of the scan
#         """
#         # Do some checks on the args provided
#         if sum(1 for arg in (step, num, breakpoints) if arg is not None) > 1:
#             raise ValueError(f"Only one of ('step', 'num', 'breakpoints') should be passed.")

#         axes = self.inputs['axes']
#         if mode == "as_is" and set(axes) - set(["x", "y", "z"]):
#             raise ValueError("To perform a scan, the axes need to be cartesian. Please set the axes to a combination of 'x', 'y' and 'z'.")

#         if self.grid.lattice.is_cartesian():
#             grid = self.grid
#         else:
#             transform_bc = kwargs.pop("transform_bc", self.get_setting("transform_bc"))
#             grid, transform_offset = self._transform_grid_cell(
#                 self.grid, mode=transform_bc, output_shape=self.grid.shape, cval=np.nan
#             )

#             kwargs["offset"] = transform_offset + kwargs.get("offset", self.get_setting("offset"))

#         # We get the key that needs to be animated (we will divide the full range in frames)
#         range_key = f"{along}_range"
#         along_i = {"x": 0, "y": 1, "z": 2}[along]

#         # Get the full range
#         if start is not None and stop is not None:
#             along_range = [start, stop]
#         else:
#             along_range = self.get_setting(range_key)
#             if along_range is None:
#                 range_param = self.get_param(range_key)
#                 along_range = [range_param[f"inputField.params.{lim}"] for lim in ["min", "max"]]
#             if start is not None:
#                 along_range[0] = start
#             if stop is not None:
#                 along_range[1] = stop

#         if breakpoints is None:
#             if step is None and num is None:
#                 step = 1.0
#             if step is None:
#                 step = (along_range[1] - along_range[0]) / num
#             else:
#                 num = (along_range[1] - along_range[0]) // step

#             # np.linspace will use the last point as a step (and we don't want it)
#             # therefore we will add an extra step
#             breakpoints = np.linspace(*along_range, int(num) + 1)

#         if breakpoints[-1] == grid.cell[along_i, along_i]:
#             breakpoints[-1] = grid.cell[along_i, along_i] - grid.dcell[along_i, along_i]

#         if mode == "moving_slice":
#             return self._moving_slice_scan(grid, along_i, breakpoints)
#         elif mode == "as_is":
#             return self._asis_scan(grid, range_key, breakpoints, animation_kwargs=animation_kwargs, **kwargs)

#     def _asis_scan(self, grid, range_key, breakpoints, animation_kwargs=None, **kwargs):
#         """
#         Returns an animation containing multiple frames scaning along an axis.

#         Parameters
#         -----------
#         range_key: {'x_range', 'y_range', 'z_range'}
#             the key of the setting that is to be animated through the scan.
#         breakpoints: array-like
#             the discrete points of the scan
#         animation_kwargs: dict, optional
#             dictionary whose keys and values are directly passed to the animated method as kwargs and therefore
#             end up being passed to animation initialization.
#         **kwargs:
#             the rest of settings that you want to apply to overwrite the existing ones.

#             This settings apply to each plot and go directly to their initialization.

#         Returns
#         ----------
#         scan: sisl Animation
#             An animation representation of the scan
#         """
#         # Generate the plot using self as a template so that plots don't need
#         # to read data, just process it and show it differently.
#         # (If each plot read the grid, the memory requirements would be HUGE)
#         scan = self.animated(
#             {
#                 range_key: [[bp, breakpoints[i+1]] for i, bp in enumerate(breakpoints[:-1])]
#             },
#             fixed={**{key: val for key, val in self.settings.items() if key != range_key}, **kwargs, "grid": grid},
#             frame_names=[f'{bp:2f}' for bp in breakpoints],
#             **(animation_kwargs or {})
#         )

#         # Set all frames to the same colorscale, if it's a 2d representation
#         if len(self.get_setting("axes")) == 2:
#             cmin = 10**6; cmax = -10**6
#             for scan_im in scan:
#                 c = getattr(scan_im.data[0], "value", scan_im.data[0].z)
#                 cmin = min(cmin, np.min(c))
#                 cmax = max(cmax, np.max(c))
#             for scan_im in scan:
#                 scan_im.update_settings(crange=[cmin, cmax])

#         scan.get_figure()

#         scan.layout = self.layout

#         return scan

#     def _moving_slice_scan(self, grid, along_i, breakpoints):
#         import plotly.graph_objs as go
#         ax = along_i
#         displayed_axes = [i for i in range(3) if i != ax]
#         shape = np.array(grid.shape)[displayed_axes]
#         cmin = np.min(grid.grid)
#         cmax = np.max(grid.grid)
#         x_ax, y_ax = displayed_axes
#         x = np.linspace(0, grid.cell[x_ax, x_ax], grid.shape[x_ax])
#         y = np.linspace(0, grid.cell[y_ax, y_ax], grid.shape[y_ax])

#         fig = go.Figure(frames=[go.Frame(data=go.Surface(
#             x=x, y=y,
#             z=(bp * np.ones(shape)).T,
#             surfacecolor=np.squeeze(grid.cross_section(grid.index(bp, ax), ax).grid).T,
#             cmin=cmin, cmax=cmax,
#             ),
#             name=f'{bp:.2f}'
#             )
#             for bp in breakpoints])

#         # Add data to be displayed before animation starts
#         fig.add_traces(fig.frames[0].data)

#         def frame_args(duration):
#             return {
#                     "frame": {"duration": duration},
#                     "mode": "immediate",
#                     "fromcurrent": True,
#                     "transition": {"duration": duration, "easing": "linear"},
#                 }

#         sliders = [
#                     {
#                         "pad": {"b": 10, "t": 60},
#                         "len": 0.9,
#                         "x": 0.1,
#                         "y": 0,
#                         "steps": [
#                             {
#                                 "args": [[f.name], frame_args(0)],
#                                 "label": str(k),
#                                 "method": "animate",
#                             }
#                             for k, f in enumerate(fig.frames)
#                         ],
#                     }
#                 ]

#         def ax_title(ax): return f'{["X", "Y", "Z"][ax]} axis [Ang]'

#         # Layout
#         fig.update_layout(
#                 title=f'Grid scan along {["X", "Y", "Z"][ax]} axis',
#                 width=600,
#                 height=600,
#                 scene=dict(
#                             xaxis=dict(title=ax_title(x_ax)),
#                             yaxis=dict(title=ax_title(y_ax)),
#                             zaxis=dict(autorange=True, title=ax_title(ax)),
#                             aspectmode="data",
#                             ),
#                 updatemenus = [
#                     {
#                         "buttons": [
#                             {
#                                 "args": [None, frame_args(50)],
#                                 "label": "&#9654;", # play symbol
#                                 "method": "animate",
#                             },
#                             {
#                                 "args": [[None], frame_args(0)],
#                                 "label": "&#9724;", # pause symbol
#                                 "method": "animate",
#                             },
#                         ],
#                         "direction": "left",
#                         "pad": {"r": 10, "t": 70},
#                         "type": "buttons",
#                         "x": 0.1,
#                         "y": 0,
#                     }
#                 ],
#                 sliders=sliders
#         )

#         # We need to add an invisible trace so that the z axis stays with the correct range
#         fig.add_trace({"type": "scatter3d", "mode": "markers", "marker_size": 0.001, "x": [0, 0], "y": [0, 0], "z": [0, grid.cell[ax, ax]]})

#         return fig
