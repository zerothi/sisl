# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, Optional, TypeVar, Union

import numpy as np

from sisl import BrillouinZone, Geometry
from sisl.typing import AtomsIndex
from sisl.viz.figure import Figure, get_figure
from sisl.viz.plotters import plot_actions as plot_actions
from sisl.viz.types import AtomArrowSpec, AtomsStyleSpec, Axes, Colorscale, StyleSpec

from ..plot import Plot
from ..plotters.cell import cell_plot_actions, get_ndim, get_z
from ..plotters.xarray import draw_xarray_xy
from ..processors.axes import sanitize_axes
from ..processors.coords import project_to_axes
from ..processors.geometry import (
    add_xyz_to_bonds_dataset,
    add_xyz_to_dataset,
    bonds_to_lines,
    find_all_bonds,
    get_sites_units,
    parse_atoms_style,
    sanitize_arrows,
    sanitize_atoms,
    sanitize_bonds_selection,
    sites_obj_to_geometry,
    stack_sc_data,
    style_bonds,
    tile_data_sc,
)
from ..processors.xarray import scale_variable, select


def _get_atom_mode(drawing_mode, ndim):
    if drawing_mode is None:
        if ndim == 3:
            return "balls"
        else:
            return "scatter"

    return drawing_mode


def _get_arrow_plottings(atoms_data, arrows, nsc=[1, 1, 1]):
    reps = np.prod(nsc)
    actions = []
    atoms_data = atoms_data.unstack("sc_atom")
    for arrows_spec in arrows:
        filtered = atoms_data.sel(atom=arrows_spec["atoms"])
        dxy = arrows_spec["data"][arrows_spec["atoms"]]
        dxy = np.tile(np.ravel(dxy), reps).reshape(-1, arrows_spec["data"].shape[-1])

        # If it is a 1D plot, make sure that the arrows have two coordinates, being 0 the second one.
        if dxy.shape[-1] == 1:
            dxy = np.array([dxy[:, 0], np.zeros_like(dxy[:, 0])]).T

        kwargs = {}
        kwargs["line"] = {
            "color": arrows_spec["color"],
            "width": arrows_spec["width"],
            "opacity": arrows_spec.get("opacity", 1),
        }
        kwargs["name"] = arrows_spec["name"]
        kwargs["arrowhead_scale"] = arrows_spec["arrowhead_scale"]
        kwargs["arrowhead_angle"] = arrows_spec["arrowhead_angle"]
        kwargs["annotate"] = arrows_spec.get("annotate", False)
        kwargs["scale"] = arrows_spec["scale"]

        if dxy.shape[-1] < 3:
            action = plot_actions.draw_arrows(
                x=np.ravel(filtered.x), y=np.ravel(filtered.y), dxy=dxy, **kwargs
            )
        else:
            action = plot_actions.draw_arrows_3D(
                x=np.ravel(filtered.x),
                y=np.ravel(filtered.y),
                z=np.ravel(filtered.z),
                dxyz=dxy,
                **kwargs,
            )
        actions.append(action)

    return actions


def _sanitize_scale(
    scale: float, ndim: int, ndim_scale: tuple[float, float, float] = (16, 16, 1)
):
    return ndim_scale[ndim - 1] * scale


def geometry_plot(
    geometry: Geometry,
    axes: Axes = ["x", "y", "z"],
    atoms: AtomsIndex = None,
    atoms_style: Sequence[AtomsStyleSpec] = [],
    atoms_scale: float = 1.0,
    atoms_colorscale: Optional[Colorscale] = None,
    drawing_mode: Literal["scatter", "balls", None] = None,
    bind_bonds_to_ats: bool = True,
    points_per_bond: int = 20,
    bonds_style: StyleSpec = {},
    bonds_scale: float = 1.0,
    bonds_colorscale: Optional[Colorscale] = None,
    show_atoms: bool = True,
    show_bonds: bool = True,
    show_cell: Literal["box", "axes", False] = "box",
    cell_style: StyleSpec = {},
    nsc: tuple[int, int, int] = (1, 1, 1),
    atoms_ndim_scale: tuple[float, float, float] = (16, 16, 1),
    bonds_ndim_scale: tuple[float, float, float] = (1, 1, 10),
    dataaxis_1d: Optional[Union[np.ndarray, Callable]] = None,
    arrows: Sequence[AtomArrowSpec] = (),
    backend="plotly",
) -> Figure:
    """Plots a geometry structure, with plentiful of customization options.

    Parameters
    ----------
    geometry:
        The geometry to plot.
    axes:
        The axes to project the geometry to.
    atoms:
        The atoms to plot. If None, all atoms are plotted.
    atoms_style:
        List of style specifications for the atoms. See the showcase notebooks for examples.
    atoms_scale:
        Scaling factor for the size of all atoms.
    atoms_colorscale:
        Colorscale to use for the atoms in case the color attribute is an array of values.
        If None, the default colorscale is used for each backend.
    drawing_mode:
        The method used to draw the atoms.
    bind_bonds_to_ats:
        Whether to display only bonds between atoms that are being displayed.
    points_per_bond:
        When the points are drawn using points instead of lines (e.g. in some frameworks
        to draw multicolor bonds), the number of points used per bond.
    bonds_style:
        Style specification for the bonds. See the showcase notebooks for examples.
    bonds_scale:
        Scaling factor for the width of all bonds.
    bonds_colorscale:
        Colorscale to use for the bonds in case the color attribute is an array of values.
        If None, the default colorscale is used for each backend.
    show_atoms:
        Whether to display the atoms.
    show_bonds:
        Whether to display the bonds.
    show_cell:
        Mode to display the cell. If False, the cell is not displayed.
    cell_style:
        Style specification for the cell. See the showcase notebooks for examples.
    nsc:
        Number of unit cells to display in each direction.
    atoms_ndim_scale:
        Scaling factor for the size of the atoms for different dimensionalities (1D, 2D, 3D).
    bonds_ndim_scale:
        Scaling factor for the width of the bonds for different dimensionalities (1D, 2D, 3D).
    dataaxis_1d:
        Only meaningful for 1D plots. The data to plot on the Y axis.
    arrows:
        List of arrow specifications to display. See the showcase notebooks for examples.
    backend:
        The backend to use to generate the figure.
    """

    # INPUTS ARE NOT GETTING PARSED BECAUSE WORKFLOWS RUN GET ON FINAL NODE
    # SO PARSING IS DELEGATED TO NODES.
    axes = sanitize_axes(axes)
    sanitized_atoms = sanitize_atoms(geometry, atoms=atoms)
    ndim = get_ndim(axes)
    z = get_z(ndim)

    # Atoms and bonds are processed in parallel paths, which means that one is able
    # to update without requiring the other. This means: 1) Faster updates if only one
    # of them needs to update; 2) It should be possible to run each path in a different
    # thread/process, potentially increasing speed.
    parsed_atom_style = parse_atoms_style(geometry, atoms_style=atoms_style)
    atoms_dataset = add_xyz_to_dataset(parsed_atom_style)
    atoms_filter = sanitized_atoms if show_atoms else []
    filtered_atoms = select(atoms_dataset, "atom", atoms_filter)
    tiled_atoms = tile_data_sc(filtered_atoms, nsc=nsc)
    sc_atoms = stack_sc_data(tiled_atoms, newname="sc_atom", dims=["atom"])
    projected_atoms = project_to_axes(
        sc_atoms, axes=axes, sort_by_depth=True, dataaxis_1d=dataaxis_1d
    )

    atoms_scale = _sanitize_scale(atoms_scale, ndim, atoms_ndim_scale)
    final_atoms = scale_variable(projected_atoms, "size", scale=atoms_scale)
    atom_mode = _get_atom_mode(drawing_mode, ndim)
    atom_plottings = draw_xarray_xy(
        data=final_atoms,
        x="x",
        y="y",
        z=z,
        width="size",
        what=atom_mode,
        colorscale=atoms_colorscale,
        set_axequal=True,
        name="Atoms",
    )

    # Here we start to process bonds
    bonds = find_all_bonds(geometry)
    show_bonds = show_bonds if ndim > 1 else False
    styled_bonds = style_bonds(bonds, bonds_style)
    bonds_dataset = add_xyz_to_bonds_dataset(styled_bonds)
    bonds_filter = sanitize_bonds_selection(
        bonds_dataset, sanitized_atoms, bind_bonds_to_ats, show_bonds
    )
    filtered_bonds = select(bonds_dataset, "bond_index", bonds_filter)
    tiled_bonds = tile_data_sc(filtered_bonds, nsc=nsc)

    projected_bonds = project_to_axes(tiled_bonds, axes=axes)
    bond_lines = bonds_to_lines(projected_bonds, points_per_bond=points_per_bond)

    bonds_scale = _sanitize_scale(bonds_scale, ndim, bonds_ndim_scale)
    final_bonds = scale_variable(bond_lines, "width", scale=bonds_scale)
    bond_plottings = draw_xarray_xy(
        data=final_bonds,
        x="x",
        y="y",
        z=z,
        set_axequal=True,
        name="Bonds",
        colorscale=bonds_colorscale,
    )

    # And now the cell
    show_cell = show_cell if ndim > 1 else False
    cell_plottings = cell_plot_actions(
        cell=geometry,
        show_cell=show_cell,
        cell_style=cell_style,
        axes=axes,
        dataaxis_1d=dataaxis_1d,
    )

    # And the arrows
    arrow_data = sanitize_arrows(
        geometry, arrows, atoms=sanitized_atoms, ndim=ndim, axes=axes
    )
    arrow_plottings = _get_arrow_plottings(projected_atoms, arrow_data, nsc=nsc)

    all_actions = plot_actions.combined(
        bond_plottings,
        atom_plottings,
        cell_plottings,
        arrow_plottings,
        composite_method=None,
    )

    return get_figure(backend=backend, plot_actions=all_actions)


class GeometryPlot(Plot):
    function = staticmethod(geometry_plot)

    @property
    def geometry(self):
        return self.nodes.inputs["geometry"]._output


_T = TypeVar("_T", list, tuple, dict)


def _sites_specs_to_atoms_specs(sites_specs: _T) -> _T:
    if isinstance(sites_specs, dict):
        if "sites" in sites_specs:
            sites_specs = sites_specs.copy()
            sites_specs["atoms"] = sites_specs.pop("sites")
        return sites_specs
    else:
        return type(sites_specs)(
            _sites_specs_to_atoms_specs(style_spec) for style_spec in sites_specs
        )


def sites_plot(
    sites_obj: BrillouinZone,
    axes: Axes = ["x", "y", "z"],
    sites: AtomsIndex = None,
    sites_style: Sequence[AtomsStyleSpec] = [],
    sites_scale: float = 1.0,
    sites_name: str = "Sites",
    sites_colorscale: Optional[Colorscale] = None,
    drawing_mode: Literal["scatter", "balls", "line", None] = None,
    show_cell: Literal["box", "axes", False] = False,
    cell_style: StyleSpec = {},
    nsc: tuple[int, int, int] = (1, 1, 1),
    sites_ndim_scale: tuple[float, float, float] = (1, 1, 1),
    dataaxis_1d: Optional[Union[np.ndarray, Callable]] = None,
    arrows: Sequence[AtomArrowSpec] = (),
    backend="plotly",
) -> Figure:
    """Plots sites from an object that can be parsed into a geometry.

    The only differences between this plot and a geometry plot is the naming of the inputs
    and the fact that there are no options to plot bonds.

    Parameters
    ----------
    sites_obj:
        The object to be converted to sites.
    axes:
        The axes to project the sites to.
    sites:
        The sites to plot. If None, all sites are plotted.
    sites_style:
        List of style specifications for the sites. See the showcase notebooks for examples.
    sites_scale:
        Scaling factor for the size of all sites.
    sites_name:
        Name to give to the trace that draws the sites.
    sites_colorscale:
        Colorscale to use for the sites in case the color attribute is an array of values.
        If None, the default colorscale is used for each backend.
    drawing_mode:
        The method used to draw the sites.
    show_cell:
        Mode to display the reciprocal cell. If False, the cell is not displayed.
    cell_style:
        Style specification for the reciprocal cell. See the showcase notebooks for examples.
    nsc:
        Number of unit cells to display in each direction.
    sites_ndim_scale:
        Scaling factor for the size of the sites for different dimensionalities (1D, 2D, 3D).
    dataaxis_1d:
        Only meaningful for 1D plots. The data to plot on the Y axis.
    arrows:
        List of arrow specifications to display. See the showcase notebooks for examples.
    backend:
        The backend to use to generate the figure.
    """

    # INPUTS ARE NOT GETTING PARSED BECAUSE WORKFLOWS RUN GET ON FINAL NODE
    # SO PARSING IS DELEGATED TO NODES.
    axes = sanitize_axes(axes)
    fake_geometry = sites_obj_to_geometry(sites_obj)
    sanitized_sites = sanitize_atoms(fake_geometry, atoms=sites)
    ndim = get_ndim(axes)
    z = get_z(ndim)

    # Process sites
    atoms_style = _sites_specs_to_atoms_specs(sites_style)
    parsed_sites_style = parse_atoms_style(fake_geometry, atoms_style=atoms_style)
    sites_dataset = add_xyz_to_dataset(parsed_sites_style)
    filtered_sites = select(sites_dataset, "atom", sanitized_sites)
    tiled_sites = tile_data_sc(filtered_sites, nsc=nsc)
    sc_sites = stack_sc_data(tiled_sites, newname="sc_atom", dims=["atom"])
    sites_units = get_sites_units(sites_obj)
    projected_sites = project_to_axes(
        sc_sites,
        axes=axes,
        sort_by_depth=True,
        dataaxis_1d=dataaxis_1d,
        cartesian_units=sites_units,
    )

    sites_scale = _sanitize_scale(sites_scale, ndim, sites_ndim_scale)
    final_sites = scale_variable(projected_sites, "size", scale=sites_scale)
    sites_mode = _get_atom_mode(drawing_mode, ndim)
    site_plottings = draw_xarray_xy(
        data=final_sites,
        x="x",
        y="y",
        z=z,
        width="size",
        what=sites_mode,
        colorscale=sites_colorscale,
        set_axequal=True,
        name=sites_name,
    )

    # And now the cell
    show_cell = show_cell if ndim > 1 else show_cell
    cell_plottings = cell_plot_actions(
        cell=fake_geometry,
        show_cell=show_cell,
        cell_style=cell_style,
        axes=axes,
        dataaxis_1d=dataaxis_1d,
    )

    # And the arrows
    atom_arrows = _sites_specs_to_atoms_specs(arrows)
    arrow_data = sanitize_arrows(
        fake_geometry, atom_arrows, atoms=sanitized_sites, ndim=ndim, axes=axes
    )
    arrow_plottings = _get_arrow_plottings(projected_sites, arrow_data, nsc=nsc)

    all_actions = plot_actions.combined(
        site_plottings, cell_plottings, arrow_plottings, composite_method=None
    )

    return get_figure(backend=backend, plot_actions=all_actions)


class SitesPlot(Plot):
    function = staticmethod(sites_plot)
