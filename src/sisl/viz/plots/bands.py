# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Optional

import numpy as np

from sisl.viz.types import Colorscale, OrbitalQueries, StyleSpec

from ..data.bands import BandsData
from ..figure import Figure, get_figure
from ..plot import Plot
from ..plotters.plot_actions import combined
from ..plotters.xarray import draw_xarray_xy
from ..plotutils import random_color
from ..processors.bands import calculate_gap, draw_gaps, filter_bands, style_bands
from ..processors.data import accept_data
from ..processors.orbital import get_orbital_queries_manager, reduce_orbital_data
from ..processors.xarray import scale_variable
from .orbital_groups_plot import OrbitalGroupsPlot


def _default_random_color(x):
    return x.get("color") or random_color()


def _group_traces(actions, group_legend: bool = True):

    if not group_legend:
        return actions

    seen_groups = []

    new_actions = []
    for action in actions:
        if action["method"].startswith("draw_"):
            group = action["kwargs"].get("name")
            action = action.copy()
            action["kwargs"]["legendgroup"] = group

            if group in seen_groups:
                action["kwargs"]["showlegend"] = False
            else:
                seen_groups.append(group)

        new_actions.append(action)

    return new_actions


def bands_plot(
    bands_data: BandsData,
    Erange: Optional[tuple[float, float]] = None,
    E0: float = 0.0,
    E_axis: Literal["x", "y"] = "y",
    bands_range: Optional[tuple[int, int]] = None,
    spin: Optional[Literal[0, 1]] = None,
    bands_style: StyleSpec = {
        "color": "black",
        "width": 1,
        "opacity": 1,
        "dash": "solid",
    },
    spindown_style: StyleSpec = {"color": "blue", "width": 1},
    colorscale: Optional[Colorscale] = None,
    gap: bool = False,
    gap_tol: float = 0.01,
    gap_color: str = "red",
    gap_marker: dict = {"size": 7},
    direct_gaps_only: bool = False,
    custom_gaps: Sequence[dict] = [],
    line_mode: Literal["line", "scatter", "area_line"] = "line",
    group_legend: bool = True,
    backend: str = "plotly",
) -> Figure:
    """Plots band structure energies, with plentiful of customization options.

    Parameters
    ----------
    bands_data :
        The object containing the data to plot.
    Erange :
        The energy range to plot.
        If None, the range is determined by `bands_range`.
    E0 :
        The energy reference.
    E_axis :
        Axis to plot the energies.
    bands_range :
        The bands to plot. Only used if `Erange` is None.
        If None, the 15 bands above and below the Fermi level are plotted.
    spin :
        Which spin channel to display. Only meaningful for spin-polarized calculations.
        If None and the calculation is spin polarized, both are plotted.
    bands_style :
        Styling attributes for bands.
    spindown_style :
        Styling attributes for the spin down bands (if present). Any missing attribute
        will be taken from `bands_style`.
    colorscale :
        Colorscale to use for the bands in case the color attribute is an array of values.
        If None, the default colorscale is used for each backend.
    gap :
        Whether to display the gap.
    gap_tol :
        Tolerance in k for determining whether two gaps are the same.
    gap_color :
        Color of the gap.
    gap_marker :
        Marker styles for the gap (as `plotly` marker's styles).
    direct_gaps_only :
        Whether to only display direct gaps.
    custom_gaps :
        List of custom gaps to display. See the showcase notebooks for examples.
    line_mode :
        The method used to draw the band lines.
    group_legend :
        Whether to group all bands in the legend to show a single legend item.

        If the bands are spin polarized, bands are grouped by spin channel.
    backend :
        The backend to use to generate the figure.
    """

    bands_data = accept_data(bands_data, cls=BandsData, check=True)

    # Filter the bands
    filtered_bands = filter_bands(
        bands_data,
        Erange=Erange,
        E0=E0,
        bands_range=bands_range,
        spin=spin,
    )

    # Add the styles
    styled_bands = style_bands(
        filtered_bands,
        bands_style=bands_style,
        spindown_style=spindown_style,
        group_legend=group_legend,
    )

    # Determine what goes on each axis
    x = "E" if E_axis == "x" else "k"
    y = "E" if E_axis == "y" else "k"

    # Get the actions to plot lines
    bands_plottings = draw_xarray_xy(
        data=styled_bands,
        x=x,
        y=y,
        set_axrange=True,
        what=line_mode,
        name="line_name",
        colorscale=colorscale,
        dependent_axis=E_axis,
    )
    grouped_bands_plottings = _group_traces(bands_plottings, group_legend=group_legend)

    # Gap calculation
    gap_info = calculate_gap(filtered_bands)
    # Plot it if the user has asked for it.
    gaps_plottings = draw_gaps(
        bands_data,
        gap,
        gap_info,
        gap_tol,
        gap_color,
        gap_marker,
        direct_gaps_only,
        custom_gaps,
        E_axis=E_axis,
    )

    all_plottings = combined(
        grouped_bands_plottings, gaps_plottings, composite_method=None
    )

    return get_figure(backend=backend, plot_actions=all_plottings)


# I keep the fatbands plot here so that one can see how similar they are.
# I am yet to find a nice solution for extending workflows.
def fatbands_plot(
    bands_data: BandsData,
    Erange: Optional[tuple[float, float]] = None,
    E0: float = 0.0,
    E_axis: Literal["x", "y"] = "y",
    bands_range: Optional[tuple[int, int]] = None,
    spin: Optional[Literal[0, 1]] = None,
    bands_style: StyleSpec = {"color": "black", "width": 1, "opacity": 1},
    spindown_style: StyleSpec = {"color": "blue", "width": 1},
    gap: bool = False,
    gap_tol: float = 0.01,
    gap_color: str = "red",
    gap_marker: dict = {"size": 7},
    direct_gaps_only: bool = False,
    custom_gaps: Sequence[dict] = [],
    bands_mode: Literal["line", "scatter", "area_line"] = "line",
    bands_group_legend: bool = True,
    # Fatbands inputs
    groups: OrbitalQueries = [],
    fatbands_var: str = "norm2",
    fatbands_mode: Literal["line", "scatter", "area_line"] = "area_line",
    fatbands_scale: float = 1.0,
    backend: str = "plotly",
) -> Figure:
    """Plots band structure energies showing the contribution of orbitals to each state.

    Parameters
    ----------
    bands_data :
        The object containing the data to plot.
    Erange :
        The energy range to plot.
        If None, the range is determined by `bands_range`.
    E0 :
        The energy reference.
    E_axis :
        Axis to plot the energies.
    bands_range :
        The bands to plot. Only used if `Erange` is None.
        If None, the 15 bands above and below the Fermi level are plotted.
    spin :
        Which spin channel to display. Only meaningful for spin-polarized calculations.
        If None and the calculation is spin polarized, both are plotted.
    bands_style :
        Styling attributes for bands.
    spindown_style :
        Styling attributes for the spin down bands (if present). Any missing attribute
        will be taken from `bands_style`.
    gap :
        Whether to display the gap.
    gap_tol :
        Tolerance in k for determining whether two gaps are the same.
    gap_color :
        Color of the gap.
    gap_marker :
        Marker styles for the gap (as `plotly` marker's styles).
    direct_gaps_only :
        Whether to only display direct gaps.
    custom_gaps :
        List of custom gaps to display. See the showcase notebooks for examples.
    bands_mode :
        The method used to draw the band lines.
    bands_group_legend :
        Whether to group all bands in the legend to show a single legend item.

        If the bands are spin polarized, bands are grouped by spin channel.
    groups :
        Orbital groups to plots. See showcase notebook for examples.
    fatbands_var :
        The variable to use from bands_data to determine the width of the fatbands.
        This variable must have as coordinates ``(k, band, orb, [spin])``.
    fatbands_mode :
        The method used to draw the fatbands.
    fatbands_scale :
        Factor that scales the size of all fatbands.
    backend :
        The backend to use to generate the figure.
    """
    bands_data = accept_data(bands_data, cls=BandsData, check=True)

    # Filter the bands
    filtered_bands = filter_bands(
        bands_data, Erange=Erange, E0=E0, bands_range=bands_range, spin=spin
    )

    # Add the styles
    styled_bands = style_bands(
        filtered_bands,
        bands_style=bands_style,
        spindown_style=spindown_style,
        group_legend=bands_group_legend,
    )

    # Process fatbands
    orbital_manager = get_orbital_queries_manager(
        bands_data,
        key_gens={
            "color": _default_random_color,
        },
    )
    fatbands_data = reduce_orbital_data(
        filtered_bands,
        groups=groups,
        orb_dim="orb",
        spin_dim="spin",
        sanitize_group=orbital_manager,
        group_vars=("color", "dash", "opacity"),
        groups_dim="group",
        drop_empty=True,
        spin_reduce=False,
    )
    scaled_fatbands_data = scale_variable(
        fatbands_data,
        var=fatbands_var,
        scale=fatbands_scale,
        default_value=1,
        allow_not_present=True,
    )

    # Determine what goes on each axis
    x = "E" if E_axis == "x" else "k"
    y = "E" if E_axis == "y" else "k"

    sanitized_fatbands_mode = "none" if groups == [] else fatbands_mode

    # Get the actions to plot lines
    fatbands_plottings = draw_xarray_xy(
        data=scaled_fatbands_data,
        x=x,
        y=y,
        color="color",
        width=fatbands_var,
        what=sanitized_fatbands_mode,
        dependent_axis=E_axis,
        name="group",
    )
    grouped_fatbands_plottings = _group_traces(fatbands_plottings)
    bands_plottings = draw_xarray_xy(
        data=styled_bands,
        x=x,
        y=y,
        set_axrange=True,
        what=bands_mode,
        name="line_name",
        dependent_axis=E_axis,
    )
    grouped_bands_plottings = _group_traces(
        bands_plottings, group_legend=bands_group_legend
    )

    # Gap calculation
    gap_info = calculate_gap(filtered_bands)
    # Plot it if the user has asked for it.
    gaps_plottings = draw_gaps(
        bands_data,
        gap,
        gap_info,
        gap_tol,
        gap_color,
        gap_marker,
        direct_gaps_only,
        custom_gaps,
        E_axis=E_axis,
    )

    all_plottings = combined(
        grouped_fatbands_plottings,
        grouped_bands_plottings,
        gaps_plottings,
        composite_method=None,
    )

    return get_figure(backend=backend, plot_actions=all_plottings)


class BandsPlot(Plot):
    function = staticmethod(bands_plot)


class FatbandsPlot(OrbitalGroupsPlot):
    function = staticmethod(fatbands_plot)
