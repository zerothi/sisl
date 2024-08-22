# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from sisl.viz.types import OrbitalStyleQuery

from ..data import PDOSData
from ..figure import Figure, get_figure
from ..plotters.xarray import draw_xarray_xy
from ..processors.data import accept_data
from ..processors.logic import swap
from ..processors.orbital import get_orbital_queries_manager, reduce_orbital_data
from ..processors.xarray import filter_energy_range, scale_variable
from .orbital_groups_plot import OrbitalGroupsPlot


def pdos_plot(
    pdos_data: PDOSData,
    groups: Sequence[OrbitalStyleQuery] = [{"name": "DOS"}],
    Erange: tuple[float, float] = (-2, 2),
    E_axis: Literal["x", "y"] = "x",
    line_mode: Literal["line", "scatter", "area_line"] = "line",
    line_scale: float = 1.0,
    backend: str = "plotly",
) -> Figure:
    """Plot the projected density of states.

    Parameters
    ----------
    pdos_data:
        The object containing the raw PDOS data (individual PDOS for each orbital/spin).
    groups:
        List of orbital specifications to filter and accumulate the PDOS.
        The contribution of each group will be displayed in a different line.
        See showcase notebook for examples.
    Erange:
        The energy range to plot.
    E_axis:
        Axis to project the energies.
    line_mode:
        Mode used to draw the PDOS lines.
    line_scale:
        Scaling factor for the width of all lines.
    backend:
        The backend to generate the figure.
    """
    pdos_data = accept_data(pdos_data, cls=PDOSData, check=True)

    E_PDOS = filter_energy_range(pdos_data, Erange=Erange, E0=0)

    orbital_manager = get_orbital_queries_manager(pdos_data)
    groups_data = reduce_orbital_data(
        E_PDOS,
        groups=groups,
        orb_dim="orb",
        spin_dim="spin",
        sanitize_group=orbital_manager,
        group_vars=("color", "size", "dash"),
        groups_dim="group",
        drop_empty=True,
        spin_reduce=np.sum,
    )

    # Determine what goes on each axis
    x = "E" if E_axis == "x" else "PDOS"
    y = "E" if E_axis == "y" else "PDOS"

    dependent_axis = swap(E_axis, ("x", "y"))

    # A PlotterNode gets the processed data and creates abstract actions (backend agnostic)
    # that should be performed on the figure. The output of this node
    # must be fed to a figure (backend specific).
    final_groups_data = scale_variable(
        groups_data, var="size", scale=line_scale, default_value=1
    )
    plot_actions = draw_xarray_xy(
        data=final_groups_data,
        x=x,
        y=y,
        width="size",
        name="group",
        what=line_mode,
        dependent_axis=dependent_axis,
    )

    return get_figure(backend=backend, plot_actions=plot_actions)


class PdosPlot(OrbitalGroupsPlot):
    function = staticmethod(pdos_plot)

    def split_DOS(self, on="species", only=None, exclude=None, clean=True, **kwargs):
        """
        Splits the density of states to the different contributions.

        Parameters
        --------
        on: str, {"species", "atoms", "Z", "orbitals", "n", "l", "m", "zeta", "spin"}, or list of str
            the parameter to split along.
            Note that you can combine parameters with a "+" to split along multiple parameters
            at the same time. You can get the same effect also by passing a list.
        only: array-like, optional
            if desired, the only values that should be plotted out of
            all of the values that come from the splitting.
        exclude: array-like, optional
            values that should not be plotted
        clean: boolean, optional
            whether the plot should be cleaned before drawing.
            If False, all the requests that come from the method will
            be drawn on top of what is already there.
        **kwargs:
            keyword arguments that go directly to each request.

            This is useful to add extra filters. For example:
            `plot.split_DOS(on="orbitals", species=["C"])`
            will split the PDOS on the different orbitals but will take
            only those that belong to carbon atoms.

        Examples
        -----------

        >>> plot = H.plot.pdos()
        >>>
        >>> # Split the DOS in n and l but show only the DOS from Au
        >>> # Also use "Au $ns" as a template for the name, where $n will
        >>> # be replaced by the value of n.
        >>> plot.split_DOS(on="n+l", species=["Au"], name="Au $ns")
        """
        return self.split_groups(
            on=on, only=only, exclude=exclude, clean=clean, **kwargs
        )
