# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.sparse import spmatrix

import sisl

from ..figure import Figure, get_figure
from ..plot import Plot
from ..plotters.grid import draw_grid, draw_grid_arrows
from ..plotters.matrix import draw_matrix_separators, set_matrix_axes
from ..plotters.plot_actions import combined
from ..processors.matrix import (
    determine_color_midpoint,
    get_geometry_from_matrix,
    get_matrix_mode,
    matrix_as_array,
    sanitize_matrix_arrows,
)
from ..types import Colorscale


def atomic_matrix_plot(
    matrix: Union[
        np.ndarray, sisl.SparseCSR, sisl.SparseAtom, sisl.SparseOrbital, spmatrix
    ],
    dim: int = 0,
    isc: Optional[int] = None,
    fill_value: Optional[float] = None,
    geometry: Union[sisl.Geometry, None] = None,
    atom_lines: Union[bool, dict] = False,
    orbital_lines: Union[bool, dict] = False,
    sc_lines: Union[bool, dict] = False,
    color_pixels: bool = True,
    colorscale: Optional[Colorscale] = "RdBu",
    crange: Optional[tuple[float, float]] = None,
    cmid: Optional[float] = None,
    text: Optional[str] = None,
    textfont: Optional[dict] = {},
    set_labels: bool = False,
    constrain_axes: bool = True,
    arrows: list[dict] = [],
    backend: str = "plotly",
) -> Figure:
    """Plots a (possibly sparse) matrix where rows and columns are either orbitals or atoms.

    Parameters
    ----------
    matrix:
        the matrix, either as a numpy array or as a sisl sparse matrix.
    dim:
        If the matrix has a third dimension (e.g. spin), which index to
        plot in that third dimension.
    isc:
        If the matrix contains data for an auxiliary supercell, the index of the
        cell to plot. If None, the whole matrix is plotted.
    fill_value:
        If the matrix is sparse, the value to use for the missing entries.
    geometry:
        Only needed if the matrix does not contain a geometry (e.g. it is a numpy array)
        and separator lines or labels are requested.
    atom_lines:
        If a boolean, whether to draw lines separating atom blocks, using default styles.
        If a dict, draws the lines with the specified plotly line styles.
    orbital_lines:
        If a boolean, whether to draw lines separating blocks of orbital sets, using default styles.
        If a dict, draws the lines with the specified plotly line styles.
    sc_lines:
        If a boolean, whether to draw lines separating the supercells, using default styles.
        If a dict, draws the lines with the specified plotly line styles.
    color_pixels:
        Whether to color the pixels of the matrix according to the colorscale.
    colorscale:
        The colorscale to use to color the pixels.
    crange:
        The minimum and maximum values of the colorscale.
    cmid:
        The midpoint of the colorscale. If ``crange`` is provided, this is ignored.

        If None and crange is also None, the midpoint
        is set to 0 if the data contains both positive and negative values.
    text:
        If provided, show text of pixel value with the specified format.
        E.g. text=".3f" shows the value with three decimal places.
    textfont:
        The font to use for the text.
        This is a dictionary that may contain the keys "family", "size", "color".
    set_labels:
        Whether to set the axes labels to the atom/orbital that each row and column corresponds to.
        For orbitals the labels will be of the form "Atom: (l, m)", where `Atom` is the index of
        the atom and l and m are the quantum numbers of the orbital.
    constrain_axes:
        Whether to set the ranges of the axes to exactly fit the matrix.
    backend:
        The backend to use for plotting.
    """

    geometry = get_geometry_from_matrix(matrix, geometry)
    mode = get_matrix_mode(matrix)

    matrix_array = matrix_as_array(matrix, dim=dim, isc=isc, fill_value=fill_value)

    color_midpoint = determine_color_midpoint(matrix, cmid=cmid, crange=crange)

    matrix_actions = draw_grid(
        matrix_array,
        crange=crange,
        cmid=color_midpoint,
        color_pixels_2d=color_pixels,
        colorscale=colorscale,
        coloraxis_name="matrix_vals",
        textformat=text,
        textfont=textfont,
    )

    arrows = sanitize_matrix_arrows(arrows)

    arrow_actions = draw_grid_arrows(matrix_array, arrows)

    draw_supercells = isc is None

    axes_actions = set_matrix_axes(
        matrix_array,
        geometry,
        matrix_mode=mode,
        constrain_axes=constrain_axes,
        set_labels=set_labels,
    )

    sc_lines_actions = draw_matrix_separators(
        sc_lines,
        geometry,
        matrix_mode=mode,
        separator_mode="supercells",
        draw_supercells=draw_supercells,
        showlegend=False,
    )

    atom_lines_actions = draw_matrix_separators(
        atom_lines,
        geometry,
        matrix_mode=mode,
        separator_mode="atoms",
        draw_supercells=draw_supercells,
        showlegend=False,
    )

    orbital_lines_actions = draw_matrix_separators(
        orbital_lines,
        geometry,
        matrix_mode=mode,
        separator_mode="orbitals",
        draw_supercells=draw_supercells,
        showlegend=False,
    )

    all_actions = combined(
        matrix_actions,
        arrow_actions,
        orbital_lines_actions,
        atom_lines_actions,
        sc_lines_actions,
        axes_actions,
    )

    return get_figure(backend, all_actions)


class AtomicMatrixPlot(Plot):
    function = staticmethod(atomic_matrix_plot)
