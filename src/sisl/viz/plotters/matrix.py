# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Literal, Union

import numpy as np

import sisl

from ..processors.matrix import get_orbital_sets_positions
from . import plot_actions


def draw_matrix_separators(
    line: Union[bool, dict],
    geometry: sisl.Geometry,
    matrix_mode: Literal["orbitals", "atoms"],
    separator_mode: Literal["orbitals", "atoms", "supercells"],
    draw_supercells: bool = True,
    showlegend: bool = True,
) -> list[dict]:
    """Returns the actions to draw separators in a matrix.

    Parameters
    ----------
    line:
        If False, no lines are drawn.
        If True, the default line style is used, which depends on `separator_mode`.
        If a dictionary, it must contain the line style.
    geometry:
        The geometry associated to the matrix.
    matrix_mode:
        Whether the elements of the matrix belong to orbitals or atoms.
    separator_mode:
        What the separators should separate.
    draw_supercells:
        Whether to draw separators for the whole matrix (not just the unit cell).
    showlegend:
        Show the separator lines in the legend.
    """
    # Orbital separators don't make sense if it is an atom matrix.
    if separator_mode == "orbitals" and matrix_mode == "atoms":
        return []

    # Sanitize the line argument
    if line is False:
        return []
    elif line is True:
        line = {}

    # Determine line styles from the defaults and the provided styles.
    default_line = {
        "orbitals": {"color": "black", "dash": "dot"},
        "atoms": {"color": "orange"},
        "supercells": {"color": "black"},
    }

    line = {**default_line[separator_mode], **line}

    # Initialize list that will hold the positions of all lines
    line_positions = []

    # Determine the shape of the matrix (how many rows)
    sc_len = geometry.no if matrix_mode == "orbitals" else geometry.na

    # If the user just wants to draw a given cell, this is effectively as if
    # the supercell was (1,1,1)
    n_supercells = geometry.n_s if draw_supercells else 1

    # Find out the line positions depending on what the separators must separate
    if separator_mode == "orbitals":
        species_lines = get_orbital_sets_positions(geometry.atoms)

        for atom_species, atom_first_o in zip(geometry.atoms.species, geometry.firsto):
            lines = species_lines[atom_species][1:]
            for line_pos in lines:
                line_positions.append(line_pos + atom_first_o - 0.5)
    elif separator_mode == "atoms":
        for atom_last_o in geometry.lasto[:-1]:
            line_pos = atom_last_o + 0.5
            line_positions.append(line_pos)
    elif separator_mode == "supercells":
        if n_supercells > 1:
            line_positions.append(float(sc_len) - 0.5)
    else:
        raise ValueError(
            "separator_mode must be one of 'orbitals', 'atoms', 'supercells'."
        )

    # If there are no lines to draw, exit
    if len(line_positions) == 0:
        return []

    # Horizontal lines: determine X and Y coordinates
    if separator_mode == "supercells":
        hor_x = hor_y = []
    else:
        hor_y = np.repeat(line_positions, 3)
        hor_y[2::3] = np.nan
        hor_x = np.tile((0, sc_len * n_supercells, np.nan), len(line_positions)) - 0.5

    # Vertical lines: determine X and Y coordinates (for all supercells)
    if n_supercells == 1:
        vert_line_positions = line_positions
    else:
        n_repeats = n_supercells - 1 if separator_mode == "supercells" else n_supercells

        vert_line_positions = np.tile(line_positions, n_repeats).reshape(n_repeats, -1)
        vert_line_positions += (np.arange(n_repeats) * sc_len).reshape(-1, 1)
        vert_line_positions = vert_line_positions.ravel()

    vert_x = np.repeat(vert_line_positions, 3)
    vert_x[2::3] = np.nan

    vert_y = np.tile((0, sc_len, np.nan), len(vert_line_positions)) - 0.5

    return [
        plot_actions.draw_line(
            x=np.concatenate([hor_x, vert_x]),
            y=np.concatenate([hor_y, vert_y]),
            line=line,
            name=f"{separator_mode} separators",
            showlegend=showlegend,
        )
    ]


def set_matrix_axes(
    matrix,
    geometry: sisl.Geometry,
    matrix_mode: Literal["orbitals", "atoms"],
    constrain_axes: bool = True,
    set_labels: bool = False,
) -> list[dict]:
    """Configure the axes of a matrix plot

    Parameters
    ----------
    matrix:
        The matrix that is plotted.
    geometry:
        The geometry associated to the matrix
    matrix_mode:
        Whether the elements of the matrix belong to orbitals or atoms.
    constrain_axes:
        Whether to try to constrain the axes to the domain of the matrix.
    set_labels:
        Whether to set the axis labels for each element of the matrix.
    """
    actions = []

    actions.append(plot_actions.set_axes_equal())

    x_kwargs = {}
    y_kwargs = {}

    if constrain_axes:
        x_kwargs["range"] = [-0.5, matrix.shape[1] - 0.5]
        x_kwargs["constrain"] = "domain"

        y_kwargs["range"] = [matrix.shape[0] - 0.5, -0.5]
        y_kwargs["constrain"] = "domain"

    if set_labels:
        if matrix_mode == "orbitals":
            atoms_ticks = []
            atoms = geometry.atoms.atom
            for i, atom in enumerate(atoms):
                atom_ticks = []
                atoms_ticks.append(atom_ticks)
                for orb in atom.orbitals:
                    atom_ticks.append(f"({orb.l}, {orb.m})")

            ticks = []
            for i, species in enumerate(geometry.atoms.species):
                ticks.extend([f"{i}: {orb}" for orb in atoms_ticks[species]])
        else:
            ticks = np.arange(matrix.shape[0]).astype(str)

        x_kwargs["ticktext"] = np.tile(ticks, geometry.n_s)
        x_kwargs["tickvals"] = np.arange(matrix.shape[1])

        y_kwargs["ticktext"] = ticks
        y_kwargs["tickvals"] = np.arange(matrix.shape[0])

    if len(x_kwargs) > 0:
        actions.append(plot_actions.set_axis(axis="x", **x_kwargs))
    if len(y_kwargs) > 0:
        actions.append(plot_actions.set_axis(axis="y", **y_kwargs))

    return actions
