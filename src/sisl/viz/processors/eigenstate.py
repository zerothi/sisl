# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Optional, Union

import numpy as np

import sisl


def get_eigenstate(
    eigenstate: sisl.EigenstateElectron, i: int
) -> sisl.EigenstateElectron:
    """Gets the i-th wavefunction from the eigenstate.

    It takes into account if the info dictionary has an "index" key, which
    might be present for example if the eigenstate object does not contain
    the full set of wavefunctions, to indicate which wavefunctions are
    present.

    Parameters
    ----------
    eigenstate : sisl.EigenstateElectron
        The eigenstate from which to extract the wavefunction.
    i : int
        The index of the wavefunction to extract.
    """

    if "index" in eigenstate.info:
        wf_i = np.nonzero(eigenstate.info["index"] == i)[0]
        if len(wf_i) == 0:
            raise ValueError(
                f"Wavefunction with index {i} is not present in the eigenstate. Available indices: {eigenstate.info['index']}."
            )
        wf_i = wf_i[0]
    else:
        max_index = eigenstate.shape[0]
        if i > max_index:
            raise ValueError(
                f"Wavefunction with index {i} is not present in the eigenstate. Available range: [0, {max_index}]."
            )
        wf_i = i

    return eigenstate[wf_i]


def eigenstate_geometry(
    eigenstate: sisl.EigenstateElectron, geometry: Optional[sisl.Geometry] = None
) -> Union[sisl.Geometry, None]:
    """Returns the geometry associated with the eigenstate.

    Parameters
    ----------
    eigenstate : sisl.EigenstateElectron
        The eigenstate from which to extract the geometry.
    geometry : sisl.Geometry, optional
        If provided, this geometry is returned instead of the one associated. This is
        a way to force a given geometry when using this function.
    """
    if geometry is None:
        geometry = getattr(eigenstate, "parent", None)
        if geometry is not None and not isinstance(geometry, sisl.Geometry):
            geometry = getattr(geometry, "geometry", None)

    return geometry


def tile_if_k(
    geometry: sisl.Geometry,
    nsc: tuple[int, int, int],
    eigenstate: sisl.EigenstateElectron,
) -> sisl.Geometry:
    """Tiles the geometry if the eigenstate does not correspond to gamma.

    If we are calculating the wavefunction for any point other than gamma,
    the periodicity of the WF will be bigger than the cell. Therefore, if
    the user wants to see more than the unit cell, we need to generate the
    wavefunction for all the supercell.

    Parameters
    ----------
    geometry : sisl.Geometry
        The geometry for which the wavefunction was calculated.
    nsc : tuple[int, int, int]
        The number of supercells that are to be displayed in each direction.
    eigenstate : sisl.EigenstateElectron
        The eigenstate for which the wavefunction was calculated.
    """

    tiled_geometry = geometry

    k = eigenstate.info.get(
        "k", (1, 1, 1) if np.iscomplexobj(eigenstate.state) else (0, 0, 0)
    )

    for ax, sc_i in enumerate(nsc):
        if k[ax] != 0:
            tiled_geometry = tiled_geometry.tile(sc_i, ax)

    return tiled_geometry


def get_grid_nsc(
    nsc: tuple[int, int, int], eigenstate: sisl.EigenstateElectron
) -> tuple[int, int, int]:
    """Returns the supercell to display once the geometry is tiled.

    The geometry must be tiled if the eigenstate is not calculated at gamma,
    as done by `tile_if_k`. This function returns the number of supercells
    to display after that tiling.

    Parameters
    ----------
    nsc : tuple[int, int, int]
        The number of supercells to be display in each direction.
    eigenstate : sisl.EigenstateElectron
        The eigenstate for which the wavefunction was calculated.
    """
    k = eigenstate.info.get(
        "k", (1, 1, 1) if np.iscomplexobj(eigenstate.state) else (0, 0, 0)
    )

    return tuple(nx if kx == 0 else 1 for nx, kx in zip(nsc, k))


def create_wf_grid(
    eigenstate: sisl.EigenstateElectron,
    grid_prec: float = 0.2,
    grid: Optional[sisl.Grid] = None,
    geometry: Optional[sisl.Geometry] = None,
) -> sisl.Grid:
    """Creates a grid to display the wavefunction.

    Parameters
    ----------
    eigenstate : sisl.EigenstateElectron
        The eigenstate for which the wavefunction was calculated. The function uses

    grid_prec : float, optional
        The precision of the grid. The grid will be created with a spacing of
        `grid_prec` Angstroms.
    grid : sisl.Grid, optional
        If provided, this grid is returned instead of the one created. This is
        a way to force a given grid when using this function.
    geometry : sisl.Geometry, optional
        Geometry that will be associated to the grid. Required unless the grid
        is provided.
    """
    if grid is None:
        grid = sisl.Grid(grid_prec, geometry=geometry, dtype=eigenstate.state.dtype)

    return grid


def project_wavefunction(
    eigenstate: sisl.EigenstateElectron,
    grid_prec: float = 0.2,
    grid: Optional[sisl.Grid] = None,
    geometry: Optional[sisl.Geometry] = None,
) -> sisl.Grid:
    """Projects the wavefunction from an eigenstate into a grid.

    Parameters
    ----------
    eigenstate : sisl.EigenstateElectron
        The eigenstate for which the wavefunction was calculated.
    grid_prec : float, optional
        The precision of the grid. The grid will be created with a spacing of
        `grid_prec` Angstroms.
    grid : sisl.Grid, optional
        If provided, the wavefunction is inserted into this grid instead of creating
        a new one.
    geometry : sisl.Geometry, optional
        Geometry that will be associated to the grid. Required unless the grid
        is provided.
    """
    grid = create_wf_grid(eigenstate, grid_prec=grid_prec, grid=grid, geometry=geometry)

    # Ensure we are dealing with the cell gauge
    eigenstate.change_gauge("lattice")

    # Finally, insert the wavefunction values into the grid.
    sisl.physics.electron.wavefunction(
        eigenstate.state,
        grid,
        geometry=geometry,
        spinor=0,
    )

    return grid
