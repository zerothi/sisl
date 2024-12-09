# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Optional

import numpy as np

import sisl
from sisl._core.geometry import Geometry
from sisl.grid import Grid
from sisl.physics.electron import EigenstateElectron, wavefunction
from sisl.physics.hamiltonian import Hamiltonian
from sisl.physics.spin import Spin
from sisl.viz.nodes.data_sources.file.siesta import FileDataSIESTA
from sisl.viz.nodes.node import Node

from .grid import GridDataNode


@Node.from_func
def get_ith_eigenstate(eigenstate: EigenstateElectron, i: int):
    """Gets the ith eigenstate.

    This is useful because an EigenstateElectron contains all the eigenstates.
    Sometimes a post-processing tool calculates only a subset of eigenstates,
    and this is what you have inside the EigenstateElectron.
    therefore getting eigenstate[0] does not mean that

    Parameters
    ----------
    eigenstate : EigenstateElectron
        The object containing all eigenstates.
    i : int
        The index of the eigenstate to get.

    Returns
    ----------
    EigenstateElectron
        The ith eigenstate.
    """

    if "index" in eigenstate.info:
        wf_i = np.nonzero(eigenstate.info["index"] == i)[0]
        if len(wf_i) == 0:
            raise ValueError(
                f"Wavefunction with index {i} is not present in the eigenstate. Available indices: {eigenstate.info['index']}."
            )
        wf_i = wf_i[0]
    else:
        max_index = len(eigenstate)
        if i > max_index:
            raise ValueError(
                f"Wavefunction with index {i} is not present in the eigenstate. Available range: [0, {max_index}]."
            )
        wf_i = i

    return eigenstate[wf_i]


class WavefunctionDataNode(GridDataNode): ...


@WavefunctionDataNode.register
def eigenstate_wf(
    eigenstate: EigenstateElectron,
    i: int,
    grid: Optional[Grid] = None,
    geometry: Optional[Geometry] = None,
    k=[0, 0, 0],
    grid_prec: float = 0.2,
    spin: Optional[Spin] = None,
):
    if geometry is None:
        if isinstance(eigenstate.parent, Geometry):
            geometry = eigenstate.parent
        else:
            geometry = getattr(eigenstate.parent, "geometry", None)
    if geometry is None:
        raise ValueError(
            "No geometry was provided and we need it the basis orbitals to build the wavefunctions from the coefficients!"
        )

    if spin is None:
        spin = getattr(eigenstate.parent, "spin", Spin())

    if grid is None:
        dtype = eigenstate.dtype
        grid = Grid(grid_prec, geometry=geometry, dtype=dtype)

    # GridPlot's after_read basically sets the x_range, y_range and z_range options
    # which need to know what the grid is, that's why we are calling it here
    # super()._after_read()

    # Get the particular WF that we want from the eigenstate object
    wf_state = get_ith_eigenstate(eigenstate, i)

    # Ensure we are dealing with the lattice gauge
    wf_state.change_gauge("lattice")

    # Finally, insert the wavefunction values into the grid.
    wavefunction(wf_state.state, grid, geometry=geometry, k=k, spinor=0, spin=spin)

    return grid


@WavefunctionDataNode.register
def hamiltonian_wf(
    H: Hamiltonian,
    i: int,
    grid: Optional[Grid] = None,
    geometry: Optional[Geometry] = None,
    k=[0, 0, 0],
    grid_prec: float = 0.2,
    spin: int = 0,
):
    eigenstate = H.eigenstate(k=k, spin=spin)

    return eigenstate_wf(eigenstate, i, grid, geometry, k, grid_prec, spin)


@WavefunctionDataNode.register
def wfsx_wf(
    fdf,
    wfsx_file,
    i: int,
    grid: Optional[Grid] = None,
    geometry: Optional[Geometry] = None,
    k=[0, 0, 0],
    grid_prec: float = 0.2,
    spin: int = 0,
):
    fdf = FileDataSIESTA(path=fdf)
    geometry = fdf.read_geometry(output=True)

    # Get the WFSX file. If not provided, it is inferred from the fdf.
    wfsx = FileDataSIESTA(fdf=fdf, path=wfsx_file, cls=sisl.io.wfsxSileSiesta)

    # Now that we have the file, read the spin size and create a fake Hamiltonian
    sizes = wfsx.read_sizes()
    H = sisl.Hamiltonian(geometry, dim=sizes.nspin)

    # Read the wfsx again, this time passing the Hamiltonian as the parent
    wfsx = sisl.get_sile(wfsx.file, parent=H)

    # Try to find the eigenstate that we need
    eigenstate = wfsx.read_eigenstate(k=k, spin=spin)
    if eigenstate is None:
        # We have not found it.
        raise ValueError(f"A state with k={k} was not found in file {wfsx.file}.")

    return eigenstate_wf(eigenstate, i, grid, geometry, k, grid_prec)
