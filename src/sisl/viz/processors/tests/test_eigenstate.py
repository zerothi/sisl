# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

import sisl
from sisl.viz.processors.eigenstate import (
    create_wf_grid,
    eigenstate_geometry,
    get_eigenstate,
    get_grid_nsc,
    project_wavefunction,
    tile_if_k,
)

pytestmark = [pytest.mark.viz, pytest.mark.processors]


@pytest.fixture(scope="module", params=["Gamma", "X"])
def k(request):
    if request.param == "Gamma":
        return (0, 0, 0)
    elif request.param == "X":
        return (0.5, 0, 0)


@pytest.fixture(scope="module")
def graphene():
    r = np.linspace(0, 3.5, 50)
    f = np.exp(-r)

    orb = sisl.AtomicOrbital("2pzZ", (r, f))
    return sisl.geom.graphene(orthogonal=True, atoms=sisl.Atom(6, orb))


@pytest.fixture(scope="module")
def eigenstate(k, graphene):
    # Create a simple graphene tight binding Hamiltonian
    H = sisl.Hamiltonian(graphene)
    H.construct([(0.1, 1.44), (0, -2.7)])

    return H.eigenstate(k=k)


def test_get_eigenstate(eigenstate, graphene):
    sel_eigenstate = get_eigenstate(eigenstate, 2)

    assert sel_eigenstate.state.shape == (1, graphene.no)
    assert np.allclose(sel_eigenstate.state, eigenstate.state[2])

    eigenstate = eigenstate.copy()

    eigenstate.info["index"] = np.array([0, 3, 1, 2])

    sel_eigenstate = get_eigenstate(eigenstate, 2)

    assert sel_eigenstate.state.shape == (1, graphene.no)
    assert np.allclose(sel_eigenstate.state, eigenstate.state[3])


def test_eigenstate_geometry(eigenstate, graphene):
    # It should give us the geometry associated with the eigenstate
    assert eigenstate_geometry(eigenstate) is graphene

    # Unless we provide a geometry
    graphene_copy = graphene.copy()
    assert eigenstate_geometry(eigenstate, graphene_copy) is graphene_copy


def test_tile_if_k(eigenstate, graphene):
    # If the eigenstate is calculated at gamma, we don't need to tile
    tiled_geometry = tile_if_k(graphene, (2, 2, 2), eigenstate)

    if eigenstate.info["k"] == (0, 0, 0):
        # If the eigenstate is calculated at gamma, we don't need to tile
        assert tiled_geometry is graphene
    elif eigenstate.info["k"] == (0.5, 0, 0):
        # If the eigenstate is calculated at X, we need to tile
        # but only the first lattice vector.
        assert tiled_geometry is not graphene
        assert np.allclose(tiled_geometry.cell, graphene.cell * (2, 1, 1))


def test_get_grid_nsc(eigenstate):
    grid_nsc = get_grid_nsc((2, 2, 2), eigenstate)

    if eigenstate.info["k"] == (0, 0, 0):
        assert grid_nsc == (2, 2, 2)
    elif eigenstate.info["k"] == (0.5, 0, 0):
        assert grid_nsc == (1, 2, 2)


def test_create_wf_grid(eigenstate, graphene):
    new_graphene = graphene.copy()
    grid = create_wf_grid(eigenstate, grid_prec=0.2, geometry=new_graphene)

    assert isinstance(grid, sisl.Grid)
    assert grid.geometry is new_graphene

    # Check that the datatype is correct
    if eigenstate.info["k"] == (0, 0, 0):
        assert grid.grid.dtype == np.float64
    else:
        assert grid.grid.dtype == np.complex128

    # Check that the grid precision is right.
    assert np.allclose(np.linalg.norm(grid.dcell, axis=1), 0.2, atol=0.01)

    provided_grid = sisl.Grid(0.2, geometry=new_graphene, dtype=np.float64)

    grid = create_wf_grid(eigenstate, grid=provided_grid)

    assert grid is provided_grid


def test_project_wavefunction(eigenstate, graphene):
    k = eigenstate.info["k"]

    grid = project_wavefunction(eigenstate[2], geometry=graphene)

    assert isinstance(grid, sisl.Grid)

    # Check that the datatype is correct
    if k == (0, 0, 0):
        assert grid.grid.dtype == np.float64
    else:
        assert grid.grid.dtype == np.complex128

    # Check that the grid is not empty
    assert not np.allclose(grid.grid, 0)
