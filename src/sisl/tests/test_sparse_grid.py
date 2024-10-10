# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
import pytest

import sisl
from sisl import Grid
from sisl._sparse_grid_ops import transpose_raveled_index


@pytest.fixture
def geometry():

    r = np.linspace(0, 3.5, 50)
    f = np.exp(-r)

    orb = sisl.AtomicOrbital("2pzZ", (r, f))
    geom = sisl.geom.graphene(orthogonal=True, atoms=sisl.Atom(6, orb))
    geom.set_nsc([3, 5, 1])

    return geom


@pytest.fixture
def grid_shape():
    return (8, 10, 12)


@pytest.fixture
def psi_values(geometry, grid_shape):
    return geometry._orbital_values(grid_shape)


@pytest.fixture
def H(geometry):
    H = sisl.Hamiltonian(geometry)
    H.construct(
        [(0.1, 1.44), (0, -2.7)],
    )
    return H


def test_transpose_raveled():
    """Tests the cython implemented function that transposes
    raveled indices."""

    # Define the reference function, which calls numpy
    def transpose_raveled_numpy(index, grid_shape, new_order):
        grid_shape = np.array(grid_shape)

        unraveled = np.unravel_index(index, grid_shape)
        unraveled = np.array(unraveled)
        new_grid_shape = grid_shape[new_order]
        new_unraveled = unraveled[new_order]

        return np.ravel_multi_index(new_unraveled, new_grid_shape)

    grid_shape = np.array([3, 4, 5], dtype=np.int32)
    grid_size = np.prod(grid_shape)
    # Move the first axis to the last dimension.
    new_order = np.array([1, 2, 0], dtype=np.int32)

    for i in range(grid_size):
        transposed = transpose_raveled_index(i, grid_shape, new_order)
        transposed_np = transpose_raveled_numpy(i, grid_shape, new_order)

        assert transposed == transposed_np

    # Move the last axis to the first dimension.
    new_order = np.array([2, 0, 1], dtype=np.int32)

    for i in range(grid_size):
        transposed = transpose_raveled_index(i, grid_shape, new_order)
        transposed_np = transpose_raveled_numpy(i, grid_shape, new_order)

        assert transposed == transposed_np


@pytest.mark.parametrize("k", [(0, 0, 0), (0.5, 0, 0)])
def test_wavefunction_correct(H, grid_shape, psi_values, k):
    """Checks that the wavefunction computed with the precalculated
    psi values is the same as the wavefunction computed directly."""

    eig = H.eigenstate(k=k)[0]

    wf_grid = sisl.Grid(grid_shape, geometry=H.geometry, dtype=np.complex128)

    eig.wavefunction(wf_grid)

    from_psi_grid = psi_values.reduce_orbitals(eig.state.T, k=k)

    assert np.allclose(from_psi_grid.grid, wf_grid.grid)


@pytest.mark.parametrize(
    ["k", "ncoeffs"],
    [[(0, 0, 0), 1], [(0, 0, 0), 2], [(0.25, 0, 0), 1], [(0.25, 0, 0), 2]],
)
def test_onthefly_reduction(geometry, psi_values, k, ncoeffs):
    """Checks that the on the fly reduction produces the same
    results as computing the whole grid and then reducing."""

    coeffs = np.random.random((geometry.no, ncoeffs))

    not_reduced = psi_values.reduce_orbitals(coeffs, k=k)

    # Reducing the last axis
    reduced = psi_values.reduce_orbitals(coeffs, k=k, reduce_grid=(2,))

    post_reduced = not_reduced.sum(2)
    assert reduced.size == post_reduced.size
    if ncoeffs == 1:
        assert np.allclose(reduced.grid, post_reduced.grid)
    else:
        assert np.allclose(reduced.reshape(post_reduced.shape), post_reduced)

    # Reducing the two last axes
    reduced = psi_values.reduce_orbitals(coeffs, k=k, reduce_grid=(2, 1))

    post_reduced = not_reduced.sum(2).sum(1)
    assert reduced.size == post_reduced.size
    if ncoeffs == 1:
        assert np.allclose(reduced.grid, post_reduced.grid)
    else:
        assert np.allclose(reduced.reshape(post_reduced.shape), post_reduced)

    # Now we are going to reduce the leading axes. This is more involved, so
    # it is more likely to fail.

    # Reducing the first axis
    reduced = psi_values.reduce_orbitals(coeffs, k=k, reduce_grid=(0,))

    post_reduced = not_reduced.sum(0)
    assert reduced.size == post_reduced.size
    if ncoeffs == 1:
        assert np.allclose(reduced.grid, post_reduced.grid)
    else:
        assert np.allclose(reduced.reshape(post_reduced.shape), post_reduced)

    # Reducing the first and second axis
    reduced = psi_values.reduce_orbitals(coeffs, k=k, reduce_grid=(0, 1))

    post_reduced = not_reduced.sum(1).sum(0)
    assert reduced.size == post_reduced.size
    if ncoeffs == 1:
        assert np.allclose(reduced.grid, post_reduced.grid)
    else:
        assert np.allclose(reduced.reshape(post_reduced.shape), post_reduced)


def test_orbital_products(geometry):
    """Very simple tests to see if the orbital products are computed correctly"""
    geometry = geometry.copy()
    geometry.set_nsc([1, 1, 1])
    geometry.lattice.set_boundary_condition("open")

    # Don't get periodic contributions for orbital values, then predicting
    # grid values will be much easier.
    psi_values = geometry._orbital_values((10, 10, 10))

    orb_csr = psi_values._csr.tocsr()
    orb_0 = orb_csr[:, 0].toarray().ravel()
    orb_1 = orb_csr[:, 1].toarray().ravel()

    # Compute the orbital products with one coefficient
    DM = sisl.DensityMatrix(geometry, dim=1, dtype=np.float64)
    DM[0, 0] = 1.0

    dens = psi_values.reduce_orbital_products(DM.tocsr(), geometry.lattice)
    assert isinstance(dens, Grid)
    assert dens.shape == psi_values.grid_shape
    assert np.any(dens.grid != 0)

    predicted = orb_0**2

    assert np.allclose(dens.grid.ravel(), predicted)

    # Compute the orbital products with two diagonal coefficients
    DM = sisl.DensityMatrix(geometry, dim=1, dtype=np.float64)
    DM[0, 0] = 1.0
    DM[1, 1] = 2.0

    dens = psi_values.reduce_orbital_products(DM.tocsr(), DM.lattice)
    predicted = (orb_0**2) * 1.0 + (orb_1**2) * 2.0

    assert np.allclose(dens.grid.ravel(), predicted)

    # Compute the orbital products with an off-diagonal coefficient
    DM = sisl.DensityMatrix(geometry, dim=1, dtype=np.float64)
    DM[0, 0] = 1.0
    DM[0, 1] = 2.0

    dens = psi_values.reduce_orbital_products(DM.tocsr(), DM.lattice)
    predicted = (orb_0**2) * 1.0 + (orb_0 * orb_1) * 2.0

    assert np.allclose(dens.grid.ravel(), predicted)

    # Compute the orbital products with both opposite off-diagonal coefficients
    DM = sisl.DensityMatrix(geometry, dim=1, dtype=np.float64)
    DM[0, 0] = 1.0
    DM[0, 1] = 1.0
    DM[1, 0] = 0.5

    dens = psi_values.reduce_orbital_products(DM.tocsr(), DM.lattice)
    predicted = (orb_0**2) * 1.0 + (orb_0 * orb_1) * 1.5

    assert np.allclose(dens.grid.ravel(), predicted)


def test_orbital_products_periodic(geometry):
    """Very simple tests to see if the orbital products are computed correctly
    when there are periodic conditions.
    """
    geometry = geometry.copy()
    geometry.set_nsc([3, 1, 1])
    geometry.lattice.set_boundary_condition(a="periodic", b="open", c="open")

    # Don't get periodic contributions for orbital values, then predicting
    # grid values will be much easier.
    psi_values = geometry._orbital_values((10, 10, 10))

    orb_csr = psi_values._csr.tocsr()
    orb_0 = orb_csr[:, [0, geometry.no, 2 * geometry.no]].toarray()
    orb_1 = orb_csr[:, [1, geometry.no + 1, 2 * geometry.no + 1]].toarray()

    # Compute the orbital products with one coefficient
    DM = sisl.DensityMatrix(geometry, dim=1, dtype=np.float64)
    DM[0, 0] = 1.0

    dens = psi_values.reduce_orbital_products(DM.tocsr(), geometry.lattice)

    predicted = (orb_0**2).sum(axis=1)

    assert np.allclose(dens.grid.ravel(), predicted)

    # Compute the orbital products with two diagonal coefficients
    DM = sisl.DensityMatrix(geometry, dim=1, dtype=np.float64)
    DM[0, 0] = 1.0
    DM[1, 1] = 2.0

    dens = psi_values.reduce_orbital_products(DM.tocsr(), DM.lattice)
    predicted = (orb_0**2).sum(axis=1) + (orb_1**2).sum(axis=1) * 2.0

    assert np.allclose(dens.grid.ravel(), predicted)

    # Predicting the result with periodic non-diagonal coefficients is a bit more involved.
    # We don't check for now.


@pytest.mark.parametrize("ncoeffs", [1, 2])
def test_orbital_products_onthefly_reduction(geometry, psi_values, ncoeffs):
    """Checks that the on the fly reduction produces the same
    results as computing the whole grid and then reducing."""

    # Build a test DM
    DM = sisl.DensityMatrix(geometry, dim=1, dtype=np.float64)
    DM[0, 0] = 1.0
    DM[1, 1] = 2.0
    DM[2, 2] = 0.75
    DM[3, 3] = 0.92
    DM[0, 1] = 0.5
    DM[1, 0] = 0.5

    csr = DM.tocsr()
    # If this run is to be done with multiple coefficients, build the csr
    # matrix with dim > 1.
    if ncoeffs > 1:
        csr = sisl.SparseCSR(csr, dim=ncoeffs)
        csr.data[:, :] = csr.data[:, 0].reshape(-1, 1)

    # Compute the values on the full grid
    not_reduced = psi_values.reduce_orbital_products(csr, DM.lattice)

    # Reducing the last axis
    reduced = psi_values.reduce_orbital_products(csr, DM.lattice, reduce_grid=(2,))

    post_reduced = not_reduced.sum(2)
    assert reduced.size == post_reduced.size
    if ncoeffs == 1:
        assert np.allclose(reduced.grid, post_reduced.grid, atol=1e-7)
    else:
        assert np.allclose(reduced.reshape(post_reduced.shape), post_reduced)

    # Reducing the two last axes
    reduced = psi_values.reduce_orbital_products(csr, DM.lattice, reduce_grid=(2, 1))

    post_reduced = not_reduced.sum(2).sum(1)
    assert reduced.size == post_reduced.size
    if ncoeffs == 1:
        assert np.allclose(reduced.grid, post_reduced.grid)
    else:
        assert np.allclose(reduced.reshape(post_reduced.shape), post_reduced)

    # Now we are going to reduce the leading axes. This is more involved, so
    # it is more likely to fail.

    # Reducing the first axis
    reduced = psi_values.reduce_orbital_products(csr, DM.lattice, reduce_grid=(0,))

    post_reduced = not_reduced.sum(0)
    assert reduced.size == post_reduced.size
    if ncoeffs == 1:
        assert np.allclose(reduced.grid, post_reduced.grid)
    else:
        assert np.allclose(reduced.reshape(post_reduced.shape), post_reduced)

    # Reducing the first and second axis
    reduced = psi_values.reduce_orbital_products(csr, DM.lattice, reduce_grid=(0, 1))

    post_reduced = not_reduced.sum(1).sum(0)
    assert reduced.size == post_reduced.size
    if ncoeffs == 1:
        assert np.allclose(reduced.grid, post_reduced.grid)
    else:
        assert np.allclose(reduced.reshape(post_reduced.shape), post_reduced)
