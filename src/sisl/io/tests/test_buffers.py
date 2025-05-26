# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import io

import numpy as np
import pytest

import sisl


def test_buffer_write_read():
    """Test that we can write and read using a text buffer."""

    buffer = io.StringIO("")

    geometry = sisl.geom.graphene()

    sisl.io.fdfSileSiesta(buffer).write_geometry(geometry)

    buffer = io.StringIO(buffer.getvalue())

    read_geometry = sisl.io.fdfSileSiesta(buffer).read_geometry()

    assert np.allclose(read_geometry.xyz, geometry.xyz)
    assert np.allclose(read_geometry.cell, geometry.cell)


def test_buffer_write_read_binary():
    """Test that we can write and read binary files using buffers"""

    buffer = io.BytesIO()

    geometry = sisl.geom.graphene()
    H = sisl.Hamiltonian(geometry)
    H[0, 0] = 1.3
    H[1, 1] = 1.5

    sisl.io.hsxSileSiesta(buffer, mode="wb").write_hamiltonian(H)

    buffer = io.BytesIO(buffer.getvalue())

    read_H = sisl.io.hsxSileSiesta(buffer).read_hamiltonian()

    assert np.allclose(H.tocsr().toarray(), read_H.tocsr().toarray())


def test_buffer_write_read_cdf():
    """Test that we can write and read CDF files using buffers"""
    pytest.importorskip("netCDF4")

    buffer = io.BytesIO()

    geometry = sisl.geom.graphene()
    grid = sisl.Grid((2, 2, 2), geometry=geometry)
    grid[:] = np.random.random(grid.shape)

    nc = sisl.io.gridncSileSiesta(buffer, mode="wb")
    nc.write_grid(grid)

    buffer = io.BytesIO(buffer.getvalue())

    read_grid = sisl.io.gridncSileSiesta(buffer).read_grid()

    assert np.allclose(grid.grid, read_grid.grid)
