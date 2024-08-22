# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

import sisl

pytestmark = [pytest.mark.io, pytest.mark.siesta]


def test_si_pdos_kgrid_grid(sisl_files):
    si = sisl.get_sile(sisl_files("siesta", "Si_pdos_k", "Si_pdos.VT"))
    si.read_grid()
    assert si.grid_unit == pytest.approx(sisl.unit.siesta.unit_convert("Ry", "eV"))


def test_si_pdos_kgrid_grid_cell(sisl_files):
    si = sisl.get_sile(sisl_files("siesta", "Si_pdos_k", "Si_pdos.VT"))
    si.read_lattice()


def test_si_pdos_kgrid_grid_fractions(sisl_files):
    si = sisl.get_sile(sisl_files("siesta", "Si_pdos_k", "Si_pdos.VT"))
    grid = si.read_grid()
    grid_halve = si.read_grid(index=[0.5])
    assert np.allclose(grid.grid * 0.5, grid_halve.grid)


def test_si_pdos_kgrid_grid_fdf(sisl_files):
    si = sisl.get_sile(sisl_files("siesta", "Si_pdos_k", "Si_pdos.fdf"))
    VT = si.read_grid("VT", order="bin")
    TotPot = si.read_grid("totalpotential", order="bin")
    assert np.allclose(VT.grid, TotPot.grid)


def test_grid_read_write(sisl_tmp):
    path = sisl_tmp("grid.bin")
    lat = sisl.Lattice(1).rotate(45, "z").rotate(45, "x")
    grid = sisl.Grid([4, 5, 6], lattice=lat)
    grid.grid = np.random.rand(*grid.shape)
    gridSile = sisl.io.siesta.gridSileSiesta

    sile = gridSile(path)
    grid.write(sile)
    grid2 = sile.read_grid()
    assert np.allclose(grid.shape, grid2.shape)
    assert np.allclose(grid.cell, grid2.cell)
    assert np.allclose(grid.grid, grid2.grid)

    sile = gridSile(path)
    sile.write_grid(grid, grid * 2)
    for idx in (0, 1):
        grid2 = sile.read_grid(index=idx)
        assert np.allclose(grid.shape, grid2.shape)
        assert np.allclose(grid.cell, grid2.cell)
        assert np.allclose(grid.grid * (1 + idx), grid2.grid)
