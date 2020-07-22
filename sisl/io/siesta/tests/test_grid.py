import pytest
import os.path as osp
import sisl
import numpy as np

pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = osp.join('sisl', 'io', 'siesta')


def test_si_pdos_kgrid_grid(sisl_files):
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.VT'))
    si.read_grid()
    assert si.grid_unit == pytest.approx(sisl.unit.siesta.unit_convert('Ry', 'eV'))


def test_si_pdos_kgrid_grid_cell(sisl_files):
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.VT'))
    si.read_supercell()


def test_si_pdos_kgrid_grid_fractions(sisl_files):
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.VT'))
    grid = si.read_grid()
    grid_halve = si.read_grid(index=[0.5])
    assert np.allclose(grid.grid * 0.5, grid_halve.grid)


def test_si_pdos_kgrid_grid_fdf(sisl_files):
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.fdf'))
    VT = si.read_grid("VT", order='bin')
    TotPot = si.read_grid("totalpotential", order='bin')
    assert np.allclose(VT.grid, TotPot.grid)
