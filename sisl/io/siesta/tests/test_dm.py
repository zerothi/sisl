""" pytest test configures """
from __future__ import print_function

import pytest
import os.path as osp
import numpy as np
import sisl


pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = osp.join('sisl', 'io', 'siesta')


def test_dm_si_pdos_kgrid(sisl_files):
    fdf = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.fdf'), base=sisl_files(_dir))

    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.DM'))

    DM1 = si.read_density_matrix(geometry=fdf.read_geometry())
    DM2 = fdf.read_density_matrix(order=['DM'])

    assert DM1._csr.spsame(DM2._csr)
    assert np.allclose(DM1._csr._D[:, :-1], DM2._csr._D[:, :-1])


def test_dm_si_pdos_kgrid_rw(sisl_files, sisl_tmp):
    f1 = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.DM'))
    f2 = sisl.get_sile(sisl_tmp('test.DM', _dir))

    DM1 = f1.read_density_matrix()
    f2.write_density_matrix(DM1)
    DM2 = f2.read_density_matrix()

    assert DM1._csr.spsame(DM2._csr)
    assert np.allclose(DM1._csr._D[:, :-1], DM2._csr._D[:, :-1])


def test_dm_si_pdos_kgrid_mulliken(sisl_files):
    fdf = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.fdf'), base=sisl_files(_dir))
    DM = fdf.read_density_matrix(order=['DM'])

    Mo = DM.mulliken('orbital')
    Ma = DM.mulliken('atom')

    o2a = DM.geometry.o2a(np.arange(DM.no))

    ma = np.zeros_like(Ma.T)
    np.add.at(ma, o2a, Mo.T)
    assert np.allclose(ma.T, Ma)


def test_dm_soc_pt2_xx_mulliken(sisl_files):
    fdf = sisl.get_sile(sisl_files(_dir, 'SOC_Pt2_xx.fdf'), base=sisl_files(_dir))
    # Force reading a geometry with correct atomic and orbital configuration
    DM = fdf.read_density_matrix(order=['DM'])

    Mo = DM.mulliken('orbital')
    Ma = DM.mulliken('atom')

    o2a = DM.geometry.o2a(np.arange(DM.no))

    ma = np.zeros_like(Ma.T)
    np.add.at(ma, o2a, Mo.T)
    assert np.allclose(ma.T, Ma)


def test_dm_soc_pt2_xx_rw(sisl_files, sisl_tmp):
    f1 = sisl.get_sile(sisl_files(_dir, 'SOC_Pt2_xx.DM'))
    f2 = sisl.get_sile(sisl_tmp('test.DM', _dir))

    DM1 = f1.read_density_matrix()
    f2.write_density_matrix(DM1)
    DM2 = f2.read_density_matrix()

    assert DM1._csr.spsame(DM2._csr)
    DM1.finalize()
    assert DM1._csr.spsame(DM2._csr)
    assert np.allclose(DM1._csr._D[:, :-1], DM2._csr._D[:, :-1])


@pytest.mark.xfail(reason="Currently reading a geometry from TSHS does not retain l, m, zeta quantum numbers")
def test_dm_soc_pt2_xx_orbital_momentum(sisl_files):
    fdf = sisl.get_sile(sisl_files(_dir, 'SOC_Pt2_xx.fdf'), base=sisl_files(_dir))
    # Force reading a geometry with correct atomic and orbital configuration
    DM = fdf.read_density_matrix(order=['DM'])

    o2a = DM.geometry.o2a(np.arange(DM.no))

    # Calculate angular momentum
    Lo = DM.orbital_momentum('orbital')
    La = DM.orbital_momentum('atom')

    la = np.zeros_like(La)
    np.add.at(la, o2a, Lo.T)
    assert np.allclose(la, La)
