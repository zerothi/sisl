# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

""" pytest test configures """


import numpy as np
import pytest

import sisl

pytestmark = [pytest.mark.io, pytest.mark.siesta]


def test_dm_si_pdos_kgrid(sisl_files):
    fdf = sisl.get_sile(sisl_files("siesta", "Si_pdos_k", "Si_pdos.fdf"))
    si = sisl.get_sile(sisl_files("siesta", "Si_pdos_k", "Si_pdos.DM"))

    DM1 = si.read_density_matrix(geometry=fdf.read_geometry())
    DM2 = fdf.read_density_matrix(order=["DM"])

    assert DM1._csr.spsame(DM2._csr)
    assert np.allclose(DM1._csr._D[:, :-1], DM2._csr._D[:, :-1])


def test_dm_si_pdos_kgrid_rw(sisl_files, sisl_tmp):
    fdf = sisl.get_sile(sisl_files("siesta", "Si_pdos_k", "Si_pdos.fdf"))
    geom = fdf.read_geometry()

    f1 = sisl.get_sile(sisl_files("siesta", "Si_pdos_k", "Si_pdos.DM"))
    f2 = sisl.get_sile(sisl_tmp("test.DM"))

    DM1 = f1.read_density_matrix(geometry=geom)
    f2.write_density_matrix(DM1, sort=False)
    DM2 = f2.read_density_matrix(geometry=geom)

    assert DM1._csr.spsame(DM2._csr)
    assert np.allclose(DM1._csr._D[:, :-1], DM2._csr._D[:, :-1])

    f2.write_density_matrix(DM1)
    DM2 = f2.read_density_matrix(sort=False)
    assert DM1._csr.spsame(DM2._csr)
    assert not np.allclose(DM1._csr._D[:, :-1], DM2._csr._D[:, :-1])
    DM2.finalize()
    assert np.allclose(DM1._csr._D[:, :-1], DM2._csr._D[:, :-1])


def test_dm_si_pdos_kgrid_mulliken(sisl_files):
    fdf = sisl.get_sile(sisl_files("siesta", "Si_pdos_k", "Si_pdos.fdf"))
    DM = fdf.read_density_matrix(order=["DM"])

    Mo = DM.mulliken("orbital")
    Ma = DM.mulliken("atom")

    o2a = DM.geometry.o2a(np.arange(DM.no))

    ma = np.zeros_like(Ma)
    np.add.at(ma, o2a, Mo)
    assert np.allclose(ma, Ma)


def test_dm_soc_pt2_xx_mulliken(sisl_files):
    fdf = sisl.get_sile(sisl_files("siesta", "Pt2_soc", "Pt2.fdf"))
    # Force reading a geometry with correct atomic and orbital configuration
    DM = fdf.read_density_matrix(order=["DM"])

    Mo = DM.mulliken("orbital")
    Ma = DM.mulliken("atom")

    o2a = DM.geometry.o2a(np.arange(DM.no))

    ma = np.zeros_like(Ma)
    np.add.at(ma.T, o2a, Mo.T)
    assert np.allclose(ma, Ma)


def test_dm_soc_pt2_xx_rw(sisl_files, sisl_tmp):
    f1 = sisl.get_sile(sisl_files("siesta", "Pt2_soc", "Pt2_xx.DM"))
    f2 = sisl.get_sile(sisl_tmp("test.DM"))

    DM1 = f1.read_density_matrix()
    f2.write_density_matrix(DM1)
    DM2 = f2.read_density_matrix()

    assert DM1._csr.spsame(DM2._csr)
    DM1.finalize()
    assert DM1._csr.spsame(DM2._csr)
    assert np.allclose(DM1._csr._D[:, :-1], DM2._csr._D[:, :-1])


@pytest.mark.xfail(
    reason="Currently reading a geometry from TSHS does not retain l, m, zeta quantum numbers"
)
def test_dm_soc_pt2_xx_orbital_momentum(sisl_files):
    fdf = sisl.get_sile(sisl_files("siesta", "Pt2_soc", "Pt2.fdf"))
    # Force reading a geometry with correct atomic and orbital configuration
    DM = fdf.read_density_matrix(order=["DM"])

    o2a = DM.geometry.o2a(np.arange(DM.no))

    # Calculate angular momentum
    Lo = DM.orbital_momentum("orbital")
    La = DM.orbital_momentum("atom")

    la = np.zeros_like(La)
    np.add.at(la, o2a, Lo.T)
    assert np.allclose(la, La)
