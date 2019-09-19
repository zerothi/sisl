from __future__ import print_function, division

import pytest
import numpy as np
import os.path as osp
import sisl
from sisl import Hamiltonian, DynamicalMatrix, DensityMatrix
from sisl import EnergyDensityMatrix
from sisl.io.siesta import *


pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = osp.join('sisl', 'io', 'siesta')


def test_nc1(sisl_tmp, sisl_system):
    f = sisl_tmp('gr.nc', _dir)
    tb = Hamiltonian(sisl_system.gtb)
    tb.construct([sisl_system.R, sisl_system.t])
    tb.write(ncSileSiesta(f, 'w'))

    ntb = ncSileSiesta(f).read_hamiltonian()

    # Assert they are the same
    assert np.allclose(tb.cell, ntb.cell)
    assert np.allclose(tb.xyz, ntb.xyz)
    tb.finalize()
    ntb.finalize()
    assert np.allclose(tb._csr._D[:, 0], ntb._csr._D[:, 0])
    assert sisl_system.g.atom.equal(ntb.atom, R=False)


def test_nc2(sisl_tmp, sisl_system):
    f = sisl_tmp('grS.nc', _dir)
    tb = Hamiltonian(sisl_system.gtb, orthogonal=False)
    tb.construct([sisl_system.R, sisl_system.tS])
    tb.write(ncSileSiesta(f, 'w'))

    ntb = ncSileSiesta(f).read_hamiltonian()

    # Assert they are the same
    assert np.allclose(tb.cell, ntb.cell)
    assert np.allclose(tb.xyz, ntb.xyz)
    tb.finalize()
    ntb.finalize()
    assert np.allclose(tb._csr._D[:, 0], ntb._csr._D[:, 0])
    assert sisl_system.g.atom.equal(ntb.atom, R=False)


def test_nc_overlap(sisl_tmp, sisl_system):
    f = sisl_tmp('gr.nc', _dir)
    tb = Hamiltonian(sisl_system.gtb)
    tb.construct([sisl_system.R, sisl_system.t])
    tb.write(ncSileSiesta(f, 'w'))

    S = ncSileSiesta(f).read_overlap()

    # Ensure no empty data-points
    S.finalize()
    assert np.allclose(S._csr._D.sum(), tb.no)


def test_nc_dynamical_matrix(sisl_tmp, sisl_system):
    f = sisl_tmp('grdyn.nc', _dir)
    dm = DynamicalMatrix(sisl_system.gtb)
    for _, ix in dm.iter_orbitals():
        dm[ix, ix] = ix / 2.
    dm.write(ncSileSiesta(f, 'w'))

    ndm = ncSileSiesta(f).read_dynamical_matrix()

    # Assert they are the same
    assert np.allclose(dm.cell, ndm.cell)
    assert np.allclose(dm.xyz, ndm.xyz)
    dm.finalize()
    ndm.finalize()
    assert np.allclose(dm._csr._D[:, 0], ndm._csr._D[:, 0])
    assert sisl_system.g.atom.equal(ndm.atom, R=False)


def test_nc_density_matrix(sisl_tmp, sisl_system):
    f = sisl_tmp('grDM.nc', _dir)
    dm = DensityMatrix(sisl_system.gtb)
    for _, ix in dm.iter_orbitals():
        dm[ix, ix] = ix / 2.
    dm.write(ncSileSiesta(f, 'w'))

    ndm = ncSileSiesta(f).read_density_matrix()

    # Assert they are the same
    assert np.allclose(dm.cell, ndm.cell)
    assert np.allclose(dm.xyz, ndm.xyz)
    dm.finalize()
    ndm.finalize()
    assert np.allclose(dm._csr._D[:, 0], ndm._csr._D[:, 0])
    assert sisl_system.g.atom.equal(ndm.atom, R=False)


def test_nc_H_non_colinear(sisl_tmp):
    H1 = Hamiltonian(sisl.geom.graphene(), spin=sisl.Spin('NC'))
    H1.construct(([0.1, 1.44],
                 [0.1, 0.2, 0.3, 0.4]))

    f1 = sisl_tmp('H1.nc', _dir)
    f2 = sisl_tmp('H2.nc', _dir)
    H1.write(f1)
    H1.finalize()
    H2 = sisl.get_sile(f1).read_hamiltonian()
    H2.write(f2)
    H3 = sisl.get_sile(f2).read_hamiltonian()
    assert H1._csr.spsame(H2._csr)
    assert np.allclose(H1._csr._D, H2._csr._D)
    assert H1._csr.spsame(H3._csr)
    assert np.allclose(H1._csr._D, H3._csr._D)


def test_nc_DM_non_colinear(sisl_tmp):
    DM1 = DensityMatrix(sisl.geom.graphene(), spin=sisl.Spin('NC'))
    DM1.construct(([0.1, 1.44],
                   [[0.1, 0.2, 0.3, 0.4],
                    [0.2, 0.3, 0.4, 0.5]]))

    f1 = sisl_tmp('DM1.nc', _dir)
    f2 = sisl_tmp('DM2.nc', _dir)
    DM1.write(f1)
    DM1.finalize()
    DM2 = sisl.get_sile(f1).read_density_matrix()
    DM2.write(f2)
    DM3 = sisl.get_sile(f2).read_density_matrix()
    assert DM1._csr.spsame(DM2._csr)
    assert np.allclose(DM1._csr._D, DM2._csr._D)
    assert DM1._csr.spsame(DM3._csr)
    assert np.allclose(DM1._csr._D, DM3._csr._D)


def test_nc_EDM_non_colinear(sisl_tmp):
    EDM1 = EnergyDensityMatrix(sisl.geom.graphene(), spin=sisl.Spin('NC'))
    EDM1.construct(([0.1, 1.44],
                    [0.1, 0.2, 0.3, 0.4]))

    f1 = sisl_tmp('EDM1.nc', _dir)
    f2 = sisl_tmp('EDM2.nc', _dir)
    EDM1.write(f1)
    EDM1.finalize()
    EDM2 = sisl.get_sile(f1).read_energy_density_matrix()
    EDM2.write(f2)
    EDM3 = sisl.get_sile(f2).read_energy_density_matrix()
    assert EDM1._csr.spsame(EDM2._csr)
    assert np.allclose(EDM1._csr._D, EDM2._csr._D)
    assert EDM1._csr.spsame(EDM3._csr)
    assert np.allclose(EDM1._csr._D, EDM3._csr._D)


def test_nc_H_spin_orbit(sisl_tmp):
    H1 = Hamiltonian(sisl.geom.graphene(), spin=sisl.Spin('SO'))
    H1.construct(([0.1, 1.44],
                  [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                   [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]))

    f1 = sisl_tmp('H1.nc', _dir)
    f2 = sisl_tmp('H2.nc', _dir)
    H1.write(f1)
    H1.finalize()
    H2 = sisl.get_sile(f1).read_hamiltonian()
    H2.write(f2)
    H3 = sisl.get_sile(f2).read_hamiltonian()
    assert H1._csr.spsame(H2._csr)
    assert np.allclose(H1._csr._D, H2._csr._D)
    assert H1._csr.spsame(H3._csr)
    assert np.allclose(H1._csr._D, H3._csr._D)


def test_nc_H_spin_orbit_nc2tshs2nc(sisl_tmp):
    H1 = Hamiltonian(sisl.geom.graphene(), spin=sisl.Spin('SO'))
    H1.construct(([0.1, 1.44],
                  [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                   [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]))

    f1 = sisl_tmp('H1.nc', _dir)
    f2 = sisl_tmp('H2.TSHS', _dir)
    H1.write(f1)
    H1.finalize()
    H2 = sisl.get_sile(f1).read_hamiltonian()
    H2.write(f2)
    H3 = sisl.get_sile(f2).read_hamiltonian()
    assert H1._csr.spsame(H2._csr)
    assert np.allclose(H1._csr._D, H2._csr._D)
    assert H1._csr.spsame(H3._csr)
    assert np.allclose(H1._csr._D, H3._csr._D)


def test_nc_DM_spin_orbit(sisl_tmp):
    DM1 = DensityMatrix(sisl.geom.graphene(), spin=sisl.Spin('SO'))
    DM1.construct(([0.1, 1.44],
                   [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                    [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]))

    f1 = sisl_tmp('DM1.nc', _dir)
    f2 = sisl_tmp('DM2.nc', _dir)
    DM1.write(f1)
    DM1.finalize()
    DM2 = sisl.get_sile(f1).read_density_matrix()
    DM2.write(f2)
    DM3 = sisl.get_sile(f2).read_density_matrix()
    assert DM1._csr.spsame(DM2._csr)
    assert np.allclose(DM1._csr._D, DM2._csr._D)
    assert DM1._csr.spsame(DM3._csr)
    assert np.allclose(DM1._csr._D, DM3._csr._D)


def test_nc_DM_spin_orbit_nc2dm2nc(sisl_tmp):
    DM1 = DensityMatrix(sisl.geom.graphene(), orthogonal=False, spin=sisl.Spin('SO'))
    DM1.construct(([0.1, 1.44],
                   [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.],
                    [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.]]))

    f1 = sisl_tmp('DM1.nc', _dir)
    f2 = sisl_tmp('DM2.DM', _dir)
    DM1.finalize()
    DM1.write(f1)
    DM2 = sisl.get_sile(f1).read_density_matrix()
    DM2.write(f2)
    DM3 = sisl.get_sile(f2).read_density_matrix()
    assert DM1._csr.spsame(DM2._csr)
    assert np.allclose(DM1._csr._D, DM2._csr._D)
    assert DM1._csr.spsame(DM3._csr)
    assert np.allclose(DM1._csr._D, DM3._csr._D)
