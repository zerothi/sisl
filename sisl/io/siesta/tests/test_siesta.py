import os.path as osp

import pytest
import numpy as np

import sisl
from sisl import Hamiltonian, DynamicalMatrix, DensityMatrix
from sisl import EnergyDensityMatrix
from sisl import Atom, Geometry
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
    assert np.allclose(tb._csr._D[:, 0], ntb._csr._D[:, 0])
    assert sisl_system.g.atoms.equal(ntb.atoms, R=False)


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
    assert np.allclose(tb._csr._D[:, 0], ntb._csr._D[:, 0])
    assert sisl_system.g.atoms.equal(ntb.atoms, R=False)


def test_nc_multiple_fail(sisl_tmp, sisl_system):
    # writing two different sparse matrices to the same
    # file will fail
    f = sisl_tmp('gr.nc', _dir)
    H = Hamiltonian(sisl_system.gtb)
    DM = DensityMatrix(sisl_system.gtb)

    sile = ncSileSiesta(f, 'w')
    H.construct([sisl_system.R, sisl_system.t])
    H.write(sile)

    DM[0, 0] = 1.
    with pytest.raises(ValueError):
        DM.write(sile)


@pytest.mark.parametrize(
    ("sort"),
    [True, False],
)
def test_nc_multiple_checks(sisl_tmp, sisl_system, sort):
    f = sisl_tmp('gr.nc', _dir)
    H = Hamiltonian(sisl_system.gtb)
    DM = DensityMatrix(sisl_system.gtb)

    sile = ncSileSiesta(f, 'w')
    H.construct([sisl_system.R, sisl_system.t])
    H.write(sile, sort=sort)

    shuffle = np.random.shuffle
    for io in range(len(H)):
        edges = H.edges(io) # get all edges
        shuffle(edges)
        DM[io, edges] = 2.

    if not sort:
        with pytest.raises(ValueError):
            DM.write(sile, sort=sort)
    else:
        DM.write(sile, sort=sort)


def test_nc_overlap(sisl_tmp, sisl_system):
    f = sisl_tmp('gr.nc', _dir)
    tb = Hamiltonian(sisl_system.gtb)
    tb.construct([sisl_system.R, sisl_system.t])
    tb.write(ncSileSiesta(f, 'w'))

    S = ncSileSiesta(f).read_overlap()

    # Ensure no empty data-points
    assert np.allclose(S._csr._D.sum(), tb.no)

    # Write test
    f = sisl_tmp('s.nc', _dir)
    S.write(ncSileSiesta(f, "w"))
    S2 = ncSileSiesta(f).read_overlap()
    assert S._csr.spsame(S2._csr)
    assert np.allclose(S._csr._D, S2._csr._D)


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
    assert np.allclose(dm._csr._D[:, 0], ndm._csr._D[:, 0])
    assert sisl_system.g.atoms.equal(ndm.atoms, R=False)


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
    assert np.allclose(dm._csr._D[:, 0], ndm._csr._D[:, 0])
    assert sisl_system.g.atoms.equal(ndm.atoms, R=False)


def test_nc_H_non_colinear(sisl_tmp):
    H1 = Hamiltonian(sisl.geom.graphene(), spin=sisl.Spin('NC'))
    H1.construct(([0.1, 1.44],
                  [[0.1, 0.2, 0.3, 0.4],
                   [0.2, 0.3, 0.4, 0.5]]))

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
    assert DM1._csr.spsame(DM3._csr)
    # DM1 is finalized, but DM2 is not finalized
    assert np.allclose(DM1._csr._D, DM2._csr._D)
    # DM2 and DM3 are the same
    assert np.allclose(DM2._csr._D, DM3._csr._D)
    DM2.finalize()
    assert np.allclose(DM1._csr._D, DM2._csr._D)


def test_nc_EDM_non_colinear(sisl_tmp):
    EDM1 = EnergyDensityMatrix(sisl.geom.graphene(), spin=sisl.Spin('NC'))
    EDM1.construct(([0.1, 1.44],
                    [[0.1, 0.2, 0.3, 0.4],
                     [0.2, 0.3, 0.4, 0.5]]))

    f1 = sisl_tmp('EDM1.nc', _dir)
    f2 = sisl_tmp('EDM2.nc', _dir)
    EDM1.write(f1, sort=False)
    EDM1.finalize()
    EDM2 = sisl.get_sile(f1).read_energy_density_matrix(sort=False)
    EDM2.write(f2, sort=False)
    EDM3 = sisl.get_sile(f2).read_energy_density_matrix(sort=False)
    assert EDM1._csr.spsame(EDM2._csr)
    assert EDM1._csr.spsame(EDM3._csr)
    # EDM1 is finalized, but EDM2 is not finalized
    assert not np.allclose(EDM1._csr._D, EDM2._csr._D)
    # EDM2 and EDM3 are the same
    assert np.allclose(EDM2._csr._D, EDM3._csr._D)
    EDM2.finalize()
    assert np.allclose(EDM1._csr._D, EDM2._csr._D)


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


def test_nc_ghost(sisl_tmp):
    f = sisl_tmp('ghost.nc', _dir)
    a1 = Atom(1)
    am1 = Atom(-1)
    g = Geometry([[0., 0., i] for i in range(2)], [a1, am1], 2.)
    g.write(ncSileSiesta(f, 'w'))

    g2 = ncSileSiesta(f).read_geometry()
    assert np.allclose(g.cell, g2.cell)
    assert np.allclose(g.xyz, g2.xyz)
    assert np.allclose(g.atoms.Z, g2.atoms.Z)
    assert g.atoms[0].__class__ is g2.atoms[0].__class__
    assert g.atoms[1].__class__ is g2.atoms[1].__class__
    assert g.atoms[0].__class__ is not g2.atoms[1].__class__
