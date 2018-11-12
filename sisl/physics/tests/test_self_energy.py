from __future__ import print_function, division

import pytest

import math as m
import warnings
import numpy as np

from sisl import Geometry, Atom, SuperCell, Hamiltonian
from sisl import BrillouinZone
from sisl import SelfEnergy, SemiInfinite, RecursiveSI
from sisl import RealSpaceSE


pytestmark = pytest.mark.self_energy


@pytest.fixture
def setup():
    class t():
        def __init__(self):
            bond = 1.42
            sq3h = 3.**.5 * 0.5
            self.sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                          [1.5, -sq3h, 0.],
                                          [0., 0., 10.]], np.float64) * bond, nsc=[3, 3, 1])

            C = Atom(Z=6, R=[bond * 1.01])
            self.g = Geometry(np.array([[0., 0., 0.],
                                        [1., 0., 0.]], np.float64) * bond,
                            atom=C, sc=self.sc)
            self.H = Hamiltonian(self.g)
            func = self.H.create_construct([0.1, bond+0.1], [0., -2.7])
            self.H.construct(func)
            self.HS = Hamiltonian(self.g, orthogonal=False)
            func = self.HS.create_construct([0.1, bond+0.1], [(0., 1.), (-2.7, 0.)])
            self.HS.construct(func)
    return t()


def test_objects(setup):
    for D, si, sid in [('+A', 0, 1),
                       ('-A', 0, -1),
                       ('+B', 1, 1),
                       ('-B', 1, -1)]:
        SE = SemiInfinite(setup.H, D)
        assert SE.semi_inf == si
        assert SE.semi_inf_dir == sid
        assert D in str(SE)


def test_sancho_orthogonal(setup):
    SE = RecursiveSI(setup.H, '+A')
    assert not np.allclose(SE.self_energy(0.1), SE.self_energy(0.1, bulk=True))


def test_sancho_orthogonal_dtype(setup):
    SE = RecursiveSI(setup.H, '+A')
    s64 = SE.self_energy(0.1, dtype=np.complex64)
    s128 = SE.self_energy(0.1)
    assert s64.dtype == np.complex64
    assert s128.dtype == np.complex128
    assert np.allclose(s64, s128)


def test_sancho_non_orthogonal(setup):
    SE = RecursiveSI(setup.HS, '-A')
    assert not np.allclose(SE.self_energy(0.1), SE.self_energy(0.1, bulk=True))


def test_sancho_non_orthogonal(setup):
    SE = RecursiveSI(setup.HS, '-A')
    s64 = SE.self_energy(0.1, dtype=np.complex64)
    s128 = SE.self_energy(0.1)
    assert s64.dtype == np.complex64
    assert s128.dtype == np.complex128
    assert np.allclose(s64, s128)


def test_sancho_lr(setup):
    SL = RecursiveSI(setup.HS, '-A')
    SR = RecursiveSI(setup.HS, '+A')

    E = 0.1
    k = [0, 0.13, 0]

    # Check that left/right are different

    L_SE = SL.self_energy(E, k)
    R_SE = SR.self_energy(E, k)
    assert not np.allclose(L_SE, R_SE)

    LB_SEL, LB_SER = SL.self_energy_lr(E, k)
    RB_SEL, RB_SER = SR.self_energy_lr(E, k)

    assert np.allclose(LB_SEL, L_SE)
    assert np.allclose(LB_SER, R_SE)
    assert np.allclose(RB_SEL, L_SE)
    assert np.allclose(RB_SER, R_SE)

    LB_SEL, LB_SER = SL.self_energy_lr(E, k, bulk=True)
    L_SE = SL.self_energy(E, k, bulk=True)
    R_SE = SR.self_energy(E, k,  bulk=True)
    assert not np.allclose(L_SE, R_SE)
    RB_SEL, RB_SER = SR.self_energy_lr(E, k, bulk=True)

    assert np.allclose(LB_SEL, L_SE)
    assert np.allclose(LB_SER, R_SE)
    assert np.allclose(RB_SEL, L_SE)
    assert np.allclose(RB_SER, R_SE)


@pytest.mark.parametrize("k_axes", [None, 0, 1])
@pytest.mark.parametrize("semi_axis", [None, 0, 1])
@pytest.mark.parametrize("trs", [True, False])
@pytest.mark.parametrize("bz", [None, BrillouinZone([1])])
@pytest.mark.parametrize("unfold", [1, 2])
def test_real_space_HS(setup, k_axes, semi_axis, trs, bz, unfold):
    if k_axes == semi_axis:
        return
    RSE = RealSpaceSE(setup.HS, (unfold, unfold, 1))
    RSE.update_option(semi_axis=semi_axis, k_axes=k_axes, dk=100, trs=trs, bz=bz)
    # Initialize and print
    with warnings.catch_warnings():
        #warnings.simplefilter('ignore')
        RSE.initialize()

    RSE.green(0.1)


@pytest.mark.parametrize("k_axes", [None, 0, 1])
@pytest.mark.parametrize("semi_axis", [None, 0, 1])
@pytest.mark.parametrize("trs", [True, False])
@pytest.mark.parametrize("bz", [None, BrillouinZone([1])])
@pytest.mark.parametrize("unfold", [1, 2])
def test_real_space_H(setup, k_axes, semi_axis, trs, bz, unfold):
    if k_axes == semi_axis:
        return
    RSE = RealSpaceSE(setup.H, (unfold, unfold, 1))
    RSE.update_option(semi_axis=semi_axis, k_axes=k_axes, dk=100, trs=trs, bz=bz)
    # Initialize and print
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        RSE.initialize()

    RSE.green(0.1)
    RSE.self_energy(0.1)


def test_real_space_H_3d():
    sc = SuperCell(1., nsc=[3] * 3)
    H = Atom(Z=1, R=[1.001])
    geom = Geometry([0] * 3, atom=H, sc=sc)
    H = Hamiltonian(geom)
    H.construct(([0.001, 1.01], (0, -1)))
    RSE = RealSpaceSE(H, (3, 4, 2))
    RSE.update_option(semi_axis=0, k_axes=(1, 2), dk=100, trs=True)
    # Initialize and print
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        RSE.initialize()
    nk = np.ones(3, np.int32)
    nk[RSE._options['k_axes']] = 23
    bz = BrillouinZone(H, nk)
    RSE.update_option(bz=bz)

    RSE.green(0.1)
    # Since there is only 2 repetitions along one direction we will have the full matrix
    # coupled!
    assert np.allclose(RSE.self_energy(0.1), RSE.self_energy(0.1, coupling=True))
    assert np.allclose(RSE.self_energy(0.1, bulk=True), RSE.self_energy(0.1, bulk=True, coupling=True))


def test_real_space_H_dtype(setup):
    RSE = RealSpaceSE(setup.H, (2, 2, 1))
    RSE.update_option(semi_axis=0, k_axes=1, dk=100)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        RSE.initialize()

    g64 = RSE.green(0.1, dtype=np.complex64)
    g128 = RSE.green(0.1, dtype=np.complex128)
    assert g64.dtype == np.complex64
    assert g128.dtype == np.complex128
    assert np.allclose(g64, g128, atol=1.e-4)

    s64 = RSE.self_energy(0.1, dtype=np.complex64)
    s128 = RSE.self_energy(0.1, dtype=np.complex128)
    assert s64.dtype == np.complex64
    assert s128.dtype == np.complex128
    assert np.allclose(s64, s128, atol=1e-2)


def test_real_space_H_SE_unfold(setup):
    # check that calculating the real-space Green function is equivalent for two equivalent systems
    RSE = RealSpaceSE(setup.H, (2, 2, 1), semi_axis=0, k_axes=1, dk=100)
    RSE_big = RealSpaceSE(setup.H.tile(2, 0).tile(2, 1), semi_axis=0, k_axes=1, dk=100)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        RSE.initialize()
        RSE_big.initialize()

    for E in [0.1, 1.5]:
        G = RSE.green(E)
        G_big = RSE_big.green(E)
        assert np.allclose(G, G_big)

        SE = RSE.self_energy(E)
        SE_big = RSE_big.self_energy(E)
        assert np.allclose(SE, SE_big)


def test_real_space_HS_SE_unfold(setup):
    # check that calculating the real-space Green function is equivalent for two equivalent systems
    RSE = RealSpaceSE(setup.HS, (2, 2, 1), semi_axis=0, k_axes=1, dk=100)
    RSE_big = RealSpaceSE(setup.HS.tile(2, 0).tile(2, 1), semi_axis=0, k_axes=1, dk=100)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        RSE.initialize()
        RSE_big.initialize()

    for E in [0.1, 1.5]:
        G = RSE.green(E)
        G_big = RSE_big.green(E)
        assert np.allclose(G, G_big)

        SE = RSE.self_energy(E)
        SE_big = RSE_big.self_energy(E)
        assert np.allclose(SE, SE_big)
