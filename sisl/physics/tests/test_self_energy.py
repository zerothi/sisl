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


def test_sancho_non_orthogonal(setup):
    SE = RecursiveSI(setup.HS, '-A')
    assert not np.allclose(SE.self_energy(0.1), SE.self_energy(0.1, bulk=True))


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


@pytest.mark.parametrize("k_axis", [None, 0, 1])
@pytest.mark.parametrize("semi_axis", [None, 0, 1])
@pytest.mark.parametrize("trs", [True, False])
@pytest.mark.parametrize("bz", [None, BrillouinZone([1])])
@pytest.mark.parametrize("unfold", [1, 2])
def test_real_space_HS(setup, k_axis, semi_axis, trs, bz, unfold):
    if k_axis == semi_axis:
        return
    RSE = RealSpaceSE(setup.HS, (unfold, unfold, 1))
    RSE.update_option(semi_axis=semi_axis, k_axis=k_axis, dk=100, trs=trs, bz=bz)
    # Initialize and print
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        RSE.initialize()

    RSE.green(0.1)
    RSE.self_energy(0.1)


@pytest.mark.parametrize("k_axis", [None, 0, 1])
@pytest.mark.parametrize("semi_axis", [None, 0, 1])
@pytest.mark.parametrize("trs", [True, False])
@pytest.mark.parametrize("bz", [None, BrillouinZone([1])])
@pytest.mark.parametrize("unfold", [1, 2])
def test_real_space_H(setup, k_axis, semi_axis, trs, bz, unfold):
    if k_axis == semi_axis:
        return
    RSE = RealSpaceSE(setup.H, (unfold, unfold, 1))
    RSE.update_option(semi_axis=semi_axis, k_axis=k_axis, dk=100, trs=trs, bz=bz)
    # Initialize and print
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        RSE.initialize()

    RSE.green(0.1)
    RSE.self_energy(0.1)
