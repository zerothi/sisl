from __future__ import print_function, division

import pytest

import math as m
import numpy as np

from sisl import Geometry, Atom, SuperCell, Hamiltonian
from sisl import SelfEnergy, SemiInfinite


@pytest.fixture
def setup():
    class t():
        def __init__(self):
            bond = 1.42
            sq3h = 3.**.5 * 0.5
            self.sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                          [1.5, -sq3h, 0.],
                                          [0., 0., 10.]], np.float64) * bond, nsc=[3, 3, 1])

            C = Atom(Z=6, R=bond * 1.01, orbs=1)
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


@pytest.mark.self_energy
class TestSelfEnergy(object):

    def test_objects(self, setup):
        for D, si, sid in [('+A', 0, 1),
                           ('-A', 0, -1),
                           ('+B', 1, 1),
                           ('-B', 1, -1)]:
            SE = SemiInfinite(setup.H, D)
            assert SE.semi_inf == si
            assert SE.semi_inf_dir == sid

    def test_sancho1(self, setup):
        SE = SemiInfinite(setup.H, '+A')
