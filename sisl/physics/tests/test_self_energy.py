from __future__ import print_function, division

from nose.tools import *
from nose.plugins.attrib import attr

import math as m
import numpy as np

from sisl import Geometry, Atom, SuperCell, Hamiltonian
from sisl import SelfEnergy, SemiInfinite


class TestSelfEnergy(object):

    def setUp(self):
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

    def tearDown(self):
        del self.sc
        del self.g
        del self.H
        del self.HS

    @raises(ValueError)
    def test_error1(self):
        SE = SemiInfinite(self.H, '+C')

    @raises(ValueError)
    def test_error2(self):
        SE = SemiInfinite(self.H, '-C')

    def test_objects(self):
        for D, si, sid in [('+A', 0, 1),
                           ('-A', 0, -1),
                           ('+B', 1, 1),
                           ('-B', 1, -1)]:
            SE = SemiInfinite(self.H, D)
            assert_equal(SE.semi_inf, si)
            assert_equal(SE.semi_inf_dir, sid)

    def test_sancho1(self):
        SE = SemiInfinite(self.H, '+A')
