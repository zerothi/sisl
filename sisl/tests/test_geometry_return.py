from __future__ import print_function, division

from nose.tools import *
from nose.plugins.attrib import attr

from numbers import Integral, Real
import math as m
import numpy as np

from sisl import Sphere
from sisl import Geometry, Atom, SuperCell


@attr('geometry')
class TestGeometryReturn(object):

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
        C = Atom(Z=6, R=bond * 1.01, orbs=2)
        self.g2 = Geometry(np.array([[0., 0., 0.],
                                     [1., 0., 0.]], np.float64) * bond,
                           atom=C, sc=self.sc)

    def tearDown(self):
        del self.g
        del self.sc

    def test_fl_o(self):
        # first o is always one element longer than the number of atoms
        assert_true(np.all(self.g.firsto == [0, 1, 2]))
        assert_true(np.all(self.g.lasto == [0, 1]))

        # first o is always one element longer than the number of atoms
        assert_true(np.all(self.g2.firsto == [0, 2, 4]))
        assert_true(np.all(self.g2.lasto == [1, 3]))

    def test_rij1(self):
        d = self.g.rij(0, 1)
        assert_true(isinstance(d, Real))
        d = self.g.rij(0, [0, 1])
        assert_true(len(d) == 2)
        d = self.g.rij([0], [0, 1])
        assert_true(len(d) == 1)

        # also test orij
        d = self.g.orij(0, 1)
        assert_true(isinstance(d, Real))
        d = self.g.orij(0, [0, 1])
        assert_true(len(d) == 2)
        d = self.g.orij([0], [0, 1])
        assert_true(len(d) == 1)

    def test_rij2(self):
        d = self.g.rij([0, 1], [0, 1])
        assert_true(np.allclose(d, [0., 0.]))

    def test_osc2uc(self):
        # single value
        d = self.g.osc2uc(0)
        assert_true(d == 0)
        # more values
        d = self.g.osc2uc([0, 1, 2])
        assert_true(len(d) == 3)
        assert_true(np.allclose(d, [0, 1, 0]))

    def test_slice1(self):
        d = self.g[1]
        print(d)
        assert_true(len(d) == 3)
        d = self.g[[1, 2]]
        assert_true(d.shape == (2, 3))
        d = self.g[1:3]
        assert_true(d.shape == (2, 3))
        d = self.g[2:4]
        assert_true(d.shape == (2, 3))

        d = self.g[2:4, 1]
        assert_true(d.shape == (2,))
        d = self.g[2:4, 1:3]
        assert_true(d.shape == (2, 2))
        d = self.g[2:10:2, 1:3]
        assert_true(d.shape == (4, 2))
        d = self.g[2:10:2, 2]
        assert_true(d.shape == (4,))

        d = self.g[None, 2]
        assert_true(d.shape == (len(self.g),))

    def test_a2o(self):
        d = self.g.a2o(1)
        assert_true(isinstance(d, Integral))
        d = self.g.a2o([0, 1])
        assert_true(len(d) == 2)
        d = self.g2.a2o(1)
        assert_true(isinstance(d, Integral))
        d = self.g2.a2o([0, 1])
        assert_true(len(d) == 2)

        d = self.g.a2o(1, True)
        assert_true(len(d) == 1)
        d = self.g.a2o([0, 1], True)
        assert_true(len(d) == 2)
        d = self.g2.a2o(1, True)
        assert_true(len(d) == 2)
        d = self.g2.a2o([0, 1], True)
        assert_true(len(d) == 4)

    def test_o2a(self):
        d = self.g.o2a(1)
        assert_true(isinstance(d, Integral))
        d = self.g.o2a([0, 1])
        assert_true(len(d) == 2)

    def test_axyz(self):
        d = self.g.axyz(1)
        assert_true(len(d) == 3)
        d = self.g.axyz([0, 1])
        assert_true(d.shape == (2, 3))
