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


    def tearDown(self):
        del self.g
        del self.sc

    def test_rij1(self):
        d = self.g.rij(0, 1)
        assert_true(isinstance(d, Real))
        d = self.g.rij(0, [0, 1])
        assert_true(len(d) == 2)
        d = self.g.rij([0], [0, 1])
        assert_true(len(d) == 2)

        d = self.g.orij(0, 1)
        assert_true(isinstance(d, Real))
        d = self.g.orij(0, [0, 1])
        assert_true(len(d) == 2)
        d = self.g.orij([0], [0, 1])
        assert_true(len(d) == 2)

    @raises(ValueError)
    def test_rij2(self):
        d = self.g.rij([0, 1], [0, 1])

    def test_a2o(self):
        d = self.g.a2o(1)
        assert_true(isinstance(d, Integral))
        d = self.g.a2o([0, 1])
        assert_true(len(d) == 2)

    @attr('only')
    def test_o2a(self):
        d = self.g.o2a(1)
        assert_true(isinstance(d, Integral))
        d = self.g.o2a([0, 1])
        assert_true(len(d) == 2)

    
