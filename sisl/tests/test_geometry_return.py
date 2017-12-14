from __future__ import print_function, division

import pytest

from numbers import Integral, Real
import math as m
import numpy as np

from sisl import Sphere
from sisl import Geometry, Atom, SuperCell


@pytest.fixture
def setup():
    class t():
        def __init__(self):
            bond = 1.42
            sq3h = 3.**.5 * 0.5
            self.sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                          [1.5, -sq3h, 0.],
                                          [0., 0., 10.]], np.float64) * bond, nsc=[3, 3, 1])
            C = Atom(Z=6, R=bond * 1.01)
            self.g = Geometry(np.array([[0., 0., 0.],
                                        [1., 0., 0.]], np.float64) * bond,
                              atom=C, sc=self.sc)

            C = Atom(Z=6, R=[bond * 1.01] * 2)
            self.g2 = Geometry(np.array([[0., 0., 0.],
                                         [1., 0., 0.]], np.float64) * bond,
                               atom=C, sc=self.sc)
    return t()


@pytest.mark.geometry
class TestGeometryReturn(object):

    def test_fl_o(self, setup):
        # first o is always one element longer than the number of atoms
        assert np.all(setup.g.firsto == [0, 1, 2])
        assert np.all(setup.g.lasto == [0, 1])

        # first o is always one element longer than the number of atoms
        assert np.all(setup.g2.firsto == [0, 2, 4])
        assert np.all(setup.g2.lasto == [1, 3])

    def test_rij1(self, setup):
        d = setup.g.rij(0, 1)
        assert isinstance(d, Real)
        d = setup.g.rij(0, [0, 1])
        assert len(d) == 2
        d = setup.g.rij([0], [0, 1])
        assert len(d) == 1

        # also test orij
        d = setup.g.orij(0, 1)
        assert isinstance(d, Real)
        d = setup.g.orij(0, [0, 1])
        assert len(d) == 2
        d = setup.g.orij([0], [0, 1])
        assert len(d) == 1

    def test_rij2(self, setup):
        d = setup.g.rij([0, 1], [0, 1])
        assert np.allclose(d, [0., 0.])

    def test_osc2uc(self, setup):
        # single value
        d = setup.g.osc2uc(0)
        assert d == 0
        # more values
        d = setup.g.osc2uc([0, 1, 2])
        assert len(d) == 3
        assert np.allclose(d, [0, 1, 0])

    def test_slice1(self, setup):
        d = setup.g[1]
        assert len(d) == 3
        d = setup.g[[1, 2]]
        assert d.shape == (2, 3)
        d = setup.g[1:3]
        assert d.shape == (2, 3)
        d = setup.g[2:4]
        assert d.shape == (2, 3)

        d = setup.g[1, 2]
        assert d == 0.
        d = setup.g[1, 1:3]
        assert d.shape == (2,)
        d = setup.g[1, :]
        assert d.shape == (3,)
        d = setup.g[2:4, :]
        assert d.shape == (2, 3)
        d = setup.g[2:4, 1]
        assert d.shape == (2,)
        d = setup.g[2:4, 1:3]
        assert d.shape == (2, 2)
        d = setup.g[2:10:2, 1:3]
        assert d.shape == (4, 2)
        d = setup.g[2:10:2, 2]
        assert d.shape == (4,)

        d = setup.g[None, 2]
        assert d.shape == (len(setup.g),)

    def test_a2o(self, setup):
        d = setup.g.a2o(1)
        assert isinstance(d, Integral)
        d = setup.g.a2o([0, 1])
        assert len(d) == 2
        d = setup.g2.a2o(1)
        assert isinstance(d, Integral)
        d = setup.g2.a2o([0, 1])
        assert len(d) == 2

        d = setup.g.a2o(1, True)
        assert len(d) == 1
        d = setup.g.a2o([0, 1], True)
        assert len(d) == 2
        d = setup.g2.a2o(1, True)
        assert len(d) == 2
        d = setup.g2.a2o([0, 1], True)
        assert len(d) == 4

    def test_o2a(self, setup):
        d = setup.g.o2a(1)
        assert isinstance(d, Integral)
        d = setup.g.o2a([0, 1])
        assert len(d) == 2

    def test_axyz(self, setup):
        d = setup.g.axyz(1)
        assert len(d) == 3
        d = setup.g.axyz([0, 1])
        assert d.shape == (2, 3)
