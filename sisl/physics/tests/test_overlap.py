import pytest

import numpy as np

from sisl import Geometry, Atom, SphericalOrbital, AtomicOrbital, SuperCell
from sisl import Grid, Spin
from sisl.physics.overlap import Overlap


pytestmark = pytest.mark.overlap


@pytest.fixture
def setup():
    class t():
        def __init__(self):
            bond = 1.42
            sq3h = 3.**.5 * 0.5
            self.sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                          [1.5, -sq3h, 0.],
                                          [0., 0., 10.]], np.float64) * bond, nsc=[3, 3, 1])

            n = 60
            rf = np.linspace(0, bond * 1.01, n)
            rf = (rf, rf)
            orb = SphericalOrbital(1, rf, 2.)
            C = Atom(6, orb.toAtomicOrbital())
            self.g = Geometry(np.array([[0., 0., 0.],
                                        [1., 0., 0.]], np.float64) * bond,
                              atoms=C, sc=self.sc)
            self.S = Overlap(self.g)

    return t()


class TestOverlap:

    def test_objects(self, setup):
        assert len(setup.S.xyz) == 2
        assert setup.g.no == len(setup.S)

    def test_dtype(self, setup):
        assert setup.S.dtype == np.float64

    def test_ortho(self, setup):
        assert setup.S.orthogonal

    def test_set1(self, setup):
        S = setup.S.copy()
        S.S[0, 0] = 1.
        assert S[0, 0] == 1.
        assert S[1, 0] == 0.

    def test_fromsp(self, setup):
        S = setup.S.copy()
        csr = S.tocsr()
        S2 = Overlap.fromsp(S.geometry, csr)
        assert np.allclose(csr.data, S2.tocsr().data)
