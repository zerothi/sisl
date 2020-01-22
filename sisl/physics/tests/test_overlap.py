from __future__ import print_function, division

import pytest

import numpy as np

from sisl import Geometry, Atom, SphericalOrbital, AtomicOrbital, SuperCell
from sisl import Grid, Spin
from sisl.physics.overlap import Overlap


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
                              atom=C, sc=self.sc)
            self.D = Overlap(self.g)

            def func(D, ia, idxs, idxs_xyz):
                idx = D.geom.close(ia, R=(0.1, 1.44), idx=idxs, idx_xyz=idxs_xyz)
                ia = ia * 3

                i0 = idx[0] * 3
                i1 = idx[1] * 3
                # on-site
                p = 1.
                D.D[ia, i0] = p
                D.D[ia+1, i0+1] = p
                D.D[ia+2, i0+2] = p

                # nn
                p = 0.1

                # on-site directions
                D.D[ia, ia+1] = p
                D.D[ia, ia+2] = p
                D.D[ia+1, ia] = p
                D.D[ia+1, ia+2] = p
                D.D[ia+2, ia] = p
                D.D[ia+2, ia+1] = p

                D.D[ia, i1+1] = p
                D.D[ia, i1+2] = p

                D.D[ia+1, i1] = p
                D.D[ia+1, i1+2] = p

                D.D[ia+2, i1] = p
                D.D[ia+2, i1+1] = p

            self.func = func
    return t()


@pytest.mark.overlap
class TestOverlap(object):

    def test_objects(self, setup):
        assert len(setup.D.xyz) == 2
        assert setup.g.no == len(setup.D)

    def test_dtype(self, setup):
        assert setup.D.dtype == np.float64

    def test_ortho(self, setup):
        assert not setup.D.orthogonal

    def test_set1(self, setup):
        D = setup.D.copy()
        D.S[0, 0] = 1.
        assert D[0, 0] == 1.
        assert D[1, 0] == 0.

    def test_fromsp(self, setup):
        D = setup.D.copy()
        csr = D.tocsr()
        D2 = Overlap.fromsp(D.geom, csr)
        assert np.allclose(csr.data, D2.tocsr().data)
