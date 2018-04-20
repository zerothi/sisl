from __future__ import print_function, division

import pytest

import math as m
import numpy as np

from sisl import Geometry, Atom, SuperCell, Hessian


@pytest.fixture
def setup():
    class t():
        def __init__(self):
            bond = 1.42
            sq3h = 3.**.5 * 0.5
            self.sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                          [1.5, -sq3h, 0.],
                                          [0., 0., 10.]], np.float64) * bond, nsc=[3, 3, 1])

            C = Atom(Z=6, R=[bond * 1.01] * 3)
            self.g = Geometry(np.array([[0., 0., 0.],
                                        [1., 0., 0.]], np.float64) * bond,
                              atom=C, sc=self.sc)
            self.H = Hessian(self.g)

            def func(H, ia, idxs, idxs_xyz):
                idx = H.geom.close(ia, R=(0.1, 1.44), idx=idxs, idx_xyz=idxs_xyz)
                ia = ia * 3

                i0 = idx[0] * 3
                i1 = idx[1] * 3
                # on-site
                p = 1.
                H.H[ia, i0] = p
                H.H[ia+1, i0+1] = p
                H.H[ia+2, i0+2] = p

                # nn
                p = 0.1

                # on-site directions
                H.H[ia, ia+1] = p
                H.H[ia, ia+2] = p
                H.H[ia+1, ia] = p
                H.H[ia+1, ia+2] = p
                H.H[ia+2, ia] = p
                H.H[ia+2, ia+1] = p

                H.H[ia, i1+1] = p
                H.H[ia, i1+2] = p

                H.H[ia+1, i1] = p
                H.H[ia+1, i1+2] = p

                H.H[ia+2, i1] = p
                H.H[ia+2, i1+1] = p

            self.func = func
    return t()


@pytest.mark.hessian
class TestHessian(object):

    def test_objects(self, setup):
        assert len(setup.H.xyz) == 2
        assert setup.g.no == len(setup.H)

    def test_dtype(self, setup):
        assert setup.H.dtype == np.float64

    def test_ortho(self, setup):
        assert setup.H.orthogonal

    def test_set1(self, setup):
        setup.H.H[0, 0] = 1.
        assert setup.H[0, 0] == 1.
        assert setup.H[1, 0] == 0.
        setup.H.empty()

    def test_correct_newton(self, setup):
        setup.H.construct(setup.func)
        assert setup.H[0, 0] == 1.
        assert setup.H[1, 0] == 0.1
        assert setup.H[0, 1] == 0.1
        setup.H.correct_Newton()
        setup.H.empty()
