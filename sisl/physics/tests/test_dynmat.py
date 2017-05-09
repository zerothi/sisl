from __future__ import print_function, division

from nose.tools import *
from nose.plugins.attrib import attr

import math as m
import numpy as np

from sisl import Geometry, Atom, SuperCell, DynamicalMatrix


class TestHamiltonian(object):
    # Base test class for MaskedArrays.

    def setUp(self):
        bond = 1.42
        sq3h = 3.**.5 * 0.5
        self.sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                      [1.5, -sq3h, 0.],
                                      [0., 0., 10.]], np.float64) * bond, nsc=[3, 3, 1])

        C = Atom(Z=6, R=bond * 1.01, orbs=3)
        self.g = Geometry(np.array([[0., 0., 0.],
                                    [1., 0., 0.]], np.float64) * bond,
                          atom=C, sc=self.sc)
        self.D = DynamicalMatrix(self.g)

        def func(D, ia, idxs, idxs_xyz):
            idx = D.geom.close(ia, dR=(0.1, 1.44), idx=idxs, idx_xyz=idxs_xyz)
            ia = ia * 3

            i0 = idx[0] * 3
            i1 = idx[1] * 3
            print(ia, i0, i1)
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

    def tearDown(self):
        del self.sc
        del self.g
        del self.D

    def test_objects(self):
        print(self.D)
        assert_true(len(self.D.xyz) == 2)
        assert_true(self.g.no == len(self.D))

    def test_dtype(self):
        assert_true(self.D.dtype == np.float64)

    def test_ortho(self):
        assert_true(self.D.orthogonal)

    def test_set1(self):
        self.D.D[0, 0] = 1.
        assert_true(self.D[0, 0] == 1.)
        assert_true(self.D[1, 0] == 0.)
        self.D.empty()

    def test_correct_newton(self):
        self.D.construct(self.func)
        assert_true(self.D[0, 0] == 1.)
        assert_true(self.D[1, 0] == 0.1)
        assert_true(self.D[0, 1] == 0.1)
        self.D.correct_Newton()
        self.D.empty()
