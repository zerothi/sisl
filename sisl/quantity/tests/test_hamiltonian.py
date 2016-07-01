from __future__ import print_function, division

from nose.tools import *

from sisl import Geometry, Atom, SuperCell, Hamiltonian

import math as m
import numpy as np


class TestHamiltonian(object):
    # Base test class for MaskedArrays.

    def setUp(self):
        bond = 1.42
        sq3h = 3.**.5 * 0.5
        self.sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                      [1.5, -sq3h, 0.],
                                      [0., 0., 10.]], np.float64) * bond, nsc=[3, 3, 1])

        C = Atom(Z=6, R=bond * 1.01, orbs=1)
        self.g = Geometry(np.array([[0., 0., 0.],
                                    [1., 0., 0.]], np.float64) * bond,
                          atoms=C, sc=self.sc)
        self.H = Hamiltonian(self.g)
        self.HS = Hamiltonian(self.g, ortho = False)

        C = Atom(Z=6, R=bond * 1.01, orbs=2)
        self.g2 = Geometry(np.array([[0., 0., 0.],
                                    [1., 0., 0.]], np.float64) * bond,
                          atoms=C, sc=self.sc)
        self.H2 = Hamiltonian(self.g2)
        self.HS2 = Hamiltonian(self.g2, ortho = False)


    def tearDown(self):
        del self.sc
        del self.g
        del self.H
        del self.HS
        del self.g2
        del self.H2
        del self.HS2

    def test_objects(self):
        assert_true(len(self.H.xyz) == 2)
        assert_true(self.g.no == len(self.H))
        assert_true(len(self.HS.xyz) == 2)
        assert_true(self.g.no == len(self.HS))

        assert_true(len(self.H2.xyz) == 2)
        assert_true(self.g2.no == len(self.H2))
        assert_true(len(self.HS2.xyz) == 2)
        assert_true(self.g2.no == len(self.HS2))

    def test_dtype(self):
        assert_true(self.H.dtype == np.float64)
        assert_true(self.HS.dtype == np.float64)
        assert_true(self.H2.dtype == np.float64)
        assert_true(self.HS2.dtype == np.float64)

    def test_ortho(self):
        assert_true(self.H.orthogonal)
        assert_false(self.HS.orthogonal)

    def test_set(self):
        self.H.H[0,0] = 1.
        assert_true(self.H[0,0] == 1.)
        assert_true(self.H[1,0] == 0.)
        self.H.empty()

        self.HS.H[0,0] = 1.
        assert_true(self.HS.H[0,0] == 1.)
        assert_true(self.HS.H[1,0] == 0.)
        assert_true(self.HS.S[0,0] == 0.)
        assert_true(self.HS.S[1,0] == 0.)
        self.HS.S[0,0] = 1.
        assert_true(self.HS.S[0,0] == 1.)
        assert_true(self.HS.S[1,0] == 0.)
        self.HS.empty()
        
        self.HS[0,0] = 1., 1.
        assert_true(self.HS.H[0,0] == 1.)
        assert_true(self.HS.S[0,0] == 1.)
        self.HS.empty()

    def test_finalized(self):
        assert_false(self.H.finalized)
        self.H.H[0,0] = 1.
        self.H.finalize()
        assert_true(self.H.finalized)
        assert_true(self.H.nnz == 1)
        self.H.empty()
        assert_false(self.HS.finalized)
        self.HS[0,0] = 1., 1.
        self.HS.finalize()
        assert_true(self.HS.finalized)
        assert_true(self.HS.nnz == 1)
        self.HS.empty()
        

        
