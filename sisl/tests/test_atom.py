from __future__ import print_function, division

from nose.tools import *

from sisl import Atom

import math as m
import numpy as np


class TestAtom(object):

    def setUp(self):
        self.C = Atom['C']
        self.C3 = Atom('C', orbs=3)
        self.Au = Atom['Au']

    def tearDown(self):
        del self.C
        del self.Au

    def test1(self):
        assert_true(self.C == Atom['C'])
        assert_true(self.Au == Atom['Au'])
        assert_true(self.Au != self.C)
        assert_false(self.Au == self.C)

    def test2(self):
        C = Atom('C', R=20)
        assert_false(self.C == C)
        Au = Atom('Au', R=20)
        assert_false(self.C == Au)
        C = Atom['C']
        assert_false(self.Au == C)
        Au = Atom['Au']
        assert_false(self.C == Au)

    def test3(self):
        assert_true(self.C.symbol == 'C')
        assert_true(self.Au.symbol == 'Au')

    def test4(self):
        assert_true(self.C.mass > 0)
        assert_true(self.Au.mass > 0)

    def test5(self):
        assert_true(Atom(Z=1, mass=12).mass == 12)
        assert_true(Atom(Z=31, mass=12).mass == 12)
        assert_true(Atom(Z=31, mass=12).Z == 31)
        
    def test6(self):
        assert_true(Atom(Z=1, orbs=3).orbs == 3)
        assert_true(Atom(Z=1, R=1.4).R == 1.4)
        assert_true(Atom(Z=1, R=1.4).dR == 1.4)
        assert_true(Atom(Z=1, R=[1.4,1.8]).orbs == 2)

    def test7(self):
        assert_true(Atom(Z=1, orbs=3).radii() > 0.)
        assert_true(len(str(Atom(Z=1, orbs=3))))

    def test_pickle(self):
        import pickle as p
        sC = p.dumps(self.C)
        sC3 = p.dumps(self.C3)
        sAu = p.dumps(self.Au)
        C = p.loads(sC)
        C3 = p.loads(sC3)
        Au = p.loads(sAu)
        assert_false(Au == C)
        assert_true(Au != C)
        assert_true(C == self.C)
        assert_true(C3 == self.C3)
        assert_false(C == self.Au)
        assert_true(Au == self.Au)
        assert_true(Au != self.C)
