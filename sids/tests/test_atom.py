from __future__ import print_function, division

from nose.tools import *

from sids import Atom

import math as m
import numpy as np


class TestAtom(object):

    def setUp(self):
        self.C = Atom['C']
        self.Au = Atom['Au']

    def tearDown(self):
        del self.C
        del self.Au

    def test1(self):
        assert_true( self.C == Atom['C'] )
        assert_true( self.Au == Atom['Au'] )
        assert_true( self.Au != self.C )
        assert_false( self.Au == self.C )

    def test2(self):
        C = Atom('C', R=20)
        assert_false( self.C == C)
        Au = Atom('Au', R=20)
        assert_false( self.C == Au)
        C = Atom['C']
        assert_false( self.Au == C)
        Au = Atom['Au']
        assert_false( self.C == Au)

    def test3(self):
        assert_true( self.C.symbol == 'C' )
        assert_true( self.Au.symbol == 'Au' )

    def test4(self):
        assert_true( self.C.mass > 0 )
        assert_true( self.Au.mass > 0 )

    def test5(self):
        assert_true( Atom(Z=1,mass=12).mass == 12 )
        assert_true( Atom(Z=31,mass=12).mass == 12 )
        assert_true( Atom(Z=31,mass=12).Z == 31 )
