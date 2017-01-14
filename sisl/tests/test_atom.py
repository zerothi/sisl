from __future__ import print_function, division

from nose.tools import *
from nose.plugins.attrib import attr

import math as m
import numpy as np

from sisl import Atom, Atoms, PeriodicTable


@attr('atom')
class TestAtom(object):

    def setUp(self):
        self.C = Atom['C']
        self.C3 = Atom('C', orbs=3)
        self.Au = Atom['Au']
        self.PT = PeriodicTable()

    def tearDown(self):
        del self.C
        del self.Au
        del self.C3
        del self.PT

    def test1(self):
        assert_true(self.C == Atom['C'])
        assert_true(self.Au == Atom['Au'])
        assert_true(self.Au != self.C)
        assert_false(self.Au == self.C)
        assert_true(self.Au == self.Au.copy())

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
        assert_true(len(Atom(Z=1, orbs=3)) == 3)
        assert_true(Atom(Z=1, R=1.4).R == 1.4)
        assert_true(Atom(Z=1, R=1.4).dR == 1.4)
        assert_true(Atom(Z=1, R=[1.4, 1.8]).orbs == 2)

    def test7(self):
        assert_true(Atom(Z=1, orbs=3).radius() > 0.)
        assert_true(len(str(Atom(Z=1, orbs=3))))

    def test8(self):
        a = self.PT.Z([1, 2])
        assert_true(len(a) == 2)
        assert_true(a[0] == 1)
        assert_true(a[1] == 2)

    def test9(self):
        a = self.PT.Z_label(['H', 2])
        assert_true(len(a) == 2)
        assert_true(a[0] == 'H')
        assert_true(a[1] == 'He')
        a = self.PT.Z_label(1)
        assert_true(a == 'H')

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


class TestAtoms(object):

    def setUp(self):
        self.C = Atom['C']
        self.C3 = Atom('C', orbs=3)
        self.Au = Atom['Au']

    def test_create1(self):
        atom1 = Atoms([self.C, self.C3, self.Au])
        atom2 = Atoms(['C', 'C', 'Au'])
        atom3 = Atoms(['C', 6, 'Au'])
        atom4 = Atoms(['Au', 6, 'C'])
        assert_true(atom2 == atom3)
        assert_true(atom2 == atom4)

    def test_create2(self):
        atom = Atoms(Atom(6, R=1.45), na=2)
        atom = Atoms(atom, na=4)
        assert_true(atom[0].dR == 1.45)

    def test_len(self):
        atom = Atoms([self.C, self.C3, self.Au])
        assert_true(len(atom) == 3)

    def test_get1(self):
        atoms = Atoms(['C', 'C', 'Au'])
        assert_true(atoms[2] == Atom('Au'))
        assert_true(atoms[0] == Atom('C'))
        assert_true(atoms[1] == Atom('C'))
        assert_true(atoms[0:2] == [Atom('C')]*2)
        assert_true(atoms[1:] == [Atom('C'), Atom('Au')])

    def test_set1(self):
        # Add new atoms to the set
        atom = Atoms(['C', 'C'])
        assert_true(atom[0] == Atom('C'))
        assert_true(atom[1] == Atom('C'))
        atom[1] = Atom('Au')
        assert_true(atom[0] == Atom('C'))
        assert_true(atom[1] == Atom('Au'))

    def test_set2(self):
        # Add new atoms to the set
        atom = Atoms(['C', 'C'])
        assert_true(atom[0] == Atom('C'))
        assert_true(atom[1] == Atom('C'))
        assert_true(len(atom.atom) == 1)
        atom[1] = Atom('Au', orbs=2)
        assert_true(atom[0] == Atom('C'))
        assert_false(atom[1] == Atom('Au'))
        assert_true(atom[1] == Atom('Au', orbs=2))
        assert_true(len(atom.atom) == 2)

    def test_set3(self):
        # Add new atoms to the set
        atom = Atoms(['C'] * 10)
        atom[range(1, 4)] = Atom('Au', orbs=2)
        assert_true(atom[0] == Atom('C'))
        for i in range(1, 4):
            assert_false(atom[i] == Atom('Au'))
            assert_true(atom[i] == Atom('Au', orbs=2))
        assert_false(atom[4] == Atom('Au'))
        assert_false(atom[4] == Atom('Au', orbs=2))
        assert_true(len(atom.atom) == 2)
        atom[1:4] = Atom('C')
        assert_true(len(atom.atom) == 2)

    def test_in1(self):
        # Add new atoms to the set
        atom = Atoms(['C', 'C'])
        assert_true(Atom[6] in atom)
        assert_false(Atom[1] in atom)

    def test_iter1(self):
        # Add new atoms to the set
        atom = Atoms(['C', 'C'])
        for a, idx in atom:
            assert_true(a == Atom[6])
            assert_true(len(idx) == 2)

        atom = Atoms(['C', 'Au', 'C', 'Au'])
        for i, aidx in enumerate(atom):
            a, idx = aidx
            if i == 0:
                assert_true(a == Atom[6])
                assert_true((idx == [0, 2]).all())
            elif i == 1:
                assert_true(a == Atom['Au'])
                assert_true((idx == [1, 3]).all())
            assert_true(len(idx) == 2)

    def test_reduce1(self):
        atom = Atoms(['C', 'Au'])
        atom = atom.sub(0)
        atom = atom.reduce()
        assert_true(atom[0] == Atom[6])
        assert_true(len(atom) == 1)
        assert_true(len(atom.atom) == 1)

    def test_reorder1(self):
        atom = Atoms(['C', 'Au'])
        atom = atom.sub(1)
        atom = atom.reorder()
        assert_true(atom[0] == Atom['Au'])
        assert_true(len(atom) == 1)
        assert_true(len(atom.atom) == 2)
