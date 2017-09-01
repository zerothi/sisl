from __future__ import print_function, division

import pytest

import math as m
import numpy as np

from sisl import Atom, Atoms, PeriodicTable


@pytest.fixture
def setup():
    class t():
        def __init__(self):
            self.C = Atom['C']
            self.C3 = Atom('C', orbs=3)
            self.Au = Atom('Au')
            self.PT = PeriodicTable()

    return t()


@pytest.mark.atom
class TestAtom(object):

    def test1(self, setup):
        assert setup.C == Atom['C']
        assert setup.C == Atom[setup.C]
        assert setup.C == Atom[Atom['C']]
        assert setup.C == Atom(Atom['C'])
        assert setup.C == Atom[Atom(6)]
        assert setup.Au == Atom['Au']
        assert setup.Au != setup.C
        assert setup.Au == setup.Au.copy()

    def test2(self, setup):
        C = Atom('C', R=20)
        assert setup.C != C
        Au = Atom('Au', R=20)
        assert setup.C != Au
        C = Atom['C']
        assert setup.Au != C
        Au = Atom['Au']
        assert setup.C != Au

    def test3(self, setup):
        assert setup.C.symbol == 'C'
        assert setup.Au.symbol == 'Au'

    def test4(self, setup):
        assert setup.C.mass > 0
        assert setup.Au.mass > 0

    def test5(self, setup):
        assert Atom(Z=1, mass=12).mass == 12
        assert Atom(Z=31, mass=12).mass == 12
        assert Atom(Z=31, mass=12).Z == 31

    def test6(self, setup):
        assert Atom(Z=1, orbs=3).orbs == 3
        assert len(Atom(Z=1, orbs=3)) == 3
        assert Atom(Z=1, R=1.4).R == 1.4
        assert Atom(Z=1, R=1.4).maxR() == 1.4
        assert Atom(Z=1, R=[1.4, 1.8]).orbs == 2
        assert Atom(Z=1, R=[1.4, 1.8]).maxR() == 1.8

    def test7(self, setup):
        assert Atom(Z=1, orbs=3).radius() > 0.
        assert len(str(Atom(Z=1, orbs=3)))

    def test8(self, setup):
        a = setup.PT.Z([1, 2])
        assert len(a) == 2
        assert a[0] == 1
        assert a[1] == 2

    def test9(self, setup):
        a = setup.PT.Z_label(['H', 2])
        assert len(a) == 2
        assert a[0] == 'H'
        assert a[1] == 'He'
        a = setup.PT.Z_label(1)
        assert a == 'H'

    def test10(self, setup):
        assert setup.PT.atomic_mass(1) == setup.PT.atomic_mass('H')
        assert np.allclose(setup.PT.atomic_mass([1, 2]), setup.PT.atomic_mass(['H', 'He']))

    def test11(self, setup):
        PT = setup.PT
        for m in ['calc', 'empirical', 'vdw']:
            assert PT.radius(1, method=m) == PT.radius('H', method=m)
            assert np.allclose(PT.radius([1, 2], method=m), PT.radius(['H', 'He'], method=m))

    @pytest.mark.xfail(raises=KeyError)
    def test12(self):
        a = Atom(1.2)

    @pytest.mark.xfail(raises=ValueError)
    def test_radius1(self, setup):
        setup.PT.radius(1, method='unknown')

    def test_tag1(self):
        a = Atom(6, tag='my-tag')
        assert a.tag == 'my-tag'

    def test_pickle(self, setup):
        import pickle as p
        sC = p.dumps(setup.C)
        sC3 = p.dumps(setup.C3)
        sAu = p.dumps(setup.Au)
        C = p.loads(sC)
        C3 = p.loads(sC3)
        Au = p.loads(sAu)
        assert Au != C
        assert C == setup.C
        assert C3 == setup.C3
        assert C != setup.Au
        assert Au == setup.Au
        assert Au != setup.C


@pytest.mark.atom
@pytest.mark.atoms
class TestAtoms(object):

    def test_create1(self, setup):
        atom1 = Atoms([setup.C, setup.C3, setup.Au])
        atom2 = Atoms(['C', 'C', 'Au'])
        atom3 = Atoms(['C', 6, 'Au'])
        atom4 = Atoms(['Au', 6, 'C'])
        assert atom2 == atom3
        assert atom2 != atom4
        assert atom2.hassame(atom4)

    def test_create2(self):
        atom = Atoms(Atom(6, R=1.45), na=2)
        atom = Atoms(atom, na=4)
        assert atom[0].maxR() == 1.45
        for ia in range(len(atom)):
            assert atom.maxR(True)[ia] == 1.45

    @pytest.mark.xfail(raises=ValueError)
    def test_create3(self):
        Atoms([{0: Atom(4)}])

    @pytest.mark.xfail(raises=ValueError)
    def test_create4(self):
        Atoms({0: Atom(4)})

    def test_len(self, setup):
        atom = Atoms([setup.C, setup.C3, setup.Au])
        assert len(atom) == 3

    def test_get1(self):
        atoms = Atoms(['C', 'C', 'Au'])
        assert atoms[2] == Atom('Au')
        assert atoms[0] == Atom('C')
        assert atoms[1] == Atom('C')
        assert atoms[0:2] == [Atom('C')]*2
        assert atoms[1:] == [Atom('C'), Atom('Au')]

    def test_set1(self):
        # Add new atoms to the set
        atom = Atoms(['C', 'C'])
        assert atom[0] == Atom('C')
        assert atom[1] == Atom('C')
        atom[1] = Atom('Au')
        assert atom[0] == Atom('C')
        assert atom[1] == Atom('Au')

    def test_set2(self):
        # Add new atoms to the set
        atom = Atoms(['C', 'C'])
        assert atom[0] == Atom('C')
        assert atom[1] == Atom('C')
        assert len(atom.atom) == 1
        atom[1] = Atom('Au', orbs=2)
        assert atom[0] == Atom('C')
        assert atom[1] != Atom('Au')
        assert atom[1] == Atom('Au', orbs=2)
        assert len(atom.atom) == 2

    def test_set3(self):
        # Add new atoms to the set
        atom = Atoms(['C'] * 10)
        atom[range(1, 4)] = Atom('Au', orbs=2)
        assert atom[0] == Atom('C')
        for i in range(1, 4):
            assert atom[i] != Atom('Au')
            assert atom[i] == Atom('Au', orbs=2)
        assert atom[4] != Atom('Au')
        assert atom[4] != Atom('Au', orbs=2)
        assert len(atom.atom) == 2
        atom[1:4] = Atom('C')
        assert len(atom.atom) == 2

    def test_append1(self):
        # Add new atoms to the set
        atom1 = Atoms(['C', 'C'])
        assert atom1[0] == Atom('C')
        assert atom1[1] == Atom('C')
        atom2 = Atoms([Atom('C', tag='DZ'), Atom[6]])
        assert atom2[0] == Atom('C', tag='DZ')
        assert atom2[1] == Atom('C')

        atom = atom1.append(atom2)
        assert atom[0] == Atom('C')
        assert atom[1] == Atom('C')
        assert atom[2] == Atom('C', tag='DZ')
        assert atom[3] == Atom('C')

        atom = atom1.append(Atom(6, tag='DZ'))
        assert atom[0] == Atom('C')
        assert atom[1] == Atom('C')
        assert atom[2] == Atom('C', tag='DZ')

        atom = atom1.append([Atom(6, tag='DZ'), Atom[6]])
        assert atom[0] == Atom('C')
        assert atom[1] == Atom('C')
        assert atom[2] == Atom('C', tag='DZ')
        assert atom[3] == Atom('C')

    def test_compare1(self):
        # Add new atoms to the set
        atom1 = Atoms([Atom('C', tag='DZ'), Atom[6]])
        atom2 = Atoms([Atom[6], Atom('C', tag='DZ')])
        assert atom1.hassame(atom2)
        assert not atom1.equal(atom2)

    def test_in1(self):
        # Add new atoms to the set
        atom = Atoms(['C', 'C'])
        assert Atom[6] in atom
        assert Atom[1] not in atom

    def test_iter1(self):
        # Add new atoms to the set
        atom = Atoms(['C', 'C'])
        for a, idx in atom:
            assert a == Atom[6]
            assert len(idx) == 2

        atom = Atoms(['C', 'Au', 'C', 'Au'])
        for i, aidx in enumerate(atom):
            a, idx = aidx
            if i == 0:
                assert a == Atom[6]
                assert (idx == [0, 2]).all()
            elif i == 1:
                assert a == Atom['Au']
                assert (idx == [1, 3]).all()
            assert len(idx) == 2

    def test_reduce1(self):
        atom = Atoms(['C', 'Au'])
        atom = atom.sub(0)
        atom = atom.reduce()
        assert atom[0] == Atom[6]
        assert atom[0] != Atom[8]
        assert len(atom) == 1
        assert len(atom.atom) == 1

    def test_remove1(self):
        atom = Atoms(['C', 'Au'])
        atom = atom.remove(1)
        atom = atom.reduce()
        assert atom[0] == Atom[6]
        assert len(atom) == 1
        assert len(atom.atom) == 1

    def test_reorder1(self):
        atom = Atoms(['C', 'Au'])
        atom = atom.sub(1)
        atom = atom.reorder()
        assert atom[0] == Atom['Au']
        assert len(atom) == 1
        assert len(atom.atom) == 2

    def test_reorder2(self):
        atom1 = Atoms(['C', 'Au'])
        atom2 = atom1.reorder()
        assert atom1 == atom2

    @pytest.mark.xfail(raises=KeyError)
    def test_index1(self):
        atom = Atoms(['C', 'Au'])
        atom.index(Atom('B'))
