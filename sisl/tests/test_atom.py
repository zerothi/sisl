from __future__ import print_function, division

import pytest

import math as m
import numpy as np

from sisl import Atom, Atoms, PeriodicTable, Orbital

pytestmark = [pytest.mark.atom]


@pytest.fixture
def setup():
    class t():
        def __init__(self):
            self.C = Atom['C']
            self.C3 = Atom('C', [-1] * 3)
            self.Au = Atom('Au')
            self.PT = PeriodicTable()

    return t()


def test1(setup):
    assert setup.C == Atom['C']
    assert setup.C == Atom[setup.C]
    assert setup.C == Atom[Atom['C']]
    assert setup.C == Atom(Atom['C'])
    assert setup.C == Atom[Atom(6)]
    assert setup.Au == Atom['Au']
    assert setup.Au != setup.C
    assert setup.Au == setup.Au.copy()


def test2(setup):
    C = Atom('C', R=20)
    assert setup.C != C
    Au = Atom('Au', R=20)
    assert setup.C != Au
    C = Atom['C']
    assert setup.Au != C
    Au = Atom['Au']
    assert setup.C != Au


def test3(setup):
    assert setup.C.symbol == 'C'
    assert setup.C.tag == 'C'
    assert setup.Au.symbol == 'Au'


def test4(setup):
    assert setup.C.mass > 0
    assert setup.Au.mass > 0
    assert setup.Au.q0 == pytest.approx(0)


def test5(setup):
    assert Atom(Z=1, mass=12).R < 0
    assert Atom(Z=1, mass=12).mass == 12
    assert Atom(Z=31, mass=12).mass == 12
    assert Atom(Z=31, mass=12).Z == 31


def test6(setup):
    assert Atom(1, [-1] * 3).no == 3
    assert len(Atom(1, [-1] * 3)) == 3
    assert Atom(1, 1.4).R == 1.4
    assert Atom(Z=1, R=1.4).R == 1.4
    assert Atom(Z=1, R=1.4).maxR() == 1.4
    assert Atom(Z=1, R=[1.4, 1.8]).no == 2
    assert Atom(Z=1, R=[1.4, 1.8]).maxR() == 1.8
    assert Atom(Z=1, R=[1.4, 1.8]).maxR() == 1.8
    a = Atom(1, Orbital(1.4))
    assert a.R == 1.4
    assert a.no == 1
    a = Atom(1, [Orbital(1.4)])
    assert a.R == 1.4
    assert a.no == 1


def test7(setup):
    assert Atom(1, [-1] * 3).radius() > 0.
    assert len(str(Atom(1, [-1] * 3)))


def test8(setup):
    a = setup.PT.Z([1, 2])
    assert len(a) == 2
    assert a[0] == 1
    assert a[1] == 2


def test9(setup):
    a = setup.PT.Z_label(['H', 2])
    assert len(a) == 2
    assert a[0] == 'H'
    assert a[1] == 'He'
    a = setup.PT.Z_label(1)
    assert a == 'H'


def test10(setup):
    assert setup.PT.atomic_mass(1) == setup.PT.atomic_mass('H')
    assert np.allclose(setup.PT.atomic_mass([1, 2]), setup.PT.atomic_mass(['H', 'He']))


def test11(setup):
    PT = setup.PT
    for m in ['calc', 'empirical', 'vdw']:
        assert PT.radius(1, method=m) == PT.radius('H', method=m)
        assert np.allclose(PT.radius([1, 2], method=m), PT.radius(['H', 'He'], method=m))


def test_fail_equal():
    assert Atom(1.2) != 2.


@pytest.mark.xfail(raises=ValueError)
def test_radius1(setup):
    setup.PT.radius(1, method='unknown')


def test_tag1():
    a = Atom(6, tag='my-tag')
    assert a.tag == 'my-tag'


def test_negative1():
    a = Atom(-1)
    assert a.symbol == 'fa'
    assert a.tag == 'fa'
    assert a.Z == -1


def test_iter1():
    r = [1, 2]
    a = Atom(5, r)
    for i, o in enumerate(a):
        assert o.R == r[i]
    for i, o in enumerate(a.iter()):
        assert o.R == r[i]


def test_iter2():
    r = [1, 1, 2, 2]
    a = Atom(5, r)
    for i, o in enumerate(a.iter(True)):
        assert len(o) == 2


def test_charge():
    r = [1, 1, 2, 2]
    a = Atom(5, [Orbital(1., 1.), Orbital(1., 1.), Orbital(2.), Orbital(2.)])
    assert len(a.q0) == 4
    assert a.q0.sum() == pytest.approx(2)


def test_atoms_set():
    a1 = Atom(1)
    a2 = Atom(2)
    a3 = Atom(3)
    a = Atoms(a1, 3)
    assert len(a) == 3
    assert len(a.atom) == 1
    a[1] = a2
    assert len(a) == 3
    assert len(a.atom) == 2

    # Add the atom, but do not specify any
    # atoms to have the species
    a[[]] = a3
    assert len(a) == 3
    assert len(a.atom) == 3
    for atom in a:
        assert atom in [a1, a2]
        assert atom != a3
    found = 0
    for atom, i_s in a.iter(True):
        if atom == a1:
            assert len(i_s) == 2
            found += 1
        elif atom == a2:
            assert len(i_s) == 1
            found += 1
        elif atom == a3:
            assert len(i_s) == 0
            found += 1
    assert found == 3


def test_charge_diff():
    o1 = Orbital(1., 1.)
    o2 = Orbital(1., .5)
    a1 = Atom(5, [o1, o2, o1, o2])
    a2 = Atom(5, [o1, o2, o1, o2, o1, o1])
    assert len(a1.q0) == 4
    assert len(a2.q0) == 6
    assert a1.q0.sum() == pytest.approx(3)
    assert a2.q0.sum() == pytest.approx(5)


def test_multiple_orbitals():
    o = [Orbital(1., 1.), Orbital(2., .5), Orbital(3., .75)]
    a1 = Atom(5, o)
    assert len(a1) == 3
    for i in range(3):
        a2 = a1.sub([i])
        assert len(a2) == 1
        assert a2.orbital[0] == o[i]
        assert a2.orbital[0] == a2[0]
        a2 = a1.remove([i])
        assert len(a2) == 2

    a2 = a1.sub([1, 2, 0])
    assert a2.orbital[0] == o[1]
    assert a2[0] == o[1]
    assert a2.orbital[1] == o[2]
    assert a2[1] == o[2]
    assert a2.orbital[2] == o[0]
    assert a2[2] == o[0]


@pytest.mark.xfail(raises=ValueError)
def test_multiple_orbitals_fail_io():
    o = [Orbital(1., 1.), Orbital(2., .5), Orbital(3., .75)]
    Atom(5, o).sub([3])


@pytest.mark.xfail(raises=ValueError)
def test_multiple_orbitals_fail_len():
    o = [Orbital(1., 1.), Orbital(2., .5), Orbital(3., .75)]
    Atom(5, o).sub([0, 0, 0, 0, 0])


def test_pickle(setup):
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
