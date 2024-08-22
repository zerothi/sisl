# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl import Atom, Atoms, Orbital, PeriodicTable

pytestmark = [pytest.mark.atom, pytest.mark.atoms]


@pytest.fixture
def setup():
    class t:
        def __init__(self):
            self.C = Atom["C"]
            self.C3 = Atom("C", [-1] * 3)
            self.Au = Atom("Au")
            self.PT = PeriodicTable()

    return t()


def test_create1(setup):
    atom1 = Atoms([setup.C, setup.C3, setup.Au])
    atom2 = Atoms(["C", "C", "Au"])
    atom3 = Atoms(["C", 6, "Au"])
    atom4 = Atoms(["Au", 6, "C"])
    assert atom2 == atom3
    assert atom2 != atom4
    assert atom2.hassame(atom4)


def test_create2():
    atom = Atoms(Atom(6, R=1.45), na=2)
    atom = Atoms(atom, na=4)
    assert atom[0].maxR() == 1.45
    for ia in range(len(atom)):
        assert atom.maxR(True)[ia] == 1.45


def test_create_map():
    atom = Atoms(map(lambda x: x, [1, 2, 3]))
    assert len(atom) == 3


def test_len(setup):
    atom = Atoms([setup.C, setup.C3, setup.Au])
    assert len(atom) == 3
    assert len(atom.q0) == 3


def test_get1():
    atoms = Atoms(["C", "C", "Au"])
    assert atoms[2] == Atom("Au")
    assert atoms["Au"] == Atom("Au")
    assert atoms[0] == Atom("C")
    assert atoms[1] == Atom("C")
    assert atoms["C"] == Atom("C")
    assert atoms[0:2] == [Atom("C")] * 2
    assert atoms[1:] == [Atom("C"), Atom("Au")]


def test_set1():
    # Add new atoms to the set
    atom = Atoms(["C", "C"])
    assert atom[0] == Atom("C")
    assert atom[1] == Atom("C")
    atom[1] = Atom("Au")
    assert atom[0] == Atom("C")
    assert atom[1] == Atom("Au")
    atom["C"] = Atom("Au")
    assert atom[0] == Atom("Au")
    assert atom[1] == Atom("Au")


@pytest.mark.filterwarnings("ignore", message="*Replacing atom")
def test_set2():
    # Add new atoms to the set
    atom = Atoms(["C", "C"])
    assert atom[0] == Atom("C")
    assert atom[1] == Atom("C")
    assert len(atom.atom) == 1
    atom[1] = Atom("Au", [-1] * 2)
    assert atom[0] == Atom("C")
    assert atom[1] != Atom("Au")
    assert atom[1] == Atom("Au", [-1] * 2)
    assert len(atom.atom) == 2
    atom["C"] = Atom("Au", [-1] * 2)
    assert atom[0] != Atom("Au")
    assert atom[0] == Atom("Au", [-1] * 2)
    assert atom[1] != Atom("Au")
    assert atom[1] == Atom("Au", [-1] * 2)
    assert len(atom.atom) == 1


def test_set3():
    # Add new atoms to the set
    atom = Atoms(["C"] * 10)
    atom[range(1, 4)] = Atom("Au", [-1] * 2)
    assert atom[0] == Atom("C")
    for i in range(1, 4):
        assert atom[i] != Atom("Au")
        assert atom[i] == Atom("Au", [-1] * 2)
    assert atom[4] != Atom("Au")
    assert atom[4] != Atom("Au", [-1] * 2)
    assert len(atom.atom) == 2
    atom[1:4] = Atom("C")
    assert len(atom.atom) == 2


@pytest.mark.filterwarnings("ignore", message="*Replacing atom")
def test_replace1():
    # Add new atoms to the set
    atom = Atoms(["C"] * 10 + ["B"] * 2)
    atom[range(1, 4)] = Atom("Au", [-1] * 2)
    assert atom[0] == Atom("C")
    for i in range(1, 4):
        assert atom[i] != Atom("Au")
        assert atom[i] == Atom("Au", [-1] * 2)
    assert atom[4] != Atom("Au")
    assert atom[4] != Atom("Au", [-1] * 2)
    assert len(atom.atom) == 3
    atom.replace(atom[0], Atom("C", [-1] * 2))
    assert atom[0] == Atom("C", [-1] * 2)
    assert len(atom.atom) == 3
    assert atom[0] == Atom("C", [-1] * 2)
    for i in range(4, 10):
        assert atom[i] == Atom("C", [-1] * 2)
    for i in range(1, 4):
        assert atom[i] == Atom("Au", [-1] * 2)
    for i in range(10, 12):
        assert atom[i] == Atom("B")


@pytest.mark.filterwarnings("ignore", message="*Substituting atom")
@pytest.mark.filterwarnings("ignore", message="*Replacing atom")
def test_replace2():
    # Add new atoms to the set
    atom = Atoms(["C"] * 10 + ["B"] * 2)
    atom.replace(range(1, 4), Atom("Au", [-1] * 2))
    assert atom[0] == Atom("C")
    for i in range(1, 4):
        assert atom[i] != Atom("Au")
        assert atom[i] == Atom("Au", [-1] * 2)
    assert atom[4] != Atom("Au")
    assert atom[4] != Atom("Au", [-1] * 2)
    assert len(atom.atom) == 3

    # Second replace call (equivalent to replace_atom)
    atom.replace(atom[0], Atom("C", [-1] * 2))
    assert atom[0] == Atom("C", [-1] * 2)
    assert len(atom.atom) == 3
    assert atom[0] == Atom("C", [-1] * 2)
    for i in range(4, 10):
        assert atom[i] == Atom("C", [-1] * 2)
    for i in range(1, 4):
        assert atom[i] == Atom("Au", [-1] * 2)
    for i in range(10, 12):
        assert atom[i] == Atom("B")


def test_append1():
    # Add new atoms to the set
    atom1 = Atoms(["C", "C"])
    assert atom1[0] == Atom("C")
    assert atom1[1] == Atom("C")
    atom2 = Atoms([Atom("C", tag="DZ"), Atom[6]])
    assert atom2[0] == Atom("C", tag="DZ")
    assert atom2[1] == Atom("C")

    atom = atom1.append(atom2)
    assert atom[0] == Atom("C")
    assert atom[1] == Atom("C")
    assert atom[2] == Atom("C", tag="DZ")
    assert atom[3] == Atom("C")

    atom = atom1.append(Atom(6, tag="DZ"))
    assert atom[0] == Atom("C")
    assert atom[1] == Atom("C")
    assert atom[2] == Atom("C", tag="DZ")

    atom = atom1.append([Atom(6, tag="DZ"), Atom[6]])
    assert atom[0] == Atom("C")
    assert atom[1] == Atom("C")
    assert atom[2] == Atom("C", tag="DZ")
    assert atom[3] == Atom("C")


def test_compare1():
    # Add new atoms to the set
    atom1 = Atoms([Atom("C", tag="DZ"), Atom[6]])
    atom2 = Atoms([Atom[6], Atom("C", tag="DZ")])
    assert atom1.hassame(atom2)
    assert not atom1.equal(atom2)


def test_in1():
    # Add new atoms to the set
    atom = Atoms(["C", "C"])
    assert Atom[6] in atom
    assert Atom[1] not in atom


def test_iter1():
    # Add new atoms to the set
    atom = Atoms(["C", "C"])
    for a in atom.iter():
        assert a == Atom[6]
    for a, idx in atom.iter(True):
        assert a == Atom[6]
        assert len(idx) == 2

    atom = Atoms(["C", "Au", "C", "Au"])
    for i, aidx in enumerate(atom.iter(True)):
        a, idx = aidx
        if i == 0:
            assert a == Atom[6]
            assert (idx == [0, 2]).all()
        elif i == 1:
            assert a == Atom["Au"]
            assert (idx == [1, 3]).all()
        assert len(idx) == 2


def test_reduce1():
    atom = Atoms(["C", "Au"])
    atom = atom.sub(0)
    atom1 = atom.reduce()
    assert atom[0] == Atom[6]
    assert len(atom) == 1
    assert len(atom.atom) == 2
    assert atom1[0] == Atom[6]
    assert atom1[0] != Atom[8]
    assert len(atom1) == 1
    assert len(atom1.atom) == 1
    atom.reduce(True)
    assert atom[0] == Atom[6]
    assert len(atom) == 1
    assert len(atom.atom) == 1


def test_remove1():
    atom = Atoms(["C", "Au"])
    atom = atom.remove(1)
    atom = atom.reduce()
    assert atom[0] == Atom[6]
    assert len(atom) == 1
    assert len(atom.atom) == 1


def test_reorder1():
    atom = Atoms(["C", "Au"])
    atom = atom.sub(1)
    atom1 = atom.reorder()
    # Check we haven't done anything to the original Atoms object
    assert atom[0] == Atom["Au"]
    assert atom.species[0] == 1
    assert len(atom) == 1
    assert len(atom.atom) == 2
    assert atom1[0] == Atom["Au"]
    assert atom1.species[0] == 0
    assert len(atom1) == 1
    assert len(atom1.atom) == 2
    # Do in-place
    atom.reorder(True)
    assert atom[0] == Atom["Au"]
    assert atom.species[0] == 0
    assert len(atom) == 1
    assert len(atom.atom) == 2


def test_reorder2():
    atom1 = Atoms(["C", "Au"])
    atom2 = atom1.reorder()
    assert atom1 == atom2


def test_charge1():
    atom = Atoms(["C", "Au"])
    assert len(atom.q0) == 2
    assert atom.q0.sum() == pytest.approx(0.0)


def test_charge_diff():
    o1 = Orbital(1.0, 1.0)
    o2 = Orbital(1.0, 0.5)
    a1 = Atom(5, [o1, o2, o1, o2])
    a2 = Atom(5, [o1, o2, o1, o2, o1, o1])
    a = Atoms([a1, a2])
    assert len(a.q0) == 2
    assert a.q0.sum() == pytest.approx(8)
    assert np.allclose(a.q0, [3, 5])


def test_index1():
    atom = Atoms(["C", "Au"])
    with pytest.raises(KeyError):
        atom.index(Atom("B"))
