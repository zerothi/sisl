# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl import Atom, AtomGhost, Atoms, AtomUnknown, Orbital

pytestmark = [pytest.mark.atom]


@pytest.fixture
def setup():
    class t:
        def __init__(self):
            self.C = Atom["C"]
            self.C3 = Atom("C", [-1] * 3)
            self.Au = Atom("Au")

    return t()


def strings(atom):
    assert isinstance(str(atom), str)
    assert isinstance(repr(atom), str)


def test_atom_simple(setup):
    assert setup.C == Atom["C"]
    assert setup.C == Atom["6"]
    assert setup.C == Atom[setup.C]
    assert setup.C == Atom[Atom["C"]]
    assert setup.C == Atom(Atom["C"])
    assert setup.C == Atom[Atom(6)]
    assert setup.Au == Atom["Au"]
    assert setup.Au != setup.C
    assert setup.Au == setup.Au.copy()


def test_atom_ghost():
    assert isinstance(Atom[1], Atom)
    assert isinstance(Atom[-1], AtomGhost)
    assert isinstance(Atom(1), Atom)
    assert isinstance(Atom(-1), AtomGhost)
    assert isinstance(AtomGhost(-1), AtomGhost)
    assert AtomGhost(-1).Z == 1
    assert Atom(-1).Z == 1


def test_atom_weird():
    for el in (6, "C", "Carbon"):
        atom = Atom(el)
        assert atom.Z == 6
        assert atom.symbol == "C"
        strings(atom)

        atom = Atom[el]
        assert atom.Z == 6
        assert atom.symbol == "C"
        strings(atom)


def test_atom_unknown():
    for el in (1000, "1000"):
        atom = Atom(el)
        assert isinstance(atom, AtomUnknown)
        strings(atom)

        atom = Atom[el]
        assert isinstance(atom, AtomUnknown)
        strings(atom)

    # negative ones are still considered ghosts!
    for el in (-1000, "-1000"):
        atom = Atom(el)
        assert isinstance(atom, AtomGhost)
        strings(atom)

        atom = Atom[el]
        assert isinstance(atom, AtomGhost)
        strings(atom)


def test2(setup):
    C = Atom("C", R=20)
    assert setup.C != C
    Au = Atom("Au", R=20)
    assert setup.C != Au
    C = Atom["C"]
    assert setup.Au != C
    Au = Atom["Au"]
    assert setup.C != Au


def test3(setup):
    assert setup.C.symbol == "C"
    assert setup.C.tag == "C"
    assert setup.Au.symbol == "Au"


def test4(setup):
    assert setup.C.mass > 0
    assert setup.Au.mass > 0
    assert setup.Au.q0 == pytest.approx(0)


def test5(setup):
    assert Atom(Z=1, mass=12).R < 0
    assert Atom(Z=1, mass=12).R.size == 1
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
    assert Atom(1, [-1] * 3).radius() > 0.0
    assert len(str(Atom(1, [-1] * 3)))


def test_fail_equal():
    assert Atom(1.2) != 2.0


def test_tag1():
    a = Atom(6, tag="my-tag")
    assert a.tag == "my-tag"


def test_negative1():
    a = Atom(-1)
    assert a.symbol == "H"
    assert a.tag == "ghost"
    assert a.Z == 1


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
    a = Atom(5, [Orbital(1.0, 1.0), Orbital(1.0, 1.0), Orbital(2.0), Orbital(2.0)])
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
    o1 = Orbital(1.0, 1.0)
    o2 = Orbital(1.0, 0.5)
    a1 = Atom(5, [o1, o2, o1, o2])
    a2 = Atom(5, [o1, o2, o1, o2, o1, o1])
    assert len(a1.q0) == 4
    assert len(a2.q0) == 6
    assert a1.q0.sum() == pytest.approx(3)
    assert a2.q0.sum() == pytest.approx(5)


def test_multiple_orbitals():
    o = [Orbital(1.0, 1.0), Orbital(2.0, 0.5), Orbital(3.0, 0.75)]
    a1 = Atom(5, o)
    assert len(a1) == 3
    for i in range(3):
        a2 = a1.sub([i])
        assert len(a2) == 1
        assert a2.orbitals[0] == o[i]
        assert a2.orbitals[0] == a2[0]
        a2 = a1.remove([i])
        assert len(a2) == 2

    a2 = a1.sub([1, 2, 0])
    assert a2.orbitals[0] == o[1]
    assert a2[0] == o[1]
    assert a2.orbitals[1] == o[2]
    assert a2[1] == o[2]
    assert a2.orbitals[2] == o[0]
    assert a2[2] == o[0]


def test_multiple_orbitals_fail_io():
    o = [Orbital(1.0, 1.0), Orbital(2.0, 0.5), Orbital(3.0, 0.75)]
    with pytest.raises(ValueError):
        Atom(5, o).sub([3])


def test_multiple_orbitals_fail_len():
    o = [Orbital(1.0, 1.0), Orbital(2.0, 0.5), Orbital(3.0, 0.75)]
    with pytest.raises(ValueError):
        Atom(5, o).sub([0, 0, 0, 0, 0])


def test_atom_getattr_orbs():
    class UOrbital(Orbital):
        def __init__(self, *args, **kwargs):
            U = kwargs.pop("U", 0.0)
            super().__init__(*args, **kwargs)
            self._U = U

        @property
        def U(self):
            return self._U

    o = [
        UOrbital(1.0, 1.0, U=1.0),
        UOrbital(2.0, 0.5, U=2.0),
        UOrbital(3.0, 0.75, U=3.0),
    ]
    a = Atom(5, o)
    assert np.allclose(a.U, [1, 2, 3])
    # also test the callable interface
    assert all(map(lambda x: len(x) == 0, a.name()))


def test_atoms_init_tags():
    a = Atoms([dict(Z="H", tag="hello"), "B", "H"])
    assert len(a) == 3
    assert len(a.atom) == 3
    assert a.atom[0].tag == "hello"
    assert a.atom[1].tag == "B"
    assert a.atom[2].tag == "H"

    a = Atoms(dict(Z="H", tag="hello"))
    assert len(a) == 1
    assert len(a.atom) == 1
    assert a.atom[0].tag == "hello"


def test_atoms_formula():
    a = Atoms([dict(Z="H", tag="hello"), "B", "H"])
    assert len(a) == 3
    assert len(a.atom) == 3
    assert a.formula() == "BH2"


def test_atoms_formula_unknown_system():
    a = Atoms([dict(Z="H", tag="hello"), "B", "H"])
    with pytest.raises(ValueError):
        assert a.formula("not_known")


def test_atom_orbitals():
    a = Atom(5, ["x", "y"])
    assert len(a) == 2
    assert a[0].tag == "x"
    assert a[1].tag == "y"
    assert a.maxR() < 0.0
    a = Atom(5, [1.2, 1.4])
    assert len(a) == 2
    assert a.maxR() == pytest.approx(1.4)


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
