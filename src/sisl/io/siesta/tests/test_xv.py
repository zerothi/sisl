# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl import Atom, Atoms, AtomUnknown, Geometry
from sisl.io.siesta.xv import *

pytestmark = [pytest.mark.io, pytest.mark.siesta]


def test_xv1(sisl_tmp, sisl_system):
    f = sisl_tmp("gr.XV")
    sisl_system.g.write(xvSileSiesta(f, "w"))
    g = xvSileSiesta(f).read_geometry()

    # Assert they are the same
    assert np.allclose(g.cell, sisl_system.g.cell)
    assert np.allclose(g.xyz, sisl_system.g.xyz)
    assert sisl_system.g.atoms.equal(g.atoms, R=False)


def test_xv_reorder(sisl_tmp, sisl_system):
    f = sisl_tmp("gr.XV")
    g = sisl_system.g.copy()
    g.atoms[0] = Atom(1)
    g.write(xvSileSiesta(f, "w"))
    g2 = xvSileSiesta(f).read_geometry()

    # Assert they are the same
    assert np.allclose(g.cell, g2.cell)
    assert np.allclose(g.xyz, g2.xyz)
    assert g.atoms.equal(g2.atoms, R=False)


def test_xv_velocity(sisl_tmp, sisl_system):
    f = sisl_tmp("gr.XV")
    g = sisl_system.g.copy()
    g.atoms[0] = Atom(1)
    v = np.random.rand(len(g), 3)
    g.write(xvSileSiesta(f, "w"), velocity=v)

    # Try to read in different ways
    g2 = xvSileSiesta(f).read_geometry()
    assert np.allclose(g.cell, g2.cell)
    assert np.allclose(g.xyz, g2.xyz)
    assert g.atoms.equal(g2.atoms, R=False)

    g2, v2 = xvSileSiesta(f).read_geometry(ret_velocity=True)
    assert np.allclose(g.cell, g2.cell)
    assert np.allclose(g.xyz, g2.xyz)
    assert g.atoms.equal(g2.atoms, R=False)
    assert np.allclose(v, v2)

    # Try to read in different ways
    v2 = xvSileSiesta(f).read_velocity()
    assert np.allclose(v, v2)


def test_xv_ghost(sisl_tmp):
    f = sisl_tmp("ghost.XV")
    a1 = Atom(1)
    am1 = Atom(-1)
    g = Geometry([[0.0, 0.0, i] for i in range(2)], [a1, am1], 2.0)
    g.write(xvSileSiesta(f, "w"))

    g2 = xvSileSiesta(f).read_geometry()
    assert np.allclose(g.cell, g2.cell)
    assert np.allclose(g.xyz, g2.xyz)
    assert np.allclose(g.atoms.Z, g2.atoms.Z)
    assert g.atoms[0].__class__ is g2.atoms[0].__class__
    assert g.atoms[1].__class__ is g2.atoms[1].__class__
    assert g.atoms[0].__class__ is not g2.atoms[1].__class__


def test_xv_missing_atoms(sisl_tmp):
    # test for #778
    f = sisl_tmp("missing.XV")
    with open(f, "w") as fh:
        fh.write(
            """\
1. 0. 0.  0. 0. 0.
0. 1. 0.  0. 0. 0.
0. 0. 2.  0. 0. 0.
6
2 6 0. 1. 0.  0. 0. 0.
1 2 0. 1. 0.  0. 0. 0.
4 3 0. 1. 0.  0. 0. 0.
4 3 0. 1. 0.  0. 0. 0.
1 2 0. 1. 0.  0. 0. 0.
2 6 0. 1. 0.  0. 0. 0.
"""
        )
    geom = xvSileSiesta(f).read_geometry()
    assert len(geom) == 6
    assert len(geom.atoms.atom) == 4
    assert np.allclose(geom.atoms.species, [1, 0, 3, 3, 0, 1])

    # start_Z + sp_idx
    atom = AtomUnknown(1000 + 2)
    assert geom.atoms.atom[2].Z == atom.Z
    assert isinstance(geom.atoms.atom[2], atom.__class__)


def test_xv_missing_atoms_end(sisl_tmp):
    # test for #778
    f = sisl_tmp("missing_end.XV")
    with open(f, "w") as fh:
        fh.write(
            """\
1. 0. 0.  0. 0. 0.
0. 1. 0.  0. 0. 0.
0. 0. 2.  0. 0. 0.
6
2 6 0. 1. 0.  0. 0. 0.
1 2 0. 1. 0.  0. 0. 0.
1 2 0. 1. 0.  0. 0. 0.
3 3 0. 1. 0.  0. 0. 0.
3 3 0. 1. 0.  0. 0. 0.
2 6 0. 1. 0.  0. 0. 0.
"""
        )
    atoms = Atoms([2, 6, 3, 5])
    geom = xvSileSiesta(f).read_geometry(atoms=atoms)
    assert len(geom) == 6
    assert len(geom.atoms.atom) == 4
    assert np.allclose(geom.atoms.species, [1, 0, 0, 2, 2, 1])


def test_xv_missing_atoms_species(sisl_tmp):
    # test for #778
    f = sisl_tmp("missing_species.XV")
    with open(f, "w") as fh:
        fh.write(
            """\
1. 0. 0.  0. 0. 0.
0. 1. 0.  0. 0. 0.
0. 0. 2.  0. 0. 0.
6
2 1 0. 1. 0.  0. 0. 0.
1 1 0. 1. 0.  0. 0. 0.
1 1 0. 1. 0.  0. 0. 0.
3 1 0. 1. 0.  0. 0. 0.
3 1 0. 1. 0.  0. 0. 0.
2 1 0. 1. 0.  0. 0. 0.
"""
        )

    atoms = Atoms([Atom(1, tag=tag) for tag in "ABCD"])
    geom = xvSileSiesta(f).read_geometry(atoms=atoms)
    assert len(geom) == 6
    assert len(geom.atoms.atom) == 4
    assert np.allclose(geom.atoms.species, [1, 0, 0, 2, 2, 1])
