# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl import Atom, Geometry, get_sile
from sisl.io.siesta.struct import *

pytestmark = [pytest.mark.io, pytest.mark.siesta]


def test_struct1(sisl_tmp, sisl_system):
    f = sisl_tmp("gr.STRUCT_IN")
    sisl_system.g.write(structSileSiesta(f, "w"))
    g = structSileSiesta(f).read_geometry()

    # Assert they are the same
    assert np.allclose(g.cell, sisl_system.g.cell)
    assert np.allclose(g.xyz, sisl_system.g.xyz)
    assert sisl_system.g.atoms.equal(g.atoms, R=False)


def test_struct_reorder(sisl_tmp, sisl_system):
    f = sisl_tmp("gr.STRUCT_IN")
    g = sisl_system.g.copy()
    g.atoms[0] = Atom(1)
    g.write(structSileSiesta(f, "w"))
    g2 = structSileSiesta(f).read_geometry()

    # Assert they are the same
    assert np.allclose(g.cell, g2.cell)
    assert np.allclose(g.xyz, g2.xyz)
    assert g.atoms.equal(g2.atoms, R=False)


def test_struct_ghost(sisl_tmp):
    f = sisl_tmp("ghost.STRUCT_IN")
    a1 = Atom(1)
    am1 = Atom(-1)
    g = Geometry([[0.0, 0.0, i] for i in range(2)], [a1, am1], 2.0)
    g.write(structSileSiesta(f, "w"))

    g2 = structSileSiesta(f).read_geometry()
    assert np.allclose(g.cell, g2.cell)
    assert np.allclose(g.xyz, g2.xyz)
    assert np.allclose(g.atoms.Z, g2.atoms.Z)
    assert g.atoms[0].__class__ is g2.atoms[0].__class__
    assert g.atoms[1].__class__ is g2.atoms[1].__class__
    assert g.atoms[0].__class__ is not g2.atoms[1].__class__


def test_si_pdos_kgrid_struct_out(sisl_files):
    fdf = get_sile(sisl_files("siesta", "Si_pdos_k", "Si_pdos.fdf"))
    struct = get_sile(sisl_files("siesta", "Si_pdos_k", "Si_pdos.STRUCT_OUT"))

    struct_geom = struct.read_geometry()
    fdf_geom = fdf.read_geometry(order="STRUCT")

    assert np.allclose(struct_geom.cell, fdf_geom.cell)
    assert np.allclose(struct_geom.xyz, fdf_geom.xyz)

    struct_sc = struct.read_lattice()
    fdf_sc = fdf.read_lattice(order="STRUCT")

    assert np.allclose(struct_sc.cell, fdf_sc.cell)
