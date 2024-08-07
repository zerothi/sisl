# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl import Atom
from sisl.io.fhiaims._geometry import *

pytestmark = [pytest.mark.io, pytest.mark.fhiaims, pytest.mark.aims]


def test_in_simple(sisl_tmp, sisl_system):
    f = sisl_tmp("gr.in")
    sisl_system.g.write(inSileFHIaims(f, "w"))
    g = inSileFHIaims(f).read_geometry()

    # Assert they are the same
    assert np.allclose(g.cell, sisl_system.g.cell)
    assert np.allclose(g.xyz, sisl_system.g.xyz)
    assert sisl_system.g.atoms.equal(g.atoms, R=False)


def test_in_velocity(sisl_tmp, sisl_system):
    f = sisl_tmp("gr.in")
    g = sisl_system.g.copy()
    g.atoms[0] = Atom(1)
    v = np.random.rand(len(g), 3)
    g.write(inSileFHIaims(f, "w"), velocity=v)

    # Try to read in different ways
    g2 = inSileFHIaims(f).read_geometry()
    assert np.allclose(g.cell, g2.cell)
    assert np.allclose(g.xyz, g2.xyz)
    assert g.atoms.equal(g2.atoms, R=False)

    g2, v2 = inSileFHIaims(f).read_geometry(ret_velocity=True)
    assert np.allclose(g.cell, g2.cell)
    assert np.allclose(g.xyz, g2.xyz)
    assert g.atoms.equal(g2.atoms, R=False)
    assert np.allclose(v, v2)

    # Try to read in different ways
    v2 = inSileFHIaims(f).read_velocity()
    assert np.allclose(v, v2)


def test_in_moment(sisl_tmp, sisl_system):
    f = sisl_tmp("gr.in")
    g = sisl_system.g.copy()
    g.atoms[0] = Atom(1)
    m = np.random.rand(len(g))
    g.write(inSileFHIaims(f, "w"), moment=m)

    # Try to read in different ways
    g2 = inSileFHIaims(f).read_geometry()
    assert np.allclose(g.cell, g2.cell)
    assert np.allclose(g.xyz, g2.xyz)
    assert g.atoms.equal(g2.atoms, R=False)

    g2, m2 = inSileFHIaims(f).read_geometry(ret_moment=True)
    assert np.allclose(g.cell, g2.cell)
    assert np.allclose(g.xyz, g2.xyz)
    assert g.atoms.equal(g2.atoms, R=False)
    assert np.allclose(m, m2)

    # Try to read in different ways
    m2 = inSileFHIaims(f).read_moment()
    assert np.allclose(m, m2)


def test_in_v_m(sisl_tmp, sisl_system):
    f = sisl_tmp("gr.in")
    g = sisl_system.g.copy()
    g.atoms[0] = Atom(1)
    v = np.random.rand(len(g), 3)
    m = np.random.rand(len(g))
    g.write(inSileFHIaims(f, "w"), velocity=v, moment=m)

    # Try to read in different ways
    g2 = inSileFHIaims(f).read_geometry()
    assert np.allclose(g.cell, g2.cell)
    assert np.allclose(g.xyz, g2.xyz)
    assert g.atoms.equal(g2.atoms, R=False)

    g2, m2 = inSileFHIaims(f).read_geometry(ret_moment=True)
    assert np.allclose(g.cell, g2.cell)
    assert np.allclose(g.xyz, g2.xyz)
    assert g.atoms.equal(g2.atoms, R=False)
    assert np.allclose(m, m2)

    g2, v2 = inSileFHIaims(f).read_geometry(ret_velocity=True)
    assert np.allclose(g.cell, g2.cell)
    assert np.allclose(g.xyz, g2.xyz)
    assert g.atoms.equal(g2.atoms, R=False)
    assert np.allclose(v, v2)

    g2, v2, m2 = inSileFHIaims(f).read_geometry(ret_velocity=True, ret_moment=True)
    assert np.allclose(g.cell, g2.cell)
    assert np.allclose(g.xyz, g2.xyz)
    assert g.atoms.equal(g2.atoms, R=False)
    assert np.allclose(v, v2)
    assert np.allclose(m, m2)
