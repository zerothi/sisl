# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl.io.xyz import *

pytestmark = [pytest.mark.io, pytest.mark.generic]


def test_xyz1(sisl_tmp, sisl_system):
    f = sisl_tmp("gr.xyz")
    sisl_system.g.write(xyzSile(f, "w"))
    g = xyzSile(f).read_geometry()

    # Assert they are the same
    assert np.allclose(g.cell, sisl_system.g.cell)
    assert np.allclose(g.xyz, sisl_system.g.xyz)
    assert sisl_system.g.atoms.equal(g.atoms, R=False)
    g = xyzSile(f).read_geometry(lattice=g.lattice)
    assert np.allclose(g.cell, sisl_system.g.cell)


def test_xyz_sisl(sisl_tmp):
    f = sisl_tmp("sisl.xyz")

    with open(f, "w") as fh:
        fh.write(
            """3
sisl-version=1 nsc=1 1 3 cell=10 0 0 0 12 0 0 0 13
C   0.00000000  0.00000000  0.00000000
C   1.000000  0.00000000  0.00000000
C   2.00000  0.00000000  0.00000000
"""
        )
    g = xyzSile(f).read_geometry()

    # Assert they are the same
    assert np.allclose(g.cell, [[10, 0, 0], [0, 12, 0], [0, 0, 13]])
    assert np.allclose(g.xyz[:, 0], [0, 1, 2])
    assert np.allclose(g.xyz[:, 1], 0.0)
    assert np.allclose(g.xyz[:, 2], 0.0)
    assert np.allclose(g.nsc, [1, 1, 3])

    g = xyzSile(f).read_geometry(lattice=[10, 11, 13])
    assert np.allclose(g.cell, [[10, 0, 0], [0, 11, 0], [0, 0, 13]])


def test_xyz_ase(sisl_tmp):
    f = sisl_tmp("ase.xyz")
    with open(f, "w") as fh:
        fh.write(
            """3
Lattice="10 0 0 0 12 0 0 0 13" Properties=species:S:1:pos:R:3 pbc="F F T"
C   0.00000000  0.00000000  0.00000000
C   1.000000  0.00000000  0.00000000
C   2.00000  0.00000000  0.00000000
"""
        )
    g = xyzSile(f).read_geometry()

    # Assert they are the same
    assert np.allclose(g.cell, [[10, 0, 0], [0, 12, 0], [0, 0, 13]])
    assert np.allclose(g.xyz[:, 0], [0, 1, 2])
    assert np.allclose(g.xyz[:, 1], 0.0)
    assert np.allclose(g.xyz[:, 2], 0.0)
    assert np.allclose(g.nsc, [1, 1, 1])
    assert np.allclose(g.pbc, [False, False, True])


def test_xyz_arbitrary(sisl_tmp):
    f = sisl_tmp("ase.xyz")
    with open(f, "w") as fh:
        fh.write(
            """3

C   0.00000000  0.00000000  0.00000000
C   1.000000  0.00000000  0.00000000
C   2.00000  0.00000000  0.00000000
"""
        )
    g = xyzSile(f).read_geometry()

    # Assert they are the same
    assert np.allclose(g.xyz[:, 0], [0, 1, 2])
    assert np.allclose(g.xyz[:, 1], 0.0)
    assert np.allclose(g.xyz[:, 2], 0.0)
    assert np.allclose(g.nsc, [1, 1, 1])


def test_xyz_multiple(sisl_tmp):
    f = sisl_tmp("sisl_multiple.xyz")
    with open(f, "w") as fh:
        fh.write(
            """1

C   0.00000000  0.00000000  0.00000000
2

C   0.00000000  0.00000000  0.00000000
C   1.000000  0.00000000  0.00000000
3

C   0.00000000  0.00000000  0.00000000
C   1.000000  0.00000000  0.00000000
C   2.00000  0.00000000  0.00000000
"""
        )
    g = xyzSile(f).read_geometry()
    assert g.na == 1
    g = xyzSile(f).read_geometry[1]()
    assert g.na == 2
    g = xyzSile(f).read_geometry[:]()
    assert len(g) == 3
    assert g[0].na == 1 and g[1].na == 2 and g[-1].na == 3
    g = xyzSile(f).read_geometry[1:]()
    assert len(g) == 2
    assert g[0].na == 2 and g[-1].na == 3
    g = xyzSile(f).read_geometry[1:-1]()
    assert len(g) == 1
    assert g[0].na == 2
    g = xyzSile(f).read_geometry[::2]()
    assert len(g) == 2
    assert g[0].na == 1 and g[-1].na == 3
    g = xyzSile(f).read_geometry[:2:1]()
    assert len(g) == 2
    assert g[0].na == 1 and g[1].na == 2
    g = xyzSile(f).read_geometry[1:3:1]()
    assert len(g) == 2
    assert g[0].na == 2 and g[1].na == 3

    g1 = xyzSile(f).read_geometry[-1]()
    g2 = xyzSile(f).read_geometry.last()
    assert g1 == g2

    # ensure it works with other arguments
    g = xyzSile(f).read_geometry(lattice=None, atoms=None)
