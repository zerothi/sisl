# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import pytest

from sisl.io.siesta import aniSileSiesta

pytestmark = [pytest.mark.io, pytest.mark.siesta]


def test_ani(sisl_tmp):
    f = sisl_tmp("sisl.ANI")
    open(f, "w").write(
        """1

C   0.00000000  0.00000000  0.00000000
2

C   0.00000000  0.00000000  0.00000000
C   1.000000  0.00000000  0.00000000
3

C   0.00000000  0.00000000  0.00000000
C   1.000000  0.00000000  0.00000000
C   2.00000  0.00000000  0.00000000
4

C   0.00000000  0.00000000  0.00000000
C   1.000000  0.00000000  0.00000000
C   2.00000  0.00000000  0.00000000
C   3.00000  0.00000000  0.00000000
"""
    )
    a = aniSileSiesta(f)
    g = a.read_geometry[:]()
    assert len(g) == 4
    assert g[0].na == 1

    g = a.read_geometry[1:]()
    assert len(g) == 3
    assert g[0].na == 2

    g = a.read_geometry[:]()
    assert len(g) == 4
    assert g[0].na == 1 and g[2].na == 3

    g = a.read_geometry[1::1]()
    assert len(g) == 3
    assert g[0].na == 2 and g[1].na == 3

    g = a.read_geometry[::2]()
    assert len(g) == 2
    assert g[0].na == 1 and g[1].na == 3

    g = a.read_geometry[:2:1]()
    assert len(g) == 2
    assert g[0].na == 1 and g[1].na == 2

    g = a.read_geometry[1:]()
    assert g[0].na == 2 and len(g) == 3

    g = a.read_geometry[1]()
    assert g.na == 2

    g = a.read_geometry[1:3:1]()
    assert g[1].na == 3 and len(g) == 2

    g = a.read_geometry[1:3:2]()
    assert g[0].na == 2 and len(g) == 1

    g = a.read_geometry[-2:]()
    assert g[0].na == 3 and len(g) == 2
    assert g[1].na == 4

    g = a.read_geometry[-2::-1]()
    assert g[0].na == 3 and len(g) == 3
    assert g[1].na == 2
    assert g[2].na == 1

    g = a.read_geometry(lattice=None, atoms=None)
