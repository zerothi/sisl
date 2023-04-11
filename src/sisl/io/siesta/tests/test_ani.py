# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pytest
import os.path as osp
import numpy as np
from sisl.io.siesta import aniSileSiesta
from sisl import Geometry, GeometryCollection

pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = osp.join('sisl', 'io', 'siesta')

def test_ani(sisl_tmp):
    f = sisl_tmp('sisl.ANI', _dir)
    open(f, 'w').write("""1

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
""")
    a = aniSileSiesta(f)
    g = a.read_geometry()
    assert isinstance(g, GeometryCollection)
    assert len(g) == 4
    assert g[0].na == 1

    g = a.read_geometry(start=1)
    assert len(g) == 3
    assert g[0].na == 2

    g = a.read_geometry(all=True)
    assert len(g) == 4
    assert g[0].na == 1 and g[2].na == 3

    g = a.read_geometry(start=1, step=1)
    assert len(g) == 3
    assert g[0].na == 2 and g[1].na == 3

    g = a.read_geometry(step=2)
    assert len(g) == 2
    assert g[0].na == 1 and g[1].na == 3

    g = a.read_geometry(stop=2, step=1)
    assert len(g) == 2
    assert g[0].na == 1 and g[1].na == 2

    g = a.read_geometry(start=1, step=None)
    assert g[0].na == 2 and len(g) == 3

    g = a.read_geometry(start=1, all=False)
    assert g.na == 2

    g = a.read_geometry(start=1, stop=3, step=1)
    assert g[1].na == 3 and len(g) == 2

    g = a.read_geometry(start=1, stop=3, step=2, all=True)
    assert g[0].na == 2 and len(g) == 1

    g = a.read_geometry(lattice=None, atoms=None)
