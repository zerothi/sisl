from __future__ import print_function, division

import pytest

from sisl.io.xyz import *

import numpy as np


pytestmark = pytest.mark.io
_dir = 'sisl/io'


def test_xyz1(sisl_tmp, sisl_system):
    f = sisl_tmp('gr.xyz', _dir)
    sisl_system.g.write(xyzSile(f, 'w'))
    g = xyzSile(f).read_geometry()

    # Assert they are the same
    assert np.allclose(g.cell, sisl_system.g.cell)
    assert np.allclose(g.xyz, sisl_system.g.xyz)
    assert sisl_system.g.atom.equal(g.atom, R=False)
