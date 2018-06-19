from __future__ import print_function, division

import pytest

from sisl.atom import Atom
from sisl.io.siesta.xv import *

import numpy as np

pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = 'sisl/io/siesta'


def test_xv1(sisl_tmp, sisl_system):
    f = sisl_tmp('gr.XV', _dir)
    sisl_system.g.write(xvSileSiesta(f, 'w'))
    g = xvSileSiesta(f).read_geometry()

    # Assert they are the same
    assert np.allclose(g.cell, sisl_system.g.cell)
    assert np.allclose(g.xyz, sisl_system.g.xyz)
    assert sisl_system.g.atom.equal(g.atom, R=False)


def test_xv_reorder(sisl_tmp, sisl_system):
    f = sisl_tmp('gr.XV', _dir)
    g = sisl_system.g.copy()
    g.atoms[0] = Atom(1)
    g.write(xvSileSiesta(f, 'w'))
    g2 = xvSileSiesta(f).read_geometry()

    # Assert they are the same
    assert np.allclose(g.cell, g2.cell)
    assert np.allclose(g.xyz, g2.xyz)
    assert g.atoms.equal(g2.atoms, R=False)
