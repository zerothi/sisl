import pytest
import os.path as osp

from sisl import Atom, Geometry
from sisl.io.siesta.xv import *

import numpy as np

pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = osp.join('sisl', 'io', 'siesta')


def test_xv1(sisl_tmp, sisl_system):
    f = sisl_tmp('gr.XV', _dir)
    sisl_system.g.write(xvSileSiesta(f, 'w'))
    g = xvSileSiesta(f).read_geometry()

    # Assert they are the same
    assert np.allclose(g.cell, sisl_system.g.cell)
    assert np.allclose(g.xyz, sisl_system.g.xyz)
    assert sisl_system.g.atoms.equal(g.atoms, R=False)


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


def test_xv_velocity(sisl_tmp, sisl_system):
    f = sisl_tmp('gr.XV', _dir)
    g = sisl_system.g.copy()
    g.atoms[0] = Atom(1)
    v = np.random.rand(len(g), 3)
    g.write(xvSileSiesta(f, 'w'), velocity=v)

    # Try to read in different ways
    g2 = xvSileSiesta(f).read_geometry()
    assert np.allclose(g.cell, g2.cell)
    assert np.allclose(g.xyz, g2.xyz)
    assert g.atoms.equal(g2.atoms, R=False)

    g2, v2 = xvSileSiesta(f).read_geometry(velocity=True)
    assert np.allclose(g.cell, g2.cell)
    assert np.allclose(g.xyz, g2.xyz)
    assert g.atoms.equal(g2.atoms, R=False)
    assert np.allclose(v, v2)

    # Try to read in different ways
    v2 = xvSileSiesta(f).read_velocity()
    assert np.allclose(v, v2)


def test_xv_ghost(sisl_tmp):
    f = sisl_tmp('ghost.XV', _dir)
    a1 = Atom(1)
    am1 = Atom(-1)
    g = Geometry([[0., 0., i] for i in range(2)], [a1, am1], 2.)
    g.write(xvSileSiesta(f, 'w'))

    g2 = xvSileSiesta(f).read_geometry()
    assert np.allclose(g.cell, g2.cell)
    assert np.allclose(g.xyz, g2.xyz)
    assert np.allclose(g.atoms.Z, g2.atoms.Z)
    assert g.atoms[0].__class__ is g2.atoms[0].__class__
    assert g.atoms[1].__class__ is g2.atoms[1].__class__
    assert g.atoms[0].__class__ is not g2.atoms[1].__class__
