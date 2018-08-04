from __future__ import print_function, division

import pytest

import math as m
import numpy as np

from sisl import geom, Geometry, Spin
from sisl.physics.sparse import SparseOrbitalBZ, SparseOrbitalBZSpin

pytestmark = pytest.mark.sparse


def _get():
    gr = geom.graphene()
    return gr


def test_str():
    gr = _get()
    # The most simple setup.
    sp = SparseOrbitalBZ(gr)
    str(sp)


def test_S():
    gr = _get()
    # The most simple setup.
    sp = SparseOrbitalBZ(gr, orthogonal=False)
    sp[0, 0] = 0.5
    sp[1, 1] = 0.5
    sp.S[0, 0] = 1.
    sp.S[1, 1] = 1.
    assert sp[1, 1, 0] == pytest.approx(0.5)
    assert sp.S[1, 1] == pytest.approx(1.)


def test_eigh_orthogonal():
    gr = _get()
    # The most simple setup.
    sp = SparseOrbitalBZ(gr)
    sp[0, 0] = 0.5
    sp[1, 1] = 0.5
    assert np.allclose(sp.eigh(), [0.5, 0.5])
    assert np.allclose(sp.eigh([0.5] * 3), [0.5, 0.5])


def test_eigh_non_orthogonal():
    gr = _get()
    # The most simple setup.
    sp = SparseOrbitalBZ(gr, orthogonal=False)
    sp[0, 0] = 0.5
    sp[1, 1] = 0.5
    sp.S[0, 0] = 1.
    sp.S[1, 1] = 1.
    assert np.allclose(sp.eigh(), [0.5, 0.5])


@pytest.mark.xfail
def test_eigsh_orthogonal():
    gr = _get()
    # The most simple setup.
    sp = SparseOrbitalBZ(gr)
    sp[0, 0] = 0.5
    sp[1, 1] = 0.5
    # Fails due to too many requested eigenvalues
    sp.eigsh()


@pytest.mark.xfail
def test_eigsh_non_orthogonal():
    gr = _get()
    # The most simple setup.
    sp = SparseOrbitalBZ(gr, orthogonal=False)
    sp.eigsh()
