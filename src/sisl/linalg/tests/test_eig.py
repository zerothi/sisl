# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sl

from sisl.linalg import eig, eig_destroy, eigh, eigh_destroy

pytestmark = [pytest.mark.linalg, pytest.mark.eig]


def test_eig1():
    np.random.seed(138012)
    a = np.random.rand(10, 10)
    ac = a.copy()
    b = np.random.rand(10, 10)
    bc = b.copy()
    xs, vs = sl.eig(a, b)
    x, v = eig(a, b)
    assert np.allclose(xs, x)
    assert np.allclose(vs, v)
    assert np.allclose(a, ac)
    assert np.allclose(b, bc)


def test_eigh1():
    np.random.seed(1204982)
    a = np.random.rand(10, 10)
    # Symmetrize
    a = a + a.T
    ac = a.copy()
    xs, vs = sl.eigh(a)
    x, v = eigh(a)
    assert np.allclose(xs, x)
    assert np.allclose(vs, v)
    assert np.allclose(a, ac)


def test_eig_d1():
    np.random.seed(138012)
    a = np.random.rand(10, 10)
    b = np.random.rand(10, 10)
    xs, vs = sl.eig(a, b)
    x, v = eig_destroy(a, b)
    assert np.allclose(xs, x)
    assert np.allclose(vs, v)


def test_eigh_d1():
    np.random.seed(1204982)
    a = np.random.rand(10, 10)
    # Symmetrize
    a = a + a.T
    xs, vs = sl.eigh(a)
    x, v = eigh_destroy(a)
    assert np.allclose(xs, x)
    assert np.allclose(vs, v)
