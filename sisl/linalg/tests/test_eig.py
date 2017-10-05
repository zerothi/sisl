from __future__ import print_function, division

import pytest

import numpy as np
import scipy.linalg as sl
from sisl.linalg import (eig, eig_destroy,
                         eigh, eigh_destroy,
                         eigh_qr, eigh_dc)

pytestmark = [pytest.mark.linalg, pytest.mark.eig]


def test_eig1():
    np.random.seed(1204982)
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

    x, v = eigh_dc(a)
    assert np.allclose(xs, x)
    assert np.allclose(vs, v)
    assert np.allclose(a, ac)

    x, v = eigh_qr(a)
    assert np.allclose(xs, x)
    assert np.allclose(vs, v)
    assert np.allclose(a, ac)


def test_eig_d1():
    np.random.seed(1204982)
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
