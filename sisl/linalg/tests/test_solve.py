from __future__ import print_function, division

import pytest

import numpy as np
import scipy.linalg as sl
from sisl.linalg import solve, solve_destroy

pytestmark = [pytest.mark.linalg, pytest.mark.eig]


def test_solve1():
    a = np.random.rand(10, 10)
    b = np.random.rand(10, 10)
    xs = sl.solve(a, b)
    x = solve(a, b)
    assert np.allclose(xs, x)


def test_solve2():
    a = np.random.rand(10, 10)
    ac = a.copy()
    b = np.random.rand(10)
    bc = b.copy()
    xs = sl.solve(a, b)
    x = solve(a, b)
    assert np.allclose(xs, x)
    assert x.shape == (10, )
    assert np.allclose(a, ac)
    assert np.allclose(b, bc)


@pytest.mark.xfail(raises=ValueError)
def test_solve3():
    a = np.random.rand(10, 2)
    b = np.random.rand(10)
    solve(a, b)


def test_solve4():
    a = np.random.rand(10, 10)
    b = np.random.rand(10)
    xs = sl.solve(a, b)
    x = solve_destroy(a, b)
    assert np.allclose(xs, x)
    assert x.shape == (10, )
