# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sl

from sisl.linalg import solve, solve_destroy

pytestmark = [pytest.mark.linalg, pytest.mark.eig]


def test_solve1():
    np.random.seed(1285947159)
    a = np.random.rand(10, 10)
    b = np.random.rand(10, 10)
    xs = sl.solve(a, b)
    x = solve(a, b)
    assert np.allclose(xs, x)


def test_solve2():
    np.random.seed(1285947159)
    a = np.random.rand(10, 10)
    ac = a.copy()
    b = np.random.rand(10)
    bc = b.copy()
    xs = sl.solve(a, b)
    x = solve(a, b)
    assert np.allclose(xs, x)
    assert x.shape == (10,)
    assert np.allclose(a, ac)
    assert np.allclose(b, bc)


def test_solve3():
    np.random.seed(1285947159)
    a = np.random.rand(10, 2)
    b = np.random.rand(10)
    with pytest.raises(ValueError):
        solve(a, b)


def test_solve4():
    np.random.seed(1285947159)
    a = np.random.rand(10, 10)
    b = np.random.rand(10)
    xs = sl.solve(a, b)
    x = solve_destroy(a, b)
    assert np.allclose(xs, x)
    assert x.shape == (10,)
