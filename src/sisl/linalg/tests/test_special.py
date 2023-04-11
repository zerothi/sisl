# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pytest

import numpy as np
import scipy.linalg as sl
from sisl.linalg import sqrth, invsqrth, signsqrt

pytestmark = [pytest.mark.linalg]


def test_sqrth():
    np.random.seed(1285947159)
    a = np.random.rand(10, 10)
    a = a + a.T
    np.fill_diagonal(a, 4)
    sa1 = sl.sqrtm(a)
    sa2 = sqrth(a)
    assert np.allclose(sa1, sa2)
    assert np.allclose(sa2 @ sa2, a)


def test_sqrth_negative():
    np.random.seed(1285947159)
    a = np.random.rand(10, 10)
    a = a + a.T
    np.fill_diagonal(a, a.diagonal() - 1)
    sa1 = sl.sqrtm(a)
    sa2 = sqrth(a)
    assert np.allclose(sa1, sa2)
    assert np.allclose(sa2 @ sa2, a)


def test_invsqrth():
    np.random.seed(1285947159)
    a = np.random.rand(10, 10)
    a = a + a.T
    np.fill_diagonal(a, 4)
    sa1 = sl.inv(sl.sqrtm(a))
    sa2 = invsqrth(a)
    assert np.allclose(sa1, sa2)


def test_invsqrth_offset():
    # offsetting eigenvalues only works if the matrix is
    # positive semi-definite
    np.random.seed(1285947159)
    a = np.random.rand(10, 10)
    a = a + a.T
    # without diagonal filling
    np.fill_diagonal(a, a.diagonal() + 4)
    sa1 = invsqrth(a)

    # similar thing, with diagonal
    np.fill_diagonal(a, a.diagonal() + 4)
    eig, ev = sl.eigh(a)
    eig = signsqrt(eig - 4)
    np.divide(1, eig, where=(eig != 0), out=eig)
    sa2 = (ev * eig) @ ev.conj().T
    assert np.allclose(sa1, sa2)
