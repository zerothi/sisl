# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl.utils.mathematics import *

pytestmark = pytest.mark.utils


def test_curl_2d():
    a = np.random.rand(3, 3)
    C = curl(a)
    assert C.shape == (3,)
    c = [a[1, 2] - a[2, 1], a[2, 0] - a[0, 2], a[0, 1] - a[1, 0]]
    assert np.allclose(C, c)


def test_curl_3d():
    a = np.random.rand(4, 3, 3)
    C = curl(a)
    assert C.shape == (4, 3)


def test_curl_4d():
    a = np.random.rand(4, 3, 4, 3)
    C1 = curl(a, axis=1, axisv=3)
    C2 = curl(a, axis=1)
    assert C1.shape == (4, 4, 3)
    assert np.allclose(C1, C2)

    b = np.swapaxes(a, 1, 2)
    C3 = curl(b)
    # no need to swap, we are removing the 2nd last axis
    assert np.allclose(C1, C3)


def test_curl_6d():
    a = np.random.rand(4, 3, 4, 10, 3, 3)
    C1 = curl(a, axis=1, axisv=5)
    assert C1.shape == (4, 4, 10, 3, 3)
    C2 = curl(a, axis=1)
    assert C2.shape == (4, 4, 10, 3, 3)
    assert np.allclose(C1, C2)

    C2 = curl(a, axis=-1, axisv=1)
    assert C2.shape == (4, 3, 4, 10, 3)


def test_curl_4d_same():
    # same axis specification
    a = np.random.rand(4, 3, 4, 3)
    with pytest.raises(ValueError):
        curl(a, axis=1, axisv=1)


def test_curl_4d_not_3():
    # axis lengths must equal 3 (nothing else is supported currently)
    a = np.random.rand(4, 3, 4, 3)
    with pytest.raises(ValueError):
        curl(a, axis=1, axisv=2)


@pytest.mark.parametrize("nd", [0, 1, 2, 3])
def test_cart2spher_nd(nd):
    r = np.random.rand(*([3] * (nd + 1)))
    R, theta, phi = cart2spher(r)
    assert R.ndim == nd
    assert theta.ndim == nd
    assert phi.ndim == nd


def test_cart2spher_nd_maxr():
    r = np.random.rand(3, 3)
    idx, R, theta, phi = cart2spher(r, maxR=0.5)
    assert idx.ndim == 1
    assert R.ndim == 1
    assert theta.ndim == 1
    assert phi.ndim == 1


def test_rotation_matrix():
    angles = [10, 40, 59]
    R1 = rotation_matrix(*angles, order="xyz")
    R2 = rotation_matrix(*list(map(lambda x: -x, angles)), order="zyx")
    assert np.allclose(R1 @ R2, np.identity(3))

    R1 = rotation_matrix(*angles, order="xz")
    R2 = rotation_matrix(*list(map(lambda x: -x, angles)), order="zx")
    assert np.allclose(R1 @ R2, np.identity(3))
