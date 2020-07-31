import pytest

import math as m
import numpy as np

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
