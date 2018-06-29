from __future__ import print_function, division

import pytest

import math as m
import numpy as np

from sisl.utils.mathematics import *


pytestmark = pytest.mark.utils


def test_curl_2d():
    a = np.random.rand(3, 3)
    C = curl(a)
    assert C.shape == (3, 3)
    cr = np.cross
    c = [cr(a[1, :], a[2, :]), cr(a[2, :], a[0, :]), cr(a[0, :], a[1, :])]
    assert np.allclose(C, c)


def test_curl_3d():
    a = np.random.rand(4, 3, 3)
    C = curl(a)
    assert C.shape == (4, 3, 3)
