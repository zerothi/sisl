import pytest

import numpy as np
import scipy.linalg as sl
from sisl.linalg import inv, inv_destroy

pytestmark = [pytest.mark.linalg, pytest.mark.inv]


def test_inv1():
    np.random.seed(1204982)
    a = np.random.rand(10, 10)
    ac = a.copy()
    xs = sl.inv(a)
    x = inv(a)
    assert np.allclose(xs, x)
    assert np.allclose(a, ac)


def test_inv_d1():
    np.random.seed(1204982)
    a = np.random.rand(10, 10)
    xs = sl.inv(a)
    x = inv_destroy(a)
    assert np.allclose(xs, x)
