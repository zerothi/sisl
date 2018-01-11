from __future__ import print_function, division

import pytest

import warnings
import math as m
import numpy as np

from sisl.physics.distribution_function import *

pytestmark = pytest.mark.distributions


def test_distribution1():
    x = np.linspace(-2, 2, 10000)
    dx = x[1] - x[0]
    d = distribution('gaussian', smearing=0.025)
    assert d(x).sum() * dx == pytest.approx(1, abs=1e-6)
    d = distribution('lorentzian', smearing=1e-3)
    assert d(x).sum() * dx == pytest.approx(1, abs=1e-3)


def test_distribution2():
    x = np.linspace(-2, 2, 10000)
    d = distribution('gaussian', smearing=0.025)
    assert np.allclose(d(x), gaussian(x, 0.025))
    d = distribution('lorentzian', smearing=1e-3)
    assert np.allclose(d(x), lorentzian(x, 1e-3))


@pytest.mark.xfail(raises=ValueError)
def test_distribution3():
    distribution('unknown-function')
