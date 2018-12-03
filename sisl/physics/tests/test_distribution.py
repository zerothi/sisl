from __future__ import print_function, division

import pytest

import math as m
import numpy as np

from sisl.physics.distribution import *

pytestmark = pytest.mark.distribution


def test_distribution1():
    x = np.linspace(-2, 2, 10000)
    dx = x[1] - x[0]
    d = get_distribution('gaussian', smearing=0.025)
    assert d(x).sum() * dx == pytest.approx(1, abs=1e-6)
    d = get_distribution('lorentzian', smearing=1e-3)
    assert d(x).sum() * dx == pytest.approx(1, abs=1e-3)


def test_distribution2():
    x = np.linspace(-2, 2, 10000)
    d = get_distribution('gaussian', smearing=0.025)
    assert np.allclose(d(x), gaussian(x, 0.025))
    d = get_distribution('lorentzian', smearing=1e-3)
    assert np.allclose(d(x), lorentzian(x, 1e-3))
    d = get_distribution('step', smearing=1e-3)
    assert np.allclose(d(x), step_function(x))
    d = get_distribution('heaviside', smearing=1e-3, x0=1)
    assert np.allclose(d(x), 1 - step_function(x, 1))
    d = get_distribution('heaviside', x0=-0.5)
    assert np.allclose(d(x), heaviside(x, -0.5))


@pytest.mark.xfail(raises=ValueError)
def test_distribution3():
    get_distribution('unknown-function')


def test_distribution_x0():
    x1 = np.linspace(-2, 2, 10000)
    x2 = np.linspace(-3, 1, 10000)
    d1 = get_distribution('gaussian')
    d2 = get_distribution('gaussian', x0=-1)
    assert np.allclose(d1(x1), d2(x2))
    d1 = get_distribution('lorentzian')
    d2 = get_distribution('lorentzian', x0=-1)
    assert np.allclose(d1(x1), d2(x2))


def test_fermi_dirac():
    E1 = np.linspace(-2, 2, 1000)
    E2 = np.linspace(-3, 1, 1000)
    assert np.allclose(fermi_dirac(E1, 0.1), fermi_dirac(E2, 0.1, -1))


def test_bose_einstein():
    E1 = np.linspace(-2, 2, 1000)
    E2 = np.linspace(-3, 1, 1000)
    assert np.allclose(bose_einstein(E1, 0.1), bose_einstein(E2, 0.1, -1))


def test_cold():
    E1 = np.linspace(-2, 2, 1000)
    E2 = np.linspace(-3, 1, 1000)
    assert np.allclose(cold(E1, 0.1), cold(E2, 0.1, -1))
