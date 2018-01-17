from __future__ import print_function, division

import pytest

import math as m
import numpy as np

from sisl.shape.ellipsoid import *

pytestmark = pytest.mark.shape


def test_create_ellipsoid():
    el = Ellipsoid([1., 1., 1.])
    el = Ellipsoid([1., 1., 1.], [1.] * 3)
    el = Ellipsoid([1., 2., 3.])
    v0 = [1., 0.2, 1.0]
    v1 = [1., -0.2, 1.0]
    v2 = [1., -0.2, -1.0]
    el = Ellipsoid([v0, v1, v2])


@pytest.mark.xfail(raises=ValueError)
def test_create_ellipsoid_fail():
    v0 = [1., 0.2, 1.0]
    v1 = [1., 0.2, 1.0]
    v2 = [1., -0.2, -1.0]
    el = Ellipsoid([v0, v1, v2])


def test_create_sphere():
    el = Sphere(1.)
    el = Sphere(1., center=[1.]*3)


def test_expand1():
    e1 = Ellipsoid([1., 1., 1.])
    e2 = e1.scale(1.1)
    assert np.allclose(e1.radius + 0.1, e2.radius)


def test_within1():
    o = Ellipsoid([1., 2., 3.])
    assert not o.within([-1.]*3)
    assert o.within([.2]*3)
    assert o.within([.5]*3)
    o = Ellipsoid([1., 1., 2.])
    assert not o.within([-1.]*3)
    assert o.within([.2]*3)
    assert o.within([.5]*3)
    o = Sphere(1.)
    assert not o.within([-1.]*3)
    assert o.within([.2]*3)
    assert o.within([.5]*3)


def test_within_index1():
    o = Ellipsoid([1., 2., 3.])
    assert not o.within_index([-1.]*3) == [0]
    assert o.within_index([.2]*3) == [0]
    assert o.within_index([.5]*3) == [0]
    o = Ellipsoid([1., 1., 2.])
    assert not o.within_index([-1.]*3) == [0]
    assert o.within_index([.2]*3) == [0]
    assert o.within_index([.5]*3) == [0]
    o = Sphere(1.)
    assert not o.within_index([-1.]*3) == [0]
    assert o.within_index([.2]*3) == [0]
    assert o.within_index([.5]*3) == [0]
