from __future__ import print_function, division

import pytest

import math as m
import numpy as np

from sisl.shape import *

pytestmark = pytest.mark.shape


def test_binary_op():
    e = Ellipsoid(1.)
    s = Sphere(1.)

    new = e + s
    print(new)

    new = new - new
    new = new & (new | e) ^ s
    print(new)


def test_binary_op_within():
    e = Ellipsoid(.5)
    c = Cube(1.)

    xc = [0.499] * 3
    new = e + c
    assert not e.within(xc)
    assert c.within(xc)
    assert new.within(xc)

    xe = [0.] * 3
    new = e - c
    assert e.within(xe)
    assert c.within(xe)
    assert not new.within(xe)

    new = (e & c)
    assert not e.within(xc)
    assert c.within(xc)
    assert not new.within(xc)
    assert e.within(xe)
    assert c.within(xe)
    assert new.within(xe)

    # e ^ c == c ^ e
    new = (e ^ c)
    assert not e.within(xc)
    assert c.within(xc)
    assert new.within(xc)
    assert e.within(xe)
    assert c.within(xe)
    assert not new.within(xe)

    new = (c ^ e)
    assert new.within(xc)
    assert not new.within(xe)


def test_binary_op_toSphere():
    e = Ellipsoid(.5)
    c = Cube(1.)

    r = 2 ** .5
    new = e + c
    assert new.toSphere().radius.max() == pytest.approx(r)

    new = e - c
    assert new.toSphere().radius.max() == pytest.approx(r)

    # with the AND operator we can reduce to smallest enclosed sphere
    new = (e & c)
    assert new.toSphere().radius.max() == pytest.approx(0.5)

    # e ^ c == c ^ e
    new = (e ^ c)
    assert new.toSphere().radius.max() == pytest.approx(r)

    new = (c ^ e)
    assert new.toSphere().radius.max() == pytest.approx(r)


def test_toSphere_and():
    left = Sphere(1.)
    right = Sphere(1., center=[0.6] * 3)

    new = left & right
    s = new.toSphere()
    assert s.radius.max() < .9

    left = Sphere(2.)
    right = Sphere(1., center=[0.5] * 3)

    new = left & right
    s = new.toSphere()
    assert s.radius.max() == pytest.approx(1.)

    left = Sphere(2., center=[10, 10, 10])
    right = Sphere(1., center=[10.5] * 3)

    new = left & right
    s2 = new.toSphere()
    assert s2.radius.max() == pytest.approx(1.)
    # Assert it also works for displaced centers
    assert np.allclose(s.radius, s2.radius)
    assert np.allclose(s.center, s2.center - 10)
