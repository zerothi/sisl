# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl.shape import *

pytestmark = pytest.mark.shape


def test_binary_op():
    e = Ellipsoid(1.0)
    s = Sphere(1.0)

    new = e + s
    str(new)

    new = new - new
    new = new & (new | e) ^ s
    new.center
    new2 = new.translate([0, 1, 2])
    assert not np.allclose(new.center, new2.center)

    assert new.volume < 0.0
    str(new)


def test_null():
    null = NullShape()
    assert null.volume == 0.0
    assert len(null.within_index(np.random.rand(1000, 3))) == 0

    assert null.to.Ellipsoid().volume < 1e-64
    assert null.to.Cuboid().volume < 1e-64
    assert null.to.Sphere().volume < 1e-64


def test_binary_op_within():
    e = Ellipsoid(0.5)
    c = Cube(1.0)

    xc = [0.499] * 3
    new = e + c
    assert not e.within(xc)
    assert c.within(xc)
    assert new.within(xc)

    xe = [0.0] * 3
    new = e - c
    assert e.within(xe)
    assert c.within(xe)
    assert not new.within(xe)

    new = e & c
    assert not e.within(xc)
    assert c.within(xc)
    assert not new.within(xc)
    assert e.within(xe)
    assert c.within(xe)
    assert new.within(xe)

    # e ^ c == c ^ e
    new = e ^ c
    assert not e.within(xc)
    assert c.within(xc)
    assert new.within(xc)
    assert e.within(xe)
    assert c.within(xe)
    assert not new.within(xe)

    new = c ^ e
    assert new.within(xc)
    assert not new.within(xe)


def test_binary_op_toSphere():
    e = Ellipsoid(0.5)
    c = Cube(1.0)

    r = 0.5 * 3**0.5
    new = e + c
    assert new.to.Sphere().radius.max() == pytest.approx(r)

    new = e - c
    assert new.to.Sphere().radius.max() == pytest.approx(r)

    # with the AND operator we can reduce to smallest enclosed sphere
    new = e & c
    assert new.to.Sphere().radius.max() == pytest.approx(0.5)

    # e ^ c == c ^ e
    new = e ^ c
    assert new.to.Sphere().radius.max() == pytest.approx(r)

    new = c ^ e
    assert new.to.Sphere().radius.max() == pytest.approx(r)

    new = c ^ e
    assert new.scale(2).to.Sphere().radius.max() == pytest.approx(r * 2)


def test_toSphere_and():
    left = Sphere(1.0)
    right = Sphere(1.0, center=[0.6] * 3)

    new = left & right
    s = new.to.Sphere()
    assert s.radius.max() < 0.9

    left = Sphere(2.0)
    right = Sphere(1.0, center=[0.5] * 3)

    new = left & right
    s = new.to.Sphere()
    assert s.radius.max() == pytest.approx(1.0)

    left = Sphere(2.0, center=[10, 10, 10])
    right = Sphere(1.0, center=[10.5] * 3)

    new = left & right
    s2 = new.to.Sphere()
    assert s2.radius.max() == pytest.approx(1.0)
    # Assert it also works for displaced centers
    assert np.allclose(s.radius, s2.radius)
    assert np.allclose(s.center, s2.center - 10)

    left = Sphere(2.0)
    right = Sphere(1.0, center=[10.5] * 3)

    new = left & right
    s = new.to.Sphere()
    assert s.radius.max() < 0.01


def test_toEllipsoid_and():
    left = Ellipsoid(1.0)
    right = Ellipsoid(1.0, center=[0.6] * 3)

    new = left & right
    s = new.to.Ellipsoid()
    assert s.radius.max() < 0.9

    left = Ellipsoid(2.0)
    right = Ellipsoid(1.0, center=[0.5] * 3)

    new = left & right
    s = new.to.Ellipsoid()
    assert s.radius.max() == pytest.approx(1.0)

    left = Ellipsoid(2.0, center=[10, 10, 10])
    right = Ellipsoid(1.0, center=[10.5] * 3)

    new = left & right
    s2 = new.to.Ellipsoid()
    assert s2.radius.max() == pytest.approx(1.0)
    # Assert it also works for displaced centers
    assert np.allclose(s.radius, s2.radius)
    assert np.allclose(s.center, s2.center - 10)

    left = Ellipsoid(2.0)
    right = Ellipsoid(1.0, center=[10.5] * 3)

    new = left & right
    s = new.to.Ellipsoid()
    assert s.radius.max() < 0.01


def test_toCuboid_and():
    left = Cuboid(1.0)
    right = Cuboid(1.0, center=[0.6] * 3)

    new = left & right
    s = new.to.Cuboid()
    assert s.edge_length.max() < 0.9 * 2

    left = Cuboid(2.0)
    right = Cuboid(1.0, center=[0.5] * 3)

    new = left & right
    s = new.to.Cuboid()
    assert s.edge_length.max() >= 1.5

    left = Cuboid(2.0, center=[10, 10, 10])
    right = Cuboid(1.0, center=[10.5] * 3)

    new = left & right
    s2 = new.to.Cuboid()
    assert s2.edge_length.max() > 1.5
    # Assert it also works for displaced centers
    assert np.allclose(s.edge_length, s2.edge_length)
    assert np.allclose(s.center, s2.center - 10)

    left = Cuboid(2.0)
    right = Cuboid(1.0, center=[10.5] * 3)

    new = left & right
    s = new.to.Cuboid()
    assert s.edge_length.max() < 0.01
