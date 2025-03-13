# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl._core.quaternion import Quaternion
from sisl.shape.ellipsoid import *
from sisl.utils.mathematics import fnorm

pytestmark = pytest.mark.shape


@pytest.mark.filterwarnings("ignore", message="*orthogonalizes the vectors")
def test_create_ellipsoid():
    el = Ellipsoid(1.0)
    el = Ellipsoid([1.0, 1.0, 1.0])
    el = Ellipsoid([1.0, 1.0, 1.0], [1.0] * 3)
    el = Ellipsoid([1.0, 2.0, 3.0])
    v0 = [1.0, 0.2, 1.0]
    v1 = [1.0, -0.2, 1.0]
    v2 = [1.0, -0.2, -1.0]
    el = Ellipsoid([v0, v1, v2])
    q = Quaternion(45, [1, 0, 0], rad=False)
    eye = np.identity(3)
    eye = q.rotate(eye)
    el = Ellipsoid(eye)
    e2 = el.scale(1.1)
    assert np.allclose(el.radius + 0.1, e2.radius)
    e2 = el.scale([1.1] * 3)
    assert np.allclose(el.radius + 0.1, e2.radius)
    e2 = el.scale([1.1, 2.1, 3.1])
    assert np.allclose(el.radius + [0.1, 1.1, 2.1], e2.radius)
    assert el.expand(2).volume == pytest.approx(4 / 3 * np.pi * 3**3)
    str(el)


def test_tosphere():
    el = Ellipsoid([1.0, 1.0, 1.0])
    assert el.to.Sphere().radius == pytest.approx(1)
    el = Ellipsoid([1.0, 2.0, 3.0])
    assert el.to.Sphere().radius == pytest.approx(3)


@pytest.mark.filterwarnings("ignore", message="*orthogonalizes the vectors")
def test_create_ellipsoid_fail():
    v0 = [1.0, 0.2, 1.0]
    v1 = [1.0, 0.2, 1.0]
    v2 = [1.0, -0.2, -1.0]
    with pytest.raises(ValueError):
        el = Ellipsoid([v0, v1, v2])


def test_create_ellipsoid_fail2():
    v0 = [1.0, 0.2, 1.0]
    v1 = [1.0, 0.2, 1.0]
    v2 = [1.0, -0.2, -1.0]
    v3 = [1.0, -0.2, -1.0]
    with pytest.raises(ValueError):
        el = Ellipsoid([v0, v1, v2, v3])


def test_create_sphere():
    el = Sphere(1.0)
    el = Sphere(1.0, center=[1.0] * 3)
    assert el.volume == pytest.approx(4 / 3 * np.pi)
    assert el.scale(2).volume == pytest.approx(4 / 3 * np.pi * 2**3)
    assert el.expand(2).volume == pytest.approx(4 / 3 * np.pi * 3**3)


def test_scale1():
    e1 = Ellipsoid([1.0, 1.0, 1.0])
    e2 = e1.scale(1.1)
    assert np.allclose(e1.radius + 0.1, e2.radius)
    e2 = e1.scale([1.1] * 3)
    assert np.allclose(e1.radius + 0.1, e2.radius)
    e2 = e1.scale([1.1, 2.1, 3.1])
    assert np.allclose(e1.radius + [0.1, 1.1, 2.1], e2.radius)


def test_expand1():
    e1 = Ellipsoid([1.0, 1.0, 1.0])
    e2 = e1.expand(1.1)
    assert np.allclose(e1.radius + 1.1, e2.radius)
    e2 = e1.expand([1.1] * 3)
    assert np.allclose(e1.radius + 1.1, e2.radius)
    e2 = e1.expand([1.1, 2.1, 3.1])
    assert np.allclose(e1.radius + [1.1, 2.1, 3.1], e2.radius)


def test_expand_fail():
    el = Ellipsoid(1)
    with pytest.raises(ValueError):
        el.expand([1, 2])


def test_within1():
    o = Ellipsoid([1.0, 2.0, 3.0])
    assert not o.within([-1.0] * 3).any()
    assert o.within([0.2] * 3)
    assert o.within([0.5] * 3)
    o = Ellipsoid([1.0, 1.0, 2.0])
    assert not o.within([-1.0] * 3).any()
    assert o.within([0.2] * 3)
    assert o.within([0.5] * 3)
    o = Sphere(1.0)
    assert not o.within([-1.0] * 3).any()
    assert o.within([0.2] * 3)
    assert o.within([0.5] * 3)


def test_within_index1():
    o = Ellipsoid([1.0, 2.0, 3.0])
    assert o.within_index([-1.0] * 3).size == 0
    assert o.within_index([0.2] * 3) == [0]
    assert o.within_index([0.5] * 3) == [0]
    o = Ellipsoid([1.0, 1.0, 2.0])
    assert o.within_index([-1.0] * 3).size == 0
    assert o.within_index([0.2] * 3) == [0]
    assert o.within_index([0.5] * 3) == [0]
    o = Sphere(1.0)
    assert o.within_index([-1.0] * 3).size == 0
    assert o.within_index([0.2] * 3) == [0]
    assert o.within_index([0.5] * 3) == [0]


def test_sphere_and():
    # For all intersections after
    v = np.array([1.0] * 3)
    v /= fnorm(v)
    D = np.linspace(0, 5, 50)
    inside = np.ones(len(D), dtype=bool)

    A = Sphere(2.0)
    is_first = True
    inside[:] = True
    for i, d in enumerate(D):
        B = Sphere(1.0, center=v * d)
        C = (A & B).to.Sphere()
        if is_first and C.radius < B.radius:
            is_first = False
            inside[i:] = False
        if inside[i]:
            assert C.radius == pytest.approx(B.radius)
        else:
            assert C.radius < B.radius

    A = Sphere(0.5)
    is_first = True
    inside[:] = True
    for i, d in enumerate(D):
        B = Sphere(1.0, center=v * d)
        C = (A & B).to.Sphere()
        str(A) + str(B) + str(C)
        if is_first and C.radius < A.radius:
            inside[i:] = False
            is_first = False
        if inside[i]:
            assert C.radius == pytest.approx(A.radius)
        else:
            assert C.radius < A.radius


def test_translate():
    el = Sphere(1.0, 1.0)
    el2 = el.translate([0, 1, 2])
    assert not np.allclose(el.center, el2.center)
