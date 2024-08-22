# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl.shape import *

pytestmark = pytest.mark.shape


def test_create_cuboid():
    cube = Cuboid([1.0] * 3)
    cube = Cuboid([1.0] * 3, [1.0] * 3)
    cube = Cuboid([1.0, 2.0, 3.0], [1.0] * 3)
    cube = Cuboid([1.0, 2.0, 3.0], origin=[1.0] * 3)
    v0 = [1.0, 0.2, 1.0]
    v1 = [1.0, -0.2, 1.0]
    v2 = [1.0, -0.2, -1.0]
    cube = Cuboid([v0, v1, v2])
    str(cube)


def test_create_fail():
    v0 = [1.0, 0.2, 1.0]
    v1 = [1.0, 0.2, 1.0]
    v2 = [1.0, -0.2, -1.0]
    v3 = [1.0, -0.2, -1.0]
    with pytest.raises(ValueError):
        el = Cuboid([v0, v1, v2, v3])
    with pytest.raises(ValueError):
        el = Cuboid(2, center=v1, origin=v2)


def test_tosphere():
    cube = Cube(1.0)
    assert cube.to.Sphere().radius == pytest.approx(0.5 * 3**0.5)
    cube = Cube(3.0)
    assert cube.to.Sphere().radius == pytest.approx(1.5 * 3**0.5)
    cube = Cuboid([1.0, 2.0, 3.0])
    assert cube.to.Sphere().radius == pytest.approx(1.5 * 3**0.5)
    assert cube.to.Sphere().radius == pytest.approx(1.5 * 3**0.5)
    assert isinstance(cube.to.Sphere(), Sphere)


def test_toellipsoid():
    cube = Cube(1.0)
    assert cube.to.Ellipsoid().radius[0] == pytest.approx(0.5 * 3**0.5)
    cube = Cube(3.0)
    assert cube.to.Ellipsoid().radius[0] == pytest.approx(1.5 * 3**0.5)
    cube = Cuboid([1.0, 2.0, 3.0])
    assert cube.to.Ellipsoid().radius[0] == pytest.approx(0.5 * 3**0.5)
    assert cube.to.Ellipsoid().radius[1] == pytest.approx(1 * 3**0.5)
    assert cube.to.Ellipsoid().radius[2] == pytest.approx(1.5 * 3**0.5)


def test_create_cube():
    cube = Cube(1.0)
    cube = Cube(1.0, [1.0] * 3)
    assert cube.volume == pytest.approx(1.0)
    assert cube.scale(2).volume == pytest.approx(2**3)
    assert cube.scale([2] * 3).volume == pytest.approx(2**3)
    assert cube.expand(2).volume == pytest.approx(3**3)
    assert cube.expand([2] * 3).volume == pytest.approx(3**3)


def test_expand_fail():
    cube = Cube(1.0)
    with pytest.raises(ValueError):
        cube.expand([2, 1])


def test_vol1():
    cube = Cuboid([1.0] * 3)
    assert cube.volume == 1.0
    cube = Cuboid([1.0, 2.0, 3.0])
    assert cube.volume == 6.0

    return
    a = (1.0 / 3) ** 0.5
    v0 = [a, a, 0]
    v1 = [-a, a, 0]
    v2 = [0, 0, a]
    cube = Cuboid([v0, v1, v2])
    assert cube.volume == 1.0


def test_origin():
    cube = Cuboid([1.0] * 3)
    assert np.allclose(cube.origin, -0.5)
    cube.origin = 1
    assert np.allclose(cube.origin, 1)


def test_within1():
    cube = Cuboid([1.0] * 3)
    assert not cube.within([-1.0] * 3).any()
    assert not cube.within([[-1.0] * 3, [-1.0, 0.5, 0.2]]).any()
    assert cube.within([[-1.0] * 3, [-1.0, 0.5, 0.2], [0.1, 0.5, 0.2]]).any()


def test_within_index1():
    cube = Cuboid([1.0] * 3)
    assert cube.within_index([-1.0] * 3).size == 0
    assert cube.within_index([[-1.0] * 3, [-1.0, 0.5, 0.2]]).size == 0
    assert (
        cube.within_index([[-1.0] * 3, [-1.0, 0.5, 0.2], [0.1, 0.5, 0.2]]) == [0, 1, 2]
    ).any()


def test_translate():
    el = Cuboid(1.0, 1.0)
    el2 = el.translate([0, 1, 2])
    assert not np.allclose(el.center, el2.center)
