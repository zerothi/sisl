from __future__ import print_function, division

import pytest

import math as m
import numpy as np

from sisl.shape.prism4 import *

pytestmark = pytest.mark.shape


def test_create_cuboid():
    cube = Cuboid([1.0]*3)
    cube = Cuboid([1.0]*3, [1.]*3)
    cube = Cuboid([1.0, 2.0, 3.0], [1.]*3)
    v0 = [1., 0.2, 1.0]
    v1 = [1., -0.2, 1.0]
    v2 = [1., -0.2, -1.0]
    cube = Cuboid([v0, v1, v2])
    str(cube)


@pytest.mark.xfail(raises=ValueError)
def test_create_fail():
    v0 = [1., 0.2, 1.0]
    v1 = [1., 0.2, 1.0]
    v2 = [1., -0.2, -1.0]
    v3 = [1., -0.2, -1.0]
    el = Cuboid([v0, v1, v2, v3])


def test_tosphere():
    cube = Cube(1.)
    assert cube.toSphere().radius == pytest.approx(.5 * 3 ** 0.5)
    cube = Cube(3.)
    assert cube.toSphere().radius == pytest.approx(1.5 * 3 ** 0.5)
    cube = Cuboid([1., 2., 3.])
    assert cube.toSphere().radius == pytest.approx(1.5 * 3 ** 0.5)


def test_toellipsoid():
    cube = Cube(1.)
    assert cube.toEllipsoid().radius[0] == pytest.approx(.5 * 3 ** 0.5)
    cube = Cube(3.)
    assert cube.toEllipsoid().radius[0] == pytest.approx(1.5 * 3 ** 0.5)
    cube = Cuboid([1., 2., 3.])
    assert cube.toEllipsoid().radius[0] == pytest.approx(.5 * 3 ** 0.5)
    assert cube.toEllipsoid().radius[1] == pytest.approx(1 * 3 ** 0.5)
    assert cube.toEllipsoid().radius[2] == pytest.approx(1.5 * 3 ** 0.5)


def test_create_cube():
    cube = Cube(1.0)
    cube = Cube(1.0, [1.]*3)
    assert cube.volume() == pytest.approx(1.)
    assert cube.scale(2).volume() == pytest.approx(2 ** 3)
    assert cube.scale([2] * 3).volume() == pytest.approx(2 ** 3)
    assert cube.expand(2).volume() == pytest.approx(3 ** 3)
    assert cube.expand([2] * 3).volume() == pytest.approx(3 ** 3)


@pytest.mark.xfail(raises=ValueError)
def test_expand_fail():
    cube = Cube(1.0)
    cube.expand([2, 1])


def test_vol1():
    cube = Cuboid([1.0]*3)
    assert cube.volume() == 1.
    cube = Cuboid([1., 2., 3.])
    assert cube.volume() == 6.

    return
    a = (1./3) ** .5
    v0 = [a, a, 0]
    v1 = [-a, a, 0]
    v2 = [0, 0, a]
    cube = Cuboid([v0, v1, v2])
    assert cube.volume() == 1.


def test_origo():
    cube = Cuboid([1.0]*3)
    assert np.allclose(cube.origo, -0.5)
    cube.set_origo(1)
    assert np.allclose(cube.origo, 1)


def test_within1():
    cube = Cuboid([1.0]*3)
    assert not cube.within([-1.]*3)
    assert not cube.within([[-1.]*3, [-1., 0.5, 0.2]]).any()
    assert cube.within([[-1.]*3,
                        [-1., 0.5, 0.2],
                        [.1, 0.5, 0.2]]).any()


def test_within_index1():
    cube = Cuboid([1.0]*3)
    assert not cube.within_index([-1.]*3) == [0]
    assert not cube.within_index([[-1.]*3, [-1., 0.5, 0.2]]) == [0, 1]
    assert (cube.within_index([[-1.]*3,
                          [-1., 0.5, 0.2],
                          [.1, 0.5, 0.2]]) == [0, 1, 2]).any()
