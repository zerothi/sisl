from __future__ import print_function, division

import pytest

import math as m
import numpy as np

from sisl.shape.prism4 import *


@pytest.mark.shape
class TestPrism4(object):

    def test_create_cuboid(self):
        cube = Cuboid([1.0]*3)
        cube = Cuboid([1.0]*3, [1.]*3)
        cube = Cuboid([1.0, 2.0, 3.0], [1.]*3)

    def test_create_cube(self):
        cube = Cube(1.0)
        cube = Cube(1.0, [1.]*3)

    def test_vol1(self):
        cube = Cuboid([1.0]*3)
        assert cube.volume == 1.
        cube = Cuboid([1., 2., 3.])
        assert cube.volume == 6.

    def test_within1(self):
        cube = Cuboid([1.0]*3)
        assert not cube.within([-1.]*3)
        assert not cube.within([[-1.]*3, [-1., 0.5, 0.2]]).any()
        assert cube.within([[-1.]*3,
                            [-1., 0.5, 0.2],
                            [.1, 0.5, 0.2]]).any()

    def test_iwithin1(self):
        cube = Cuboid([1.0]*3)
        assert not cube.iwithin([-1.]*3) == [0]
        assert not cube.iwithin([[-1.]*3, [-1., 0.5, 0.2]]) == [0, 1]
        assert (cube.iwithin([[-1.]*3,
                              [-1., 0.5, 0.2],
                              [.1, 0.5, 0.2]]) == [0, 1, 2]).any()
