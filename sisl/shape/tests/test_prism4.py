from __future__ import print_function, division

from nose.tools import *
from nose.plugins.attrib import attr

import math as m
import numpy as np

from sisl.shape.prism4 import *


@attr('shape')
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
        assert_true(cube.volume == 1.)
        cube = Cuboid([1., 2., 3.])
        assert_true(cube.volume == 6.)

    def test_within1(self):
        cube = Cuboid([1.0]*3)
        assert_false(cube.within([-1.]*3))
        assert_false(cube.within([[-1.]*3, [-1., 0.5, 0.2]]).any())
        assert_true(cube.within([[-1.]*3,
                                 [-1., 0.5, 0.2],
                                 [.1, 0.5, 0.2]]).any())

    def test_iwithin1(self):
        cube = Cuboid([1.0]*3)
        assert_false(cube.iwithin([-1.]*3) == [0])
        assert_false(cube.iwithin([[-1.]*3, [-1., 0.5, 0.2]]) == [0, 1])
        assert_true((cube.iwithin([[-1.]*3,
                                   [-1., 0.5, 0.2],
                                   [.1, 0.5, 0.2]]) == [0, 1, 2]).any())
