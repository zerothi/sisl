from __future__ import print_function, division

from nose.tools import *
from nose.plugins.attrib import attr

import math as m
import numpy as np

from sisl import Quaternion


@attr('quaternion')
class TestQuaternion(object):

    def setUp(self):
        self.qx = Quaternion(90, [1, 0, 0])
        self.qy = Quaternion(90, [0, 1, 0])
        self.qz = Quaternion(90, [0, 0, 1])
        self.Qx = Quaternion(90, [2, 0, 0])
        self.Qy = Quaternion(90, [0, 2, 0])
        self.Qz = Quaternion(90, [0, 0, 2])

    def tearDown(self):
        del self.qx
        del self.qy
        del self.qz
        del self.Qx
        del self.Qy
        del self.Qz

    def test_copy(self):
        qx = self.qx.copy()
        assert_equal(qx, self.qx)

    def test_conj(self):
        qx = self.qx.conj()
        assert_equal(qx.conj(), self.qx)

    def test_norm(self):
        for c in 'xyz':
            assert_equal(getattr(self, 'q'+c).norm(), 1.)

    def test_degree1(self):
        for c in 'xyz':
            assert_equal(getattr(self, 'q'+c).degree, 90)

    def test_radians1(self):
        rx = self.qx.radians
        ry = self.qy.radians
        assert_equal(rx, ry)

    def test_op1(self):
        rx = -self.qx
        assert_equal(-rx, self.qx)

        rxy = self.qx + self.qy
        self.qx += self.qy
        assert_equal(rxy, self.qx)
        self.qx -= self.qy
        assert_equal(-rx, self.qx)

        rxy = self.qx - self.qy
        self.qx -= self.qy
        assert_equal(rxy, self.qx)
        self.qx += self.qy
        assert_equal(-rx, self.qx)
