from __future__ import print_function, division

import pytest

import math as m
import numpy as np

from sisl import Quaternion


@pytest.fixture
def setup():
    class t():
        def __init__(self):
            self.qx = Quaternion(90, [1, 0, 0])
            self.qy = Quaternion(90, [0, 1, 0])
            self.qz = Quaternion(90, [0, 0, 1])
            self.Qx = Quaternion(90, [2, 0, 0])
            self.Qy = Quaternion(90, [0, 2, 0])
            self.Qz = Quaternion(90, [0, 0, 2])
    return t()


@pytest.mark.quaternion
class TestQuaternion(object):

    def test_copy(self, setup):
        qx = setup.qx.copy()
        assert qx == setup.qx

    def test_conj(self, setup):
        qx = setup.qx.conj()
        assert qx.conj() == setup.qx

    def test_norm(self, setup):
        for c in 'xyz':
            assert getattr(setup, 'q'+c).norm() == 1.

    def test_degree1(self, setup):
        for c in 'xyz':
            assert getattr(setup, 'q'+c).degree == 90

    def test_radians1(self, setup):
        rx = setup.qx.radian
        ry = setup.qy.radian
        assert rx == ry

    def test_op1(self, setup):
        rx = -setup.qx
        assert -rx == setup.qx

        rxy = setup.qx + setup.qy
        setup.qx += setup.qy
        assert rxy == setup.qx
        setup.qx -= setup.qy
        assert -rx == setup.qx

        rxy = setup.qx - setup.qy
        setup.qx -= setup.qy
        assert rxy == setup.qx
        setup.qx += setup.qy
        assert -rx == setup.qx

    def test_op2(self, setup):
        rx = setup.qx + 1.
        assert rx - 1. == setup.qx

        rx = setup.qx * 1.
        assert rx == setup.qx

        rx = setup.qx * 1.
        assert rx == setup.qx

        rx = setup.qx / 1.
        assert rx == setup.qx

        rx = setup.qx.copy()
        rx += 1.
        rx -= 1.
        assert rx == setup.qx

        rx = setup.qx.copy()
        rx *= setup.qy
        assert rx == setup.qx * setup.qy

    @pytest.mark.xfail(raises=ValueError)
    def test_fail_div1(self, setup):
        setup.qx /= setup.qy

    @pytest.mark.xfail(raises=ValueError)
    def test_fail_div2(self, setup):
        a = setup.qx / setup.qy
