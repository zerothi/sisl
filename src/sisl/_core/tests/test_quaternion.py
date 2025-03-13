# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import pytest

from sisl import Quaternion


@pytest.fixture
def setup():
    class t:
        def __init__(self):
            self.qx = Quaternion(90, [1, 0, 0], rad=False)
            self.qy = Quaternion(90, [0, 1, 0], rad=False)
            self.qz = Quaternion(90, [0, 0, 1], rad=False)
            self.Qx = Quaternion(90, [2, 0, 0], rad=False)
            self.Qy = Quaternion(90, [0, 2, 0], rad=False)
            self.Qz = Quaternion(90, [0, 0, 2], rad=False)

    return t()


@pytest.mark.quaternion
class TestQuaternion:
    def test_copy(self, setup):
        qx = setup.qx.copy()
        assert qx == setup.qx
        assert qx._v is not setup.qx._v.base

    def test_init(self):
        a = Quaternion(90, [1, 0, 1])
        b = Quaternion([1, 0, 1], 90)
        assert a == b
        c = Quaternion([1, 0, 2, 1])
        assert a != c

    def test_str_repr(self):
        a = Quaternion(90, [1, 0, 1])
        assert str(a) != repr(a)

    def test_conj(self, setup):
        qx = setup.qx.conj()
        assert qx.conj() == setup.qx

    def test_norm(self, setup):
        for c in "xyz":
            assert getattr(setup, "q" + c).norm() == pytest.approx(1.0)

    def test_degree1(self, setup):
        for c in "xyz":
            assert getattr(setup, "q" + c).angle(in_rad=False) == pytest.approx(90)

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
        rx = setup.qx + 1.0
        assert rx - 1.0 == setup.qx

        rx = setup.qx * 1.0
        assert rx == setup.qx

        rx = setup.qx * 1.0
        assert rx == setup.qx

        rx = setup.qx / 1.0
        assert rx == setup.qx

        rx = setup.qx.copy()
        rx += 1.0
        rx -= 1.0
        assert rx == setup.qx

        rx = setup.qx.copy()
        rx *= setup.qy
        assert rx == setup.qx * setup.qy

    def test_fail_div1(self, setup):
        with pytest.raises(NotImplementedError):
            setup.qx /= setup.qy

    def test_fail_div2(self, setup):
        with pytest.raises(NotImplementedError):
            a = setup.qx / setup.qy
