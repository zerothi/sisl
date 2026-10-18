# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import math as m

import numpy as np
import pytest

from sisl.utils.misc import *
from sisl.utils.misc import size_to_elements, size_to_num

pytestmark = pytest.mark.utils


def test_direction_int():
    assert direction(0) == 0
    assert direction(1) == 1
    assert direction(2) == 2
    assert direction(2) != 1


def test_direction_str():
    assert direction("A") == 0
    assert direction("B") == 1
    assert direction("C") == 2
    assert direction("a") == 0
    assert direction("b") == 1
    assert direction("c") == 2
    assert direction("X") == 0
    assert direction("Y") == 1
    assert direction("Z") == 2
    assert direction("x") == 0
    assert direction("y") == 1
    assert direction("z") == 2
    assert direction("0") == 0
    assert direction("1") == 1
    assert direction("2") == 2
    assert direction(" 0") == 0
    assert direction(" 1  ") == 1
    assert direction("   2   ") == 2
    assert np.allclose(direction("   2   ", abc=np.diag([1, 2, 3])), [0, 0, 3])


def test_direction_int_raises():
    with pytest.raises(ValueError):
        direction(4)


def test_direction_str_raises():
    with pytest.raises(ValueError):
        direction("aosetuh")


def test_angle_r2r():
    assert pytest.approx(angle("2pi")) == 2 * m.pi
    assert pytest.approx(angle("2pi/2")) == m.pi
    assert pytest.approx(angle("3pi/4")) == 3 * m.pi / 4

    assert pytest.approx(angle("a2*180")) == 2 * m.pi
    assert pytest.approx(angle("2*180", in_rad=False)) == 2 * m.pi
    assert pytest.approx(angle("a2*180r")) == 2 * m.pi


def test_angle_a2a():
    assert pytest.approx(angle("a2pia")) == 360
    assert pytest.approx(angle("a2pi/2a")) == 180
    assert pytest.approx(angle("a3pi/4a")) == 3 * 180.0 / 4

    assert pytest.approx(angle("a2pia", True, True)) == 360
    assert pytest.approx(angle("a2pi/2a", True, False)) == 180
    assert pytest.approx(angle("a2pi/2a", False, True)) == 180
    assert pytest.approx(angle("a2pi/2a", False, False)) == 180


def test_iter_shape_all():
    shape = [2, 1, 3]
    expected = list(np.ndindex(*shape))
    assert list(iter_shape(shape)) == expected


def test_size_to_num_number():
    assert size_to_num(100, "B") == pytest.approx(100 * 1024**2)
    assert size_to_num(100.0, "kB") == pytest.approx(100 * 1024)
    assert size_to_num(100) == pytest.approx(100)
    assert size_to_num(1024, "GB") == pytest.approx(1)
    assert size_to_num(100, "B") == pytest.approx(size_to_num("100MB", "B"))


def test_size_to_elements_number():
    assert size_to_elements(100) == 100 * 1024**2 // 8
    assert size_to_elements(100) == size_to_elements("100MB")


def test_str_spec1():
    a = str_spec("foo")
    assert a[0] == "foo"
    assert a[1] is None
    a = str_spec("foo{bar}")
    assert a[0] == "foo"
    assert a[1] == "bar"


def test_listify():
    assert isinstance(listify([1, 2]), list)
    assert isinstance(1 | listify, list)
    a = np.ones(2)
    assert isinstance(listify(a), list)
    assert isinstance(a | listify, list)


def test_listify_as_index():
    idx = 1 | listify
    a = np.arange(2)
    assert a[idx] == 1


def test_merge_instances1():
    class A:
        pass

    a = A()
    a.hello = 1

    class B:
        pass

    b = B()
    b.hello = 2
    b.foo = 2

    class C:
        pass

    c = C()
    c.bar = 3
    d = merge_instances(a, b, c, name="TestClass")
    assert d.__class__.__name__ == "TestClass"
    assert d.hello == 2
    assert d.foo == 2
    assert d.bar == 3


def test_propertydict_property():
    a = PropertyDict()
    a.hello = "Hello"
    assert a["hello"] == "Hello"
    assert a["hello"] == a.hello
    del a.hello
    assert "hello" not in a


def test_propertydict_key():
    a = PropertyDict()
    a["hello"] = "Hello"
    assert a["hello"] == "Hello"
    assert a["hello"] == a.hello
    del a.hello
    assert "hello" not in a
