# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl.viz.processors.axes import (
    axes_cross_product,
    axis_direction,
    get_ax_title,
    sanitize_axes,
)

pytestmark = [pytest.mark.viz, pytest.mark.processors]


def test_sanitize_axes():
    assert sanitize_axes(["x", "y", "z"]) == ["x", "y", "z"]
    assert sanitize_axes("xyz") == ["x", "y", "z"]
    assert sanitize_axes("abc") == ["a", "b", "c"]
    assert sanitize_axes([0, 1, 2]) == ["a", "b", "c"]
    assert sanitize_axes("-xy") == ["-x", "y"]
    assert sanitize_axes("x-y") == ["x", "-y"]
    assert sanitize_axes("-x-y") == ["-x", "-y"]
    assert sanitize_axes("a-b") == ["a", "-b"]

    axes = sanitize_axes([[0, 1, 2]])
    assert isinstance(axes[0], np.ndarray)
    assert axes[0].shape == (3,)
    assert np.all(axes[0] == [0, 1, 2])

    with pytest.raises(ValueError):
        sanitize_axes([None])


def test_axis_direction():
    assert np.allclose(axis_direction("x"), [1, 0, 0])
    assert np.allclose(axis_direction("y"), [0, 1, 0])
    assert np.allclose(axis_direction("z"), [0, 0, 1])

    assert np.allclose(axis_direction("-x"), [-1, 0, 0])
    assert np.allclose(axis_direction("-y"), [0, -1, 0])
    assert np.allclose(axis_direction("-z"), [0, 0, -1])

    assert np.allclose(axis_direction([1, 0, 0]), [1, 0, 0])

    cell = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    assert np.allclose(axis_direction("a", cell), [0, 0, 1])
    assert np.allclose(axis_direction("b", cell), [1, 0, 0])
    assert np.allclose(axis_direction("c", cell), [0, 1, 0])

    assert np.allclose(axis_direction("-a", cell), [0, 0, -1])
    assert np.allclose(axis_direction("-b", cell), [-1, 0, 0])
    assert np.allclose(axis_direction("-c", cell), [0, -1, 0])


def test_axes_cross_product():
    assert np.allclose(axes_cross_product("x", "y"), [0, 0, 1])
    assert np.allclose(axes_cross_product("y", "x"), [0, 0, -1])
    assert np.allclose(axes_cross_product("-x", "y"), [0, 0, -1])

    assert np.allclose(axes_cross_product([1, 0, 0], [0, 1, 0]), [0, 0, 1])
    assert np.allclose(axes_cross_product([0, 1, 0], [1, 0, 0]), [0, 0, -1])

    cell = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    assert np.allclose(axes_cross_product("b", "c", cell), [0, 0, 1])
    assert np.allclose(axes_cross_product("c", "b", cell), [0, 0, -1])
    assert np.allclose(axes_cross_product("-b", "c", cell), [0, 0, -1])


def test_axis_title():
    assert get_ax_title("title") == "title"

    assert get_ax_title("x") == "X axis [Ang]"
    assert get_ax_title("y") == "Y axis [Ang]"
    assert get_ax_title("-z") == "-Z axis [Ang]"

    assert get_ax_title("a") == "A lattice vector"
    assert get_ax_title("b") == "B lattice vector"
    assert get_ax_title("-c") == "-C lattice vector"

    assert get_ax_title(None) == ""

    assert get_ax_title(np.array([1, 2, 3])) == "[1 2 3]"

    def some_axis():
        pass

    assert get_ax_title(some_axis) == "some_axis"
