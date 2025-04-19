# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import sys

import numpy as np
import pytest

from sisl._help import (
    array_fill_repeat,
    array_replace,
    dtype_complex_to_float,
    get_dtype,
)

pytestmark = pytest.mark.help


def test_array_fill_repeat1():
    assert array_fill_repeat([1], 20).shape[0] == 20
    assert array_fill_repeat([1, 2], 20).shape[0] == 20
    assert array_fill_repeat(1, 20).shape[0] == 20
    assert array_fill_repeat([[1]], 20, axis=0).shape == (20, 1)
    assert array_fill_repeat([[1, 2]], 20, axis=0).shape == (20, 2)


def test_array_fill_repeat2():
    with pytest.raises(ValueError):
        array_fill_repeat([1, 2, 3], 20)


@pytest.mark.xfail(
    sys.platform.startswith("win"), reason="Datatype cannot be int64 on windows"
)
def test_get_dtype1():
    assert np.int32 == get_dtype(1)
    assert np.int64 == get_dtype(1, int=np.int64)


@pytest.mark.xfail(
    sys.platform.startswith("win"), reason="Datatype cannot be int64 on windows"
)
def test_dtype_complex_to_float():
    for d in (np.int32, np.int64, np.float32, np.float64):
        assert dtype_complex_to_float(d) == d
    assert dtype_complex_to_float(np.complex64) == np.float32
    assert dtype_complex_to_float(np.complex128) == np.float64


def test_array_replace():
    ar = np.arange(6)

    arnew = array_replace(ar, (1, 2))
    assert arnew[1] == 2
    arnew[1] -= 1
    assert np.all(arnew == ar)

    arnew = array_replace(ar, ([1, 3, 5], None), other=4)
    assert np.all(arnew[[1, 3, 5]] == [1, 3, 5])
    assert np.all(np.delete(arnew, [1, 3, 5]) == 4)

    arnew = array_replace(ar, ([1, 3], None), (5, None), other=4)
    assert np.all(arnew[[1, 3, 5]] == [1, 3, 5])
    assert np.all(np.delete(arnew, [1, 3, 5]) == 4)
