from __future__ import print_function, division

import pytest

import math as m
import numpy as np

from sisl._help import array_fill_repeat, get_dtype
from sisl._help import dtype_complex_to_real

pytestmark = pytest.mark.help


def test_array_fill_repeat1():
    assert array_fill_repeat([1], 20).shape[0] == 20
    assert array_fill_repeat([1, 2], 20).shape[0] == 20
    assert array_fill_repeat(1, 20).shape[0] == 20


@pytest.mark.xfail(raises=ValueError)
def test_array_fill_repeat2():
    array_fill_repeat([1, 2, 3], 20)


def test_get_dtype1():
    assert np.int32 == get_dtype(1)
    assert np.int64 == get_dtype(1, int=np.int64)


def test_dtype_complex_to_real():
    for d in [np.int32, np.int64, np.float32, np.float64]:
        assert dtype_complex_to_real(d) == d
    assert dtype_complex_to_real(np.complex64) == np.float32
    assert dtype_complex_to_real(np.complex128) == np.float64
