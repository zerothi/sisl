from __future__ import print_function, division

import pytest

import math as m
import numpy as np

from sisl._help import array_fill_repeat, get_dtype


class TestHelp(object):

    def test_array_fill_repeat1(self):
        assert array_fill_repeat([1], 20).shape[0] == 20
        assert array_fill_repeat([1, 2], 20).shape[0] == 20
        assert array_fill_repeat(1, 20).shape[0] == 20

    @pytest.mark.xfail(raises=ValueError)
    def test_array_fill_repeat2(self):
        array_fill_repeat([1, 2, 3], 20)

    def test_get_dtype1(self):
        assert np.int32 == get_dtype(1)
        assert np.int64 == get_dtype(1, int=np.int64)
