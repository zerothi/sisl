from __future__ import print_function, division

from nose.tools import *
from nose.plugins.attrib import attr

import math as m
import numpy as np

from sisl._help import array_fill_repeat, get_dtype


class TestHelp(object):

    def test_array_fill_repeat1(self):
        assert_equal(array_fill_repeat([1], 20).shape[0], 20)
        assert_equal(array_fill_repeat([1, 2], 20).shape[0], 20)
        assert_equal(array_fill_repeat(1, 20).shape[0], 20)

    @raises(ValueError)
    def test_array_fill_repeat2(self):
        array_fill_repeat([1, 2, 3], 20)

    def test_get_dtype1(self):
        assert_equal(np.int32, get_dtype(1))
        assert_equal(np.int64, get_dtype(1, int=np.int64))
