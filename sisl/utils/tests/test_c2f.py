from __future__ import print_function, division

from nose.tools import *

from sisl.utils.c2f import *

import math as m
import numpy as np


class TestC2F(object):

    def test_index_int(self):
        assert_true(index_c2f(1) == 0)
        assert_true(index_c2f(20) == 19)

    def test_index_tuple(self):
        assert_true(index_c2f((1, 0, 2)) == (0, -1, 1))

    def test_index_list(self):
        assert_true(index_c2f([1, 0, 2]) == [0, -1, 1])

    def test_index_tuple_list(self):
        assert_true(index_c2f(([1, 0, 2], [4])) == ([0, -1, 1], [3]))

    def test_index_list_tuple(self):
        assert_true(index_c2f([(1, 0, 2), (4,)]) == [(0, -1, 1), (3,)])

    def test_index_array(self):
        a1 = np.array([1, 0, 2])
        b1 = a1 - 1
        assert_true(np.array_equal(index_c2f(a1), b1))

    def test_index_tuple_array(self):
        a1 = np.array([1, 0, 2])
        a2 = a1 + 3
        b1 = a1 - 1
        b2 = a2 - 1
        for a, b in zip(index_c2f((a1, a2)), (b1, b2)):
            assert_true(np.array_equal(a, b))

    def test_index_list_array(self):
        a1 = np.array([1, 0, 2])
        a2 = a1 + 3
        b1 = a1 - 1
        b2 = a2 - 1
        for a, b in zip(index_c2f([a1, a2]), [b1, b2]):
            assert_true(np.array_equal(a, b))
