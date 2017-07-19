from __future__ import print_function, division

from nose.tools import *
from nose.plugins.attrib import attr

import math as m
import numpy as np

from sisl import Spin


@attr('spin')
class TestSpin(object):

    def test_spin1(self):
        for val in ['unpolarized', '',
                    'polarized', 'p',
                    'non-colinear', 'nc',
                    'spin-orbit', 'so']:
            s = Spin(val)
            repr(s)
            s1 = s.copy()
            assert_equal(s, s1)

    def test_spin2(self):
        s1 = Spin()
        s2 = Spin('p')
        s3 = Spin('nc')
        s4 = Spin('so')

        assert_true(s1 == s1.copy())
        assert_true(s2 == s2.copy())
        assert_true(s3 == s3.copy())
        assert_true(s4 == s4.copy())

        assert_true(s1 < s2)
        assert_true(s2 < s3)
        assert_true(s3 < s4)

        assert_true(s1 <= s2)
        assert_true(s2 <= s3)
        assert_true(s3 <= s4)

        assert_true(s2 > s1)
        assert_true(s3 > s2)
        assert_true(s4 > s3)

        assert_true(s2 >= s1)
        assert_true(s3 >= s2)
        assert_true(s4 >= s3)

        assert_true(s1.is_unpolarized)
        assert_false(s1.is_polarized)
        assert_false(s1.is_noncolinear)
        assert_false(s1.is_spinorbit)

        assert_false(s2.is_unpolarized)
        assert_true(s2.is_polarized)
        assert_false(s2.is_noncolinear)
        assert_false(s2.is_spinorbit)

        assert_false(s3.is_unpolarized)
        assert_false(s3.is_polarized)
        assert_true(s3.is_noncolinear)
        assert_false(s3.is_spinorbit)

        assert_false(s4.is_unpolarized)
        assert_false(s4.is_polarized)
        assert_false(s4.is_noncolinear)
        assert_true(s4.is_spinorbit)
