from __future__ import print_function, division

import pytest

import math as m
import numpy as np

from sisl import Spin


@pytest.mark.spin
class TestSpin(object):

    def test_spin1(self):
        for val in ['unpolarized', '',
                    'polarized', 'p',
                    'non-colinear', 'nc',
                    'spin-orbit', 'so']:
            s = Spin(val)
            repr(s)
            s1 = s.copy()
            assert s == s1

    def test_spin2(self):
        s1 = Spin()
        s2 = Spin('p')
        s3 = Spin('nc')
        s4 = Spin('so')

        assert s1.kind == Spin.UNPOLARIZED
        assert s2.kind == Spin.POLARIZED
        assert s3.kind == Spin.NONCOLINEAR
        assert s4.kind == Spin.SPINORBIT

        assert s1 == s1.copy()
        assert s2 == s2.copy()
        assert s3 == s3.copy()
        assert s4 == s4.copy()

        assert s1 < s2
        assert s2 < s3
        assert s3 < s4

        assert s1 <= s2
        assert s2 <= s3
        assert s3 <= s4

        assert s2 > s1
        assert s3 > s2
        assert s4 > s3

        assert s2 >= s1
        assert s3 >= s2
        assert s4 >= s3

        assert s1.is_unpolarized
        assert not s1.is_polarized
        assert not s1.is_noncolinear
        assert not s1.is_spinorbit

        assert not s2.is_unpolarized
        assert s2.is_polarized
        assert not s2.is_noncolinear
        assert not s2.is_spinorbit

        assert not s3.is_unpolarized
        assert not s3.is_polarized
        assert s3.is_noncolinear
        assert not s3.is_spinorbit

        assert not s4.is_unpolarized
        assert not s4.is_polarized
        assert not s4.is_noncolinear
        assert s4.is_spinorbit

    @pytest.mark.xfail(raises=ValueError)
    def test_spin3(self):
        s = Spin('satoehus')

    def test_spin4(self):
        s1 = Spin(1)
        S1 = Spin(1, np.complex64)
        s2 = Spin(2)
        S2 = Spin(2, np.complex64)
        s3 = Spin(3)
        S3 = Spin(3, np.complex64)
        s4 = Spin(4)
        S4 = Spin(4, np.complex64)
        assert sr == sc
