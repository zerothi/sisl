from __future__ import print_function, division

import pytest

import math as m
import numpy as np

from sisl import Spin


@pytest.mark.spin
class TestSpin(object):

    def test_spin1(self):
        for val in ['unpolarized', '', Spin.UNPOLARIZED,
                    'polarized', 'p', Spin.POLARIZED,
                    'non-colinear', 'nc', Spin.NONCOLINEAR,
                    'spin-orbit', 'so', Spin.SPINORBIT]:
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
        s1 = Spin(Spin.UNPOLARIZED)
        S1 = Spin(Spin.UNPOLARIZED, np.complex64)
        s2 = Spin(Spin.POLARIZED)
        S2 = Spin(Spin.POLARIZED, np.complex64)
        s3 = Spin(Spin.NONCOLINEAR)
        S3 = Spin(Spin.NONCOLINEAR, np.complex64)
        s4 = Spin(Spin.SPINORBIT)
        S4 = Spin(Spin.SPINORBIT, np.complex64)
        assert s1 == S1
        assert s2 == S2
        assert s3 == S3
        assert s4 == S4

        # real comparison
        assert s1 < S2
        assert s1 < S3
        assert s1 < S4

        assert s2 > S1
        assert s2 < S3
        assert s2 < S4

        assert s3 > S1
        assert s3 > S2
        assert s3 < S4

        assert s4 > S1
        assert s4 > S2
        assert s4 > S3

        # complex complex
        assert S1 < S2
        assert S1 < S3
        assert S1 < S4

        assert S2 > S1
        assert S2 < S3
        assert S2 < S4

        assert S3 > S1
        assert S3 > S2
        assert S3 < S4

        assert S4 > S1
        assert S4 > S2
        assert S4 > S3

        # real comparison
        assert S1 < s2
        assert S1 < s3
        assert S1 < s4

        assert S2 > s1
        assert S2 < s3
        assert S2 < s4

        assert S3 > s1
        assert S3 > s2
        assert S3 < s4

        assert S4 > s1
        assert S4 > s2
        assert S4 > s3

        # complex complex
        assert S1 < s2
        assert S1 < s3
        assert S1 < s4

        assert S2 > s1
        assert S2 < s3
        assert S2 < s4

        assert S3 > s1
        assert S3 > s2
        assert S3 < s4

        assert S4 > s1
        assert S4 > s2
        assert S4 > s3
