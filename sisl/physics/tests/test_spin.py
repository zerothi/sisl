from __future__ import print_function, division

import pytest

pytestmark = pytest.mark.spin

import math as m
import numpy as np

from sisl import Spin


def test_spin1():
    for val in ['unpolarized', '', Spin.UNPOLARIZED,
                'polarized', 'p', Spin.POLARIZED,
                'non-collinear', 'nc', Spin.NONCOLINEAR,
                'spin-orbit', 'so', Spin.SPINORBIT]:
        s = Spin(val)
        str(s)
        s1 = s.copy()
        assert s == s1


def test_spin2():
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
def test_spin3():
    s = Spin('satoehus')


def test_spin4():
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


def test_pauli():
    # just grab the default spin
    S = Spin()

    # Create a fictituous wave-function
    sq2 = 2 ** .5
    W = np.array([
        [1/sq2, 1/sq2], # M_x = 1
        [1/sq2, -1/sq2], # M_x = -1
        [0.5 + 0.5j, 0.5 + 0.5j], # M_x = 1
        [0.5 - 0.5j, -0.5 + 0.5j], # M_x = -1
        [1/sq2, 1j/sq2], # M_y = 1
        [1/sq2, -1j/sq2], # M_y = -1
        [0.5 - 0.5j, 0.5 + 0.5j], # M_y = 1
        [0.5 + 0.5j, 0.5 - 0.5j], # M_y = -1
        [1, 0], # M_z = 1
        [0, 1], # M_z = -1
    ])
    x = np.array([1, -1, 1, -1, 0, 0, 0, 0, 0, 0])
    assert np.allclose(x, (np.conj(W)*S.X.dot(W.T).T).sum(1).real)
    y = np.array([0, 0, 0, 0, 1, -1, 1, -1, 0, 0])
    assert np.allclose(y, (np.conj(W)*np.dot(S.Y, W.T).T).sum(1).real)
    z = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, -1])
    assert np.allclose(z, (np.conj(W)*np.dot(S.Z, W.T).T).sum(1).real)


def test_pickle():
    import pickle as p

    S = Spin('nc')
    n = p.dumps(S)
    s = p.loads(n)
    assert S == s
