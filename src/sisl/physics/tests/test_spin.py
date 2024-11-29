# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl import Spin

pytestmark = [pytest.mark.physics, pytest.mark.spin]


def test_spin_init():
    for val in [
        "unpolarized",
        "",
        Spin.UNPOLARIZED,
        "polarized",
        "p",
        Spin.POLARIZED,
        "non-collinear",
        "nc",
        Spin.NONCOLINEAR,
        "spin-orbit",
        "so",
        Spin.SPINORBIT,
        "nambu",
        "bdg",
        Spin.NAMBU,
    ]:
        s = Spin(val)
        str(s)
        s1 = s.copy()
        assert s == s1


def test_spin_comparisons():
    s1 = Spin()
    s2 = Spin("p")
    s3 = Spin("nc")
    s4 = Spin("so")
    s5 = Spin("nambu")

    assert s1.kind == Spin.UNPOLARIZED
    assert s2.kind == Spin.POLARIZED
    assert s3.kind == Spin.NONCOLINEAR
    assert s4.kind == Spin.SPINORBIT
    assert s5.kind == Spin.NAMBU

    assert s1 == s1.copy()
    assert s2 == s2.copy()
    assert s3 == s3.copy()
    assert s4 == s4.copy()
    assert s5 == s5.copy()

    assert s1 < s2
    assert s2 < s3
    assert s3 < s4
    assert s4 < s5

    assert s1 <= s2
    assert s2 <= s3
    assert s3 <= s4

    assert s2 > s1
    assert s3 > s2
    assert s4 > s3
    assert s5 > s4

    assert s2 >= s1
    assert s3 >= s2
    assert s4 >= s3
    assert s5 >= s4

    assert s1.is_unpolarized
    assert not s1.is_polarized
    assert not s1.is_noncolinear
    assert not s1.is_spinorbit
    assert not s1.is_nambu

    assert not s2.is_unpolarized
    assert s2.is_polarized
    assert not s2.is_noncolinear
    assert not s2.is_spinorbit
    assert not s2.is_nambu

    assert not s3.is_unpolarized
    assert not s3.is_polarized
    assert s3.is_noncolinear
    assert not s3.is_spinorbit
    assert not s3.is_nambu

    assert not s4.is_unpolarized
    assert not s4.is_polarized
    assert not s4.is_noncolinear
    assert s4.is_spinorbit
    assert not s4.is_nambu

    assert not s5.is_unpolarized
    assert not s5.is_polarized
    assert not s5.is_noncolinear
    assert not s5.is_spinorbit
    assert s5.is_nambu


def test_spin_unaccepted_arg():
    with pytest.raises(ValueError):
        s = Spin("satoehus")


def test_spin_pauli():
    # just grab the default spin
    S = Spin()

    # Create a fictituous wave-function
    sq2 = 2**0.5
    W = np.array(
        [
            [1 / sq2, 1 / sq2],  # M_x = 1
            [1 / sq2, -1 / sq2],  # M_x = -1
            [0.5 + 0.5j, 0.5 + 0.5j],  # M_x = 1
            [0.5 - 0.5j, -0.5 + 0.5j],  # M_x = -1
            [1 / sq2, 1j / sq2],  # M_y = 1
            [1 / sq2, -1j / sq2],  # M_y = -1
            [0.5 - 0.5j, 0.5 + 0.5j],  # M_y = 1
            [0.5 + 0.5j, 0.5 - 0.5j],  # M_y = -1
            [1, 0],  # M_z = 1
            [0, 1],  # M_z = -1
        ]
    )
    x = np.array([1, -1, 1, -1, 0, 0, 0, 0, 0, 0])
    assert np.allclose(x, (np.conj(W) * S.X.dot(W.T).T).sum(1).real)
    y = np.array([0, 0, 0, 0, 1, -1, 1, -1, 0, 0])
    assert np.allclose(y, (np.conj(W) * np.dot(S.Y, W.T).T).sum(1).real)
    z = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, -1])
    assert np.allclose(z, (np.conj(W) * np.dot(S.Z, W.T).T).sum(1).real)


def test_spin_pickle():
    import pickle as p

    S = Spin("nc")
    n = p.dumps(S)
    s = p.loads(n)
    assert S == s
