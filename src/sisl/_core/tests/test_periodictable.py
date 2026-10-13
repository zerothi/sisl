# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl import PeriodicTable

pytestmark = [pytest.mark.atom, pytest.mark.periodictable]


@pytest.fixture
def setup():
    class t:
        def __init__(self):
            self.PT = PeriodicTable()

    return t()


def test_z(setup):
    a = setup.PT.Z([1, 2])
    assert len(a) == 2
    assert a[0] == 1
    assert a[1] == 2
    a = setup.PT.Z(1)
    assert a == 1


def test_z_row(setup):
    assert np.array_equal(setup.PT.Z_row([0, 1, 2]), [-1, 1, 1])
    assert setup.PT.Z_row(1) == 1
    assert setup.PT.Z_row(0) == -1


def test_z_column(setup):
    assert np.array_equal(setup.PT.Z_column([0, 1, 2]), [-1, 1, 18])
    assert setup.PT.Z_column(1) == 1
    assert setup.PT.Z_column(0) == -1


def test_z_label(setup):
    a = setup.PT.Z_label(["H", 2])
    assert len(a) == 2
    assert a[0] == "H"
    assert a[1] == "He"
    a = setup.PT.Z_label(1)
    assert a == "H"


def test_atomic_mass(setup):
    assert setup.PT.atomic_mass(1) == setup.PT.atomic_mass("H")
    assert np.allclose(setup.PT.atomic_mass([1, 2]), setup.PT.atomic_mass(["H", "He"]))


def test_radius(setup):
    PT = setup.PT
    for m in ["calc", "empirical", "vdw"]:
        assert PT.radius(1, method=m) == PT.radius("H", method=m)
        assert np.allclose(
            PT.radius([1, 2], method=m), PT.radius(["H", "He"], method=m)
        )


def test_radius_unknown(setup):
    with pytest.raises(AttributeError):
        setup.PT.radius(1, method="unknown")
