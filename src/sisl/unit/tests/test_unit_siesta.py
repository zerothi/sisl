# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import pytest

approx = pytest.approx

from sisl.unit.siesta import unit_convert, unit_default, unit_group

pytestmark = [pytest.mark.unit, pytest.mark.siesta]


def test_group():
    assert unit_group("kg") == "mass"
    assert unit_group("eV") == "energy"
    assert unit_group("N") == "force"


def test_unit_convert():
    assert approx(unit_convert("kg", "g")) == 1.0e3
    assert approx(unit_convert("eV", "J")) == 1.602176634e-19
    assert approx(unit_convert("J", "eV")) == 1 / 1.602176634e-19
    assert approx(unit_convert("J", "eV", {"^": 2})) == (1 / 1.602176634e-19) ** 2
    assert approx(unit_convert("J", "eV", {"/": 2})) == (1 / 1.602176634e-19) / 2
    assert approx(unit_convert("J", "eV", {"*": 2})) == (1 / 1.602176634e-19) * 2


def test_default():
    assert unit_default("mass") == "amu"
    assert unit_default("energy") == "eV"
    assert unit_default("force") == "eV/Ang"


def test_group_f1():
    with pytest.raises(ValueError):
        unit_group("not-existing")


def test_default_f1():
    with pytest.raises(ValueError):
        unit_default("not-existing")


def test_unit_convert_f1():
    with pytest.raises(ValueError):
        unit_convert("eV", "megaerg")


def test_unit_convert_f2():
    with pytest.raises(ValueError):
        unit_convert("eV", "kg")
