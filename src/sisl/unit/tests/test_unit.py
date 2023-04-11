# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pytest
approx = pytest.approx

import numpy as np

from sisl.unit import unit_group, unit_convert, unit_default, units

pytestmark = pytest.mark.unit


def test_group():
    assert unit_group('kg') == 'mass'
    assert unit_group('eV') == 'energy'
    assert unit_group('N') == 'force'


def test_unit_convert():
    assert approx(unit_convert('kg', 'g')) == 1.e3
    assert approx(unit_convert('eV', 'J')) == 1.60217733e-19
    assert approx(unit_convert('J', 'eV')) == 1/1.60217733e-19
    assert approx(unit_convert('J', 'eV', opts={'^': 2})) == (1/1.60217733e-19) ** 2
    assert approx(unit_convert('J', 'eV', opts={'/': 2})) == (1/1.60217733e-19) / 2
    assert approx(unit_convert('J', 'eV', opts={'*': 2})) == (1/1.60217733e-19) * 2


def test_class_unit():
    assert np.allclose(units.convert('J', 'J', 'J'), 1)

    assert approx(units.convert('kg', 'g')) == 1.e3
    assert approx(units.convert('eV', 'J')) == 1.60217733e-19
    assert approx(units.convert('J', 'eV')) == 1/1.60217733e-19
    assert approx(units.convert('J^2', 'eV**2')) == (1/1.60217733e-19) ** 2
    assert approx(units.convert('J/2', 'eV/2')) == (1/1.60217733e-19)
    assert approx(units.convert('J', 'eV/2')) == (1/1.60217733e-19) * 2
    assert approx(units.convert('J2', '2eV')) == (1/1.60217733e-19)
    assert approx(units.convert('J2', 'eV')) == (1/1.60217733e-19) * 2
    assert approx(units.convert('J/m', 'eV/Ang')) == unit_convert('J', 'eV') / unit_convert('m', 'Ang')
    units('J**eV', 'eV**eV')
    units('J/m', 'eV/m')


def test_default():
    assert unit_default('mass') == 'amu'
    assert unit_default('energy') == 'eV'
    assert unit_default('force') == 'eV/Ang'


def test_group_f1():
    with pytest.raises(ValueError):
        unit_group('not-existing')


def test_default_f1():
    with pytest.raises(ValueError):
        unit_default('not-existing')


def test_unit_convert_f1():
    with pytest.raises(ValueError):
        unit_convert('eV', 'megaerg')


def test_unit_convert_f2():
    with pytest.raises(ValueError):
        unit_convert('eV', 'kg')
