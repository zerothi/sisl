from __future__ import print_function, division

import pytest
approx = pytest.approx

from sisl.unit.siesta import unit_group, unit_convert, unit_default

pytestmark = pytest.mark.unit


def test_group():
    assert unit_group('kg') == 'mass'
    assert unit_group('eV') == 'energy'
    assert unit_group('N') == 'force'


def test_unit_convert():
    assert approx(unit_convert('kg', 'g')) == 1.e3
    assert approx(unit_convert('eV', 'J')) == 1.60219e-19
    assert approx(unit_convert('J', 'eV')) == 1/1.60219e-19
    assert approx(unit_convert('J', 'eV', opts={'^': 2})) == (1/1.60219e-19) ** 2
    assert approx(unit_convert('J', 'eV', opts={'/': 2})) == (1/1.60219e-19) / 2
    assert approx(unit_convert('J', 'eV', opts={'*': 2})) == (1/1.60219e-19) * 2


def test_default():
    assert unit_default('mass') == 'amu'
    assert unit_default('energy') == 'Ry'
    assert unit_default('force') == 'Ry/Bohr'


@pytest.mark.xfail(raises=ValueError)
def test_group_f1():
    unit_group('not-existing')


@pytest.mark.xfail(raises=ValueError)
def test_default_f1():
    unit_default('not-existing')


@pytest.mark.xfail(raises=ValueError)
def test_unit_convert_f1():
    unit_convert('eV', 'megaerg')


@pytest.mark.xfail(raises=ValueError)
def test_unit_convert_f2():
    unit_convert('eV', 'kg')
