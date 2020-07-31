import pytest
approx = pytest.approx

from sisl.unit.siesta import unit_group, unit_convert, unit_default, unit_table_siesta

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('tbl', [None, unit_table_siesta])
def test_group(tbl):
    assert unit_group('kg', tbl) == 'mass'
    assert unit_group('eV', tbl) == 'energy'
    assert unit_group('N', tbl) == 'force'


@pytest.mark.parametrize('tbl', [None, unit_table_siesta])
def test_unit_convert(tbl):
    assert approx(unit_convert('kg', 'g', tbl=tbl)) == 1.e3
    assert approx(unit_convert('eV', 'J', tbl=tbl)) == 1.60219e-19
    assert approx(unit_convert('J', 'eV', tbl=tbl)) == 1/1.60219e-19
    assert approx(unit_convert('J', 'eV', {'^': 2}, tbl)) == (1/1.60219e-19) ** 2
    assert approx(unit_convert('J', 'eV', {'/': 2}, tbl)) == (1/1.60219e-19) / 2
    assert approx(unit_convert('J', 'eV', {'*': 2}, tbl)) == (1/1.60219e-19) * 2


@pytest.mark.parametrize('tbl', [None, unit_table_siesta])
def test_default(tbl):
    assert unit_default('mass', tbl) == 'amu'
    assert unit_default('energy', tbl) == 'eV'
    assert unit_default('force', tbl) == 'eV/Ang'


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
