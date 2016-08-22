from __future__ import print_function, division

from nose.tools import *

from sisl.units import *

class TestUnits(object):

    def test_group(self):
        assert_equal(unit_group('kg'), 'mass')
        assert_equal(unit_group('eV'), 'energy')
        assert_equal(unit_group('N'), 'force')

    def test_unit_convert(self):
        assert_almost_equal(unit_convert('kg', 'g'), 1.e3)
        assert_almost_equal(unit_convert('eV', 'J'), 1.60217733e-19)
        assert_almost_equal(unit_convert('J', 'eV'), 1./1.60217733e-19)

    def test_default(self):
        assert_equal(unit_default('mass'), 'amu')
        assert_equal(unit_default('energy'), 'eV')
        assert_equal(unit_default('force'), 'eV/Ang')

