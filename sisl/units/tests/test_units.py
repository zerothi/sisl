from __future__ import print_function, division

import pytest

from sisl.units import *


class TestUnits(object):

    def test_group(self):
        assert unit_group('kg') == 'mass'
        assert unit_group('eV') == 'energy'
        assert unit_group('N') == 'force'

    def test_unit_convert(self):
        assert pytest.approx(unit_convert('kg', 'g')) == 1.e3
        assert pytest.approx(unit_convert('eV', 'J')) == 1.60217733e-19
        assert pytest.approx(unit_convert('J', 'eV')) == 1/1.60217733e-19

    def test_default(self):
        assert unit_default('mass') == 'amu'
        assert unit_default('energy') == 'eV'
        assert unit_default('force') == 'eV/Ang'
