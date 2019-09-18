from __future__ import print_function, division

import pytest
import os.path as osp
from sisl import units
from sisl.io.wannier90 import *
import numpy as np


pytestmark = [pytest.mark.io, pytest.mark.wannier90, pytest.mark.w90]
_dir = osp.join('sisl', 'io', 'wannier90')


@pytest.mark.parametrize("unit", ['', 'Ang\n', 'ang\n', 'bohr\n'])
def test_seedname_read_frac(sisl_tmp, unit):
    f = sisl_tmp('read_frac.win', _dir)
    with open(f, 'w') as fh:
        fh.write("""
begin unit_cell_cart
  {} 2. 0. 0.
  0. 2. 0
  0 0 2.
end unit_cell_cart

begin atoms_frac
  C 0.5 0.5 0.5
end
""".format(unit))
    g = winSileWannier90(f).read_geometry()

    if len(unit) == 0:
        unit = 'ang'
    unit = units(unit.strip().capitalize(), 'Ang')

    assert np.allclose(g.cell, np.identity(3) * 2 * unit)
    assert np.allclose(g.xyz, [1 * unit] * 3)


@pytest.mark.parametrize("unit_sc", ['', 'Ang\n', 'ang\n', 'bohr\n'])
@pytest.mark.parametrize("unit", ['', 'Ang\n', 'ang\n', 'bohr\n'])
def test_seedname_read_coord(sisl_tmp, unit_sc, unit):
    f = sisl_tmp('read_coord.win', _dir)
    with open(f, 'w') as fh:
        fh.write("""
begin unit_cell_cart
  {} 2. 0. 0.
  0. 2. 0
  0 0 2.
end unit_cell_cart

begin atoms_cart
  {} C 0.5 0.5 0.5
end
""".format(unit_sc, unit))
    g = winSileWannier90(f).read_geometry()

    if len(unit) == 0:
        unit = 'ang'
    unit = units(unit.strip().capitalize(), 'Ang')

    if len(unit_sc) == 0:
        unit_sc = 'ang'
    unit_sc = units(unit_sc.strip().capitalize(), 'Ang')

    assert np.allclose(g.cell, np.identity(3) * 2 * unit_sc)
    assert np.allclose(g.xyz, [0.5 * unit] * 3)


@pytest.mark.parametrize("frac", [True, False])
def test_seedname_write_read(sisl_tmp, sisl_system, frac):
    f = sisl_tmp('write_read.win', _dir)
    sile = winSileWannier90(f, 'w')
    sile.write_geometry(sisl_system.g, frac=frac)

    g = winSileWannier90(f).read_geometry()
    assert np.allclose(g.cell, sisl_system.g.cell)
    assert np.allclose(g.xyz, sisl_system.g.xyz)
