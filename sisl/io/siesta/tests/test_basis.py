# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pytest
import os.path as osp
from sisl.io.siesta.basis import *


pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = osp.join('sisl', 'io', 'siesta')


def test_si_ion_nc(sisl_files):
    f = sisl_files(_dir, 'Si.ion.nc')
    atom = ionncSileSiesta(f).read_basis()

    # Number of orbitals
    assert len(atom) == 13


def test_si_ion_xml(sisl_files):
    f = sisl_files(_dir, 'Si.ion.xml')
    atom = ionxmlSileSiesta(f).read_basis()

    # Number of orbitals
    assert len(atom) == 13


def test_si_ion_compare(sisl_files):
    f = sisl_files(_dir, 'Si.ion.nc')
    nc = ionncSileSiesta(f).read_basis()
    f = sisl_files(_dir, 'Si.ion.xml')
    xml = ionxmlSileSiesta(f).read_basis()

    assert nc == xml
