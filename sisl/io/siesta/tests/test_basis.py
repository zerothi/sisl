# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pytest
import os.path as osp
from io import StringIO
from sisl.io.siesta.basis import *


pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = osp.join('sisl', 'io', 'siesta')


def test_si_ion_nc(sisl_files):
    f = sisl_files(_dir, 'Si.ion.nc')
    with ionncSileSiesta(f) as sile:
        atom = sile.read_basis()

    # Number of orbitals
    assert len(atom) == 13


def test_si_ion_xml(sisl_files):
    f = sisl_files(_dir, 'Si.ion.xml')
    with ionxmlSileSiesta(f) as sile:
        atom = sile.read_basis()

    # Number of orbitals
    assert len(atom) == 13


def test_si_ion_xml_handle(sisl_files):
    f = open(sisl_files(_dir, 'Si.ion.xml'), 'r')

    with ionxmlSileSiesta(f) as sile:
        assert "Buffer" in sile.__class__.__name__
        atom = sile.read_basis()

    # Number of orbitals
    assert len(atom) == 13


def test_si_ion_xml_stringio(sisl_files):
    f = StringIO(open(sisl_files(_dir, 'Si.ion.xml'), 'r').read())

    with ionxmlSileSiesta(f) as sile:
        assert "Buffer" in sile.__class__.__name__
        atom = sile.read_basis()

    # Number of orbitals
    assert len(atom) == 13


def test_si_ion_compare(sisl_files):
    f = sisl_files(_dir, 'Si.ion.nc')
    with ionncSileSiesta(f) as sile:
        nc = sile.read_basis()

    f = sisl_files(_dir, 'Si.ion.xml')
    with ionxmlSileSiesta(f) as sile:
        xml = sile.read_basis()

    assert nc == xml
