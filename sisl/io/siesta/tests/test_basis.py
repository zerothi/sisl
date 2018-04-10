from __future__ import print_function, division

import pytest

from sisl.io.siesta.basis import *

import numpy as np

pytestmark = pytest.mark.io
_dir = 'sisl/io/siesta'


def test_si_ion_nc(sisl_files):
    f = sisl_files(_dir, 'Si.ion.nc')
    atom = ionncSileSiesta(f).read_basis()

    # Number of orbitals
    assert len(atom) == 13
