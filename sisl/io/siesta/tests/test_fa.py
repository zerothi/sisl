from __future__ import print_function, division

import pytest

from sisl.io.siesta.fa import *

import numpy as np

pytestmark = pytest.mark.io
_dir = 'sisl/io/siesta'


def test_si_pdos_kgrid_fa(sisl_files):
    f = sisl_files(_dir, 'si_pdos_kgrid.FA')
    fa = faSileSiesta(f).read_data()

    assert len(fa) == 2
    fa1 = faSileSiesta(f).read_force()
    assert np.allclose(fa, fa1)
