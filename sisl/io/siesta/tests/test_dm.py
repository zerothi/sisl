""" pytest test configures """
from __future__ import print_function

import pytest
import numpy as np

import sisl


pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = 'sisl/io/siesta'


def test_si_pdos_kgrid_dm(sisl_files):
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.DM'))
    DM1 = si.read_density_matrix()

    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.fdf'), base=sisl_files(_dir))
    DM2 = si.read_density_matrix(order=['DM'])

    # Force the shapes to align
    # This is because reading from fdf tries to read the correct
    # supercell from elsewere.
    csr1 = DM1._csr
    csr2 = DM2._csr
    csr1._shape = csr2._shape
    assert csr1.spsame(csr2)
    assert np.allclose(csr1._D, csr2._D)
