""" pytest test configures """
from __future__ import print_function

import pytest
import numpy as np

import sisl


pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = 'sisl/io/siesta'


def test_si_pdos_kgrid_tsde_dm(sisl_files):
    fdf = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.fdf'), base=sisl_files(_dir))

    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.TSDE'))

    DM1 = si.read_density_matrix(geometry=fdf.read_geometry())
    DM2 = fdf.read_density_matrix(order=['TSDE'])

    assert DM1._csr.spsame(DM2._csr)
    assert np.allclose(DM1._csr._D, DM2._csr._D)


def test_si_pdos_kgrid_tsde_edm(sisl_files):
    fdf = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.fdf'), base=sisl_files(_dir))

    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.TSDE'))

    EDM1 = si.read_energy_density_matrix(geometry=fdf.read_geometry())
    EDM2 = fdf.read_energy_density_matrix(order=['TSDE'])

    assert EDM1._csr.spsame(EDM2._csr)
    assert np.allclose(EDM1._csr._D, EDM2._csr._D)
