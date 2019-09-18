""" pytest test configures """
from __future__ import print_function

import pytest
import os.path as osp
import numpy as np
import sisl


pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = osp.join('sisl', 'io', 'siesta')


def si_pdos_kgrid_geom():
    return sisl.Geometry([[0, 0, 0], [1, 1, 1]], sisl.Atom('Si', R=np.arange(13) + 1))


def test_si_pdos_kgrid_hsx_H(sisl_files, sisl_tmp):
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.HSX'))
    si.read_hamiltonian(geometry=si_pdos_kgrid_geom())


def test_si_pdos_kgrid_hsx_overlap(sisl_files, sisl_tmp):
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.HSX'))
    HS = si.read_hamiltonian(geometry=si_pdos_kgrid_geom())
    S = si.read_overlap(geometry=si_pdos_kgrid_geom())

    assert HS._csr.spsame(S._csr)
    assert np.allclose(HS._csr._D[:, HS.S_idx], S._csr._D[:, 0])
