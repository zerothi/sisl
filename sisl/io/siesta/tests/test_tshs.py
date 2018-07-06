""" pytest test configures """
from __future__ import print_function

import pytest
import numpy as np
import sisl


pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = 'sisl/io/siesta'


def test_si_pdos_kgrid_tshs(sisl_files, sisl_tmp):
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.TSHS'))
    HS1 = si.read_hamiltonian()
    f = sisl_tmp('tmp.TSHS', _dir)
    HS1.write(f)
    si = sisl.get_sile(f)
    HS2 = si.read_hamiltonian()
    assert HS1._csr.spsame(HS2._csr)
    HS1.finalize()
    HS2.finalize()
    assert np.allclose(HS1._csr._D, HS2._csr._D)


def test_si_pdos_kgrid_tshs_overlap(sisl_files, sisl_tmp):
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.TSHS'))
    HS = si.read_hamiltonian()
    S = si.read_overlap()
    assert HS._csr.spsame(S._csr)
    HS.finalize()
    S.finalize()
    assert np.allclose(HS._csr._D[:, HS.S_idx], S._csr._D[:, 0])
