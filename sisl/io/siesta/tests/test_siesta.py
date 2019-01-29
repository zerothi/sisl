from __future__ import print_function, division

import pytest
import numpy as np

from sisl import Hamiltonian, DynamicalMatrix
from sisl.io.siesta import *


pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = 'sisl/io/siesta'


def test_nc1(sisl_tmp, sisl_system):
    f = sisl_tmp('gr.nc', _dir)
    tb = Hamiltonian(sisl_system.gtb)
    tb.construct([sisl_system.R, sisl_system.t])
    tb.write(ncSileSiesta(f, 'w'))

    ntb = ncSileSiesta(f).read_hamiltonian()

    # Assert they are the same
    assert np.allclose(tb.cell, ntb.cell)
    assert np.allclose(tb.xyz, ntb.xyz)
    tb.finalize()
    ntb.finalize()
    assert np.allclose(tb._csr._D[:, 0], ntb._csr._D[:, 0])
    assert sisl_system.g.atom.equal(ntb.atom, R=False)


def test_nc2(sisl_tmp, sisl_system):
    f = sisl_tmp('grS.nc', _dir)
    tb = Hamiltonian(sisl_system.gtb, orthogonal=False)
    tb.construct([sisl_system.R, sisl_system.tS])
    tb.write(ncSileSiesta(f, 'w'))

    ntb = ncSileSiesta(f).read_hamiltonian()

    # Assert they are the same
    assert np.allclose(tb.cell, ntb.cell)
    assert np.allclose(tb.xyz, ntb.xyz)
    tb.finalize()
    ntb.finalize()
    assert np.allclose(tb._csr._D[:, 0], ntb._csr._D[:, 0])
    assert sisl_system.g.atom.equal(ntb.atom, R=False)


def test_nc_overlap(sisl_tmp, sisl_system):
    f = sisl_tmp('gr.nc', _dir)
    tb = Hamiltonian(sisl_system.gtb)
    tb.construct([sisl_system.R, sisl_system.t])
    tb.write(ncSileSiesta(f, 'w'))

    S = ncSileSiesta(f).read_overlap()

    # Ensure no empty data-points
    S.finalize()
    assert np.allclose(S._csr._D.sum(), tb.no)


def test_nc_dynamical_matrix(sisl_tmp, sisl_system):
    f = sisl_tmp('grS.nc', _dir)
    dm = DynamicalMatrix(sisl_system.gtb)
    for _, ix in dm:
        dm[ix, ix] = ix / 2.
    dm.write(ncSileSiesta(f, 'w'))

    ndm = ncSileSiesta(f).read_dynamical_matrix()

    # Assert they are the same
    assert np.allclose(dm.cell, ndm.cell)
    assert np.allclose(dm.xyz, ndm.xyz)
    dm.finalize()
    ndm.finalize()
    assert np.allclose(dm._csr._D[:, 0], ndm._csr._D[:, 0])
    assert sisl_system.g.atom.equal(ndm.atom, R=False)
