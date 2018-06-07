from __future__ import print_function, division

import pytest
import numpy as np

from sisl import Hamiltonian
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
    assert np.allclose(tb._csr._D[:, 0], ntb._csr._D[:, 0])
    assert sisl_system.g.atom.equal(ntb.atom, R=False)
