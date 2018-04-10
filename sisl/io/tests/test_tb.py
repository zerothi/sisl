from __future__ import print_function, division

import pytest
import numpy as np

from sisl.io.ham import *


pytestmark = pytest.mark.io
_dir = 'sisl/io'


def test_ham1(sisl_tmp, sisl_system):
    f = sisl_tmp('gr.ham', _dir)
    sisl_system.g.write(HamiltonianSile(f, 'w'))
    g = HamiltonianSile(f).read_geometry()

    # Assert they are the same
    assert np.allclose(g.cell, sisl_system.g.cell)
    assert np.allclose(g.xyz, sisl_system.g.xyz)
    assert g.atom.equal(sisl_system.g.atom, R=False)


def test_ham2(sisl_tmp, sisl_system):
    f = sisl_tmp('gr.ham', _dir)
    sisl_system.ham.write(HamiltonianSile(f, 'w'))
    ham = HamiltonianSile(f).read_hamiltonian()
    assert ham.spsame(sisl_system.ham)
