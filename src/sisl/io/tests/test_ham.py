# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

import sisl as si
from sisl.io.ham import *

pytestmark = [pytest.mark.io, pytest.mark.generic]


def test_ham_geometry(sisl_tmp, sisl_system):
    f = sisl_tmp("gr.ham")
    sisl_system.g.write(hamiltonianSile(f, "w"))
    g = hamiltonianSile(f).read_geometry()

    # Assert they are the same
    assert np.allclose(g.cell, sisl_system.g.cell)
    assert np.allclose(g.xyz, sisl_system.g.xyz)
    assert g.atoms.equal(sisl_system.g.atoms, R=False)


def test_ham_nn(sisl_tmp, sisl_system):
    f = sisl_tmp("gr.ham")
    sisl_system.ham.write(hamiltonianSile(f, "w"))
    ham = hamiltonianSile(f).read_hamiltonian()
    assert ham.spsame(sisl_system.ham)


@pytest.mark.parametrize("hermitian", [True, False])
def test_ham_3nn(sisl_tmp, hermitian):
    f = sisl_tmp("gr.ham")

    g = si.geom.graphene()
    H = si.Hamiltonian(g)

    # build a 3rd nearest neighbor model
    H.construct([[0.1, 1.6, 2.6, 3.1], [0, -2.7, -0.20, -0.18]])

    H.write(f, hermitian=hermitian)

    H1 = hamiltonianSile(f).read_hamiltonian()
    assert np.allclose((H1 - H)._csr._D, 0)

    if not hermitian:
        H1 = hamiltonianSile(f).read_hamiltonian(hermitian=hermitian)
        assert np.allclose((H1 - H)._csr._D, 0)


@pytest.mark.parametrize("hermitian", [True, False])
def test_ham_3nn_no(sisl_tmp, hermitian):
    f = sisl_tmp("gr.ham")

    g = si.geom.graphene()
    g.cell[2, 2] = 1.00
    H = si.Hamiltonian(g, orthogonal=False)
    H.set_nsc([5, 5, 5])

    # build a 3rd nearest neighbor, nonorthogonal model
    H.construct(
        [[0.1, 1.6, 2.6, 3.1], [(1, 1), (-2.7, 0.073), (-0.09, 0.045), (-0.33, 0.026)]]
    )

    H.write(f, hermitian=hermitian)

    H1 = hamiltonianSile(f).read_hamiltonian()
    assert np.allclose((H1 - H)._csr._D, 0)

    if not hermitian:
        H1 = hamiltonianSile(f).read_hamiltonian(hermitian=hermitian)
        assert np.allclose((H1 - H)._csr._D, 0)
