# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

import sisl

pytestmark = [pytest.mark.io, pytest.mark.siesta]


def test_gf_write(sisl_tmp, sisl_system):
    tb = sisl.Hamiltonian(sisl_system.gtb)
    f = sisl_tmp("file.TSGF")
    gf = sisl.io.get_sile(f)
    bz = sisl.MonkhorstPack(tb, [3, 3, 1])
    E = np.linspace(-2, 2, 20) + 1j * 1e-4
    S = np.eye(len(tb), dtype=np.complex128)

    gf.write_header(bz, E)
    for i, (ispin, new_hs, k, e) in enumerate(gf):
        Hk = tb.Hk(k, format="array")
        assert ispin == 0
        if new_hs and i % 2 == 0:
            gf.write_hamiltonian(Hk)
        elif new_hs:
            gf.write_hamiltonian(Hk, S)
        gf.write_self_energy(S * e - Hk)


def test_gf_write_read(sisl_tmp, sisl_system):
    tb = sisl.Hamiltonian(sisl_system.gtb)
    f = sisl_tmp("file.TSGF")

    bz = sisl.MonkhorstPack(tb, [3, 3, 1])
    E = np.linspace(-2, 2, 20) + 1j * 1e-4
    S = np.eye(len(tb), dtype=np.complex128)

    gf = sisl.io.get_sile(f)

    gf.write_header(bz, E)
    for i, (ispin, write_hs, k, e) in enumerate(gf):
        assert ispin == 0
        Hk = tb.Hk(k, format="array")
        if write_hs and i % 2 == 0:
            gf.write_hamiltonian(Hk)
        elif write_hs:
            gf.write_hamiltonian(Hk, S)
        gf.write_self_energy(S * e - Hk)

    nspin, no_u, k, E_file = gf.read_header()
    assert nspin == 1
    assert np.allclose(E, E_file)
    assert np.allclose(k, bz.k)

    for i, (ispin, write_hs, k, e) in enumerate(gf):
        assert ispin == 0
        Hk = tb.Hk(k, format="array")
        if write_hs and i % 2 == 0:
            Hk_file, _ = gf.read_hamiltonian()
        elif write_hs:
            Hk_file, Sk_file = gf.read_hamiltonian()
            assert np.allclose(S, Sk_file)
        assert np.allclose(Hk, Hk_file)

        SE_file = gf.read_self_energy()
        assert np.allclose(SE_file, S * e - Hk)


def test_gf_write_read_spin(sisl_tmp, sisl_system):
    f = sisl_tmp("file.TSGF")

    tb = sisl.Hamiltonian(sisl_system.gtb, spin=sisl.Spin("P"))
    tb.construct([(0.1, 1.5), ([0.1, -0.1], [2.7, 1.6])])

    bz = sisl.MonkhorstPack(tb, [3, 3, 1])
    E = np.linspace(-2, 2, 3) + 1j * 1e-4
    S = np.eye(len(tb), dtype=np.complex128)

    gf = sisl.io.get_sile(f)

    gf.write_header(bz, E)
    for i, (ispin, write_hs, k, e) in enumerate(gf):
        Hk = tb.Hk(k, spin=ispin, format="array")
        if write_hs and i % 2 == 0:
            gf.write_hamiltonian(Hk)
        elif write_hs:
            gf.write_hamiltonian(Hk, S)
        gf.write_self_energy(S * e - Hk)

    # Check it isn't opened
    assert not gf._fortran_is_open()

    nspin, no_u, k, E_file = gf.read_header()
    assert nspin == 2
    assert np.allclose(E, E_file)
    assert np.allclose(k, bz.k)

    for i, (ispin, write_hs, k, e) in enumerate(gf):
        Hk = tb.Hk(k, spin=ispin, format="array")
        if write_hs and i % 2 == 0:
            Hk_file, _ = gf.read_hamiltonian()
        elif write_hs:
            Hk_file, Sk_file = gf.read_hamiltonian()
            assert np.allclose(S, Sk_file)
        assert np.allclose(Hk, Hk_file)

        SE_file = gf.read_self_energy()
        assert np.allclose(SE_file, S * e - Hk)


def test_gf_write_read_direct(sisl_tmp, sisl_system):
    f = sisl_tmp("file.TSGF")

    tb = sisl.Hamiltonian(sisl_system.gtb, spin=sisl.Spin("P"))
    tb.construct([(0.1, 1.5), ([0.1, -0.1], [2.7, 1.6])])

    bz = sisl.MonkhorstPack(tb, [3, 3, 1])
    E = np.linspace(-2, 2, 3) + 1j * 1e-4
    S = np.eye(len(tb), dtype=np.complex128)

    gf = sisl.io.get_sile(f)

    gf.write_header(bz, E)
    for i, (ispin, write_hs, k, e) in enumerate(gf):
        Hk = tb.Hk(k, spin=ispin, format="array")
        if write_hs and i % 2 == 0:
            gf.write_hamiltonian(Hk)
        elif write_hs:
            gf.write_hamiltonian(Hk, S)
        gf.write_self_energy(S * e - Hk)

    # ensure it is not opened
    assert not gf._fortran_is_open()

    # First try from beginning
    for e in [0, 1, E[1], 0, E[0]]:
        ie = gf.Eindex(e)
        SE1 = gf.self_energy(e, bz.k[2, :])
        assert gf._state == 1
        assert gf._ik == 2
        assert gf._iE == ie
        assert gf._ispin == 0
        assert gf._is_read == 1

        SE2 = gf.self_energy(e, bz.k[2, :], spin=1)
        assert gf._state == 1
        assert gf._ik == 2
        assert gf._iE == ie
        assert gf._ispin == 1
        assert gf._is_read == 1

        assert not np.allclose(SE1, SE2)

        # In the middle we read some hamiltonians
        H1, S1 = gf.HkSk(bz.k[2, :], spin=0)
        assert gf._state == 0
        assert gf._ik == 2
        assert gf._iE == 0
        assert gf._ispin == 0
        assert gf._is_read == 1
        assert np.allclose(S, S1)

        H2, S1 = gf.HkSk(bz.k[2, :], spin=1)
        assert gf._state == 0
        assert gf._ik == 2
        assert gf._iE == 0
        assert gf._ispin == 1
        assert gf._is_read == 1
        assert np.allclose(S, S1)
        assert not np.allclose(H1, H2)
        assert not np.allclose(H1, SE1)

        H2, S1 = gf.HkSk(bz.k[2, :], spin=0)
        assert gf._state == 0
        assert gf._ik == 2
        assert gf._iE == 0
        assert gf._ispin == 0
        assert gf._is_read == 1
        assert np.allclose(S, S1)
        assert np.allclose(H1, H2)

        # Now read self-energy
        SE2 = gf.self_energy(e, bz.k[2, :], spin=0)
        assert gf._state == 1
        assert gf._ik == 2
        assert gf._iE == ie
        assert gf._ispin == 0
        assert gf._is_read == 1

        assert np.allclose(SE1, SE2)


def test_gf_sile_error():
    with pytest.raises(sisl.SileError):
        sisl.get_sile("non_existing_file.TSGF").read_header()
