from __future__ import print_function, division

import pytest

import sisl
import numpy as np

pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = 'sisl/io/siesta'


def test_gf_write(sisl_tmp, sisl_system):
    tb = sisl.Hamiltonian(sisl_system.gtb)
    f = sisl_tmp('file.TSGF', _dir)
    gf = sisl.io.get_sile(f)
    bz = sisl.MonkhorstPack(tb, [3, 3, 1])
    E = np.linspace(-2, 2, 20) + 1j * 1e-4
    S = np.eye(len(tb), dtype=np.complex128)

    gf.write_header(E, bz, tb)
    for i, (write_hs, k, e) in enumerate(gf):
        Hk = tb.Hk(k, format='array')
        if write_hs and i % 2 == 0:
            gf.write_hamiltonian(Hk)
        elif write_hs:
            gf.write_hamiltonian(Hk, S)
        gf.write_self_energy(S * e - Hk)


def test_gf_write_read(sisl_tmp, sisl_system):
    tb = sisl.Hamiltonian(sisl_system.gtb)
    f = sisl_tmp('file.TSGF', _dir)

    bz = sisl.MonkhorstPack(tb, [3, 3, 1])
    E = np.linspace(-2, 2, 20) + 1j * 1e-4
    S = np.eye(len(tb), dtype=np.complex128)

    gf = sisl.io.get_sile(f)

    gf.write_header(E, bz, tb)
    for i, (write_hs, k, e) in enumerate(gf):
        Hk = tb.Hk(k, format='array')
        if write_hs and i % 2 == 0:
            gf.write_hamiltonian(Hk)
        elif write_hs:
            gf.write_hamiltonian(Hk, S)
        gf.write_self_energy(S * e - Hk)

    no_u, k, E_file = gf.read_header()
    assert np.allclose(E, E_file)
    assert np.allclose(k, bz.k)

    for i, (write_hs, k, e) in enumerate(gf):
        Hk = tb.Hk(k, format='array')
        if write_hs and i % 2 == 0:
            Hk_file, _ = gf.read_hamiltonian()
        elif write_hs:
            Hk_file, Sk_file = gf.read_hamiltonian()
            assert np.allclose(S, Sk_file)
        assert np.allclose(Hk, Hk_file)

        SE_file = gf.read_self_energy()
        assert np.allclose(SE_file, S * e - Hk)
