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


def test_tshs_warn(sisl_files, sisl_tmp):
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.TSHS'))

    # check number of orbitals
    geom = si.read_geometry()
    geom._atoms = sisl.Atoms([sisl.Atom(i + 1) for i in range(geom.na)])
    with pytest.warns(sisl.SislWarning, match='number of orbitals'):
        si.read_hamiltonian(geometry=geom)

    # check cell
    geom = si.read_geometry()
    geom.sc.cell[:, :] = 1.
    with pytest.warns(sisl.SislWarning, match='lattice vectors'):
        si.read_hamiltonian(geometry=geom)

    # check atomic coordinates
    geom = si.read_geometry()
    geom.xyz[0, :] += 10.
    with pytest.warns(sisl.SislWarning, match='atomic coordinates'):
        si.read_hamiltonian(geometry=geom)

    # check supercell
    geom = si.read_geometry()
    geom.set_nsc([1, 1, 1])
    with pytest.warns(sisl.SislWarning, match='supercell'):
        si.read_hamiltonian(geometry=geom)


@pytest.mark.xfail(raises=sisl.SileError)
def test_tshs_error(sisl_files, sisl_tmp):
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.TSHS'))

    # check number of orbitals
    geom = si.read_geometry()
    geom = sisl.Geometry(np.random.rand(geom.na + 1, 3))
    si.read_hamiltonian(geometry=geom)


def test_si_pdos_kgrid_tshs_overlap(sisl_files, sisl_tmp):
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.TSHS'))
    HS = si.read_hamiltonian()
    S = si.read_overlap()
    assert HS._csr.spsame(S._csr)
    HS.finalize()
    S.finalize()
    assert np.allclose(HS._csr._D[:, HS.S_idx], S._csr._D[:, 0])
