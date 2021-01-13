import re
import os.path as osp
import numpy as np
import pytest
import sisl


pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = osp.join('sisl', 'io', 'siesta')


def si_pdos_kgrid_geom(with_orbs=True):
    if with_orbs:
        return sisl.geom.diamond(5.43, sisl.Atom('Si', R=np.arange(13) + 1))
    return sisl.geom.diamond(5.43, sisl.Atom('Si'))



def test_si_pdos_kgrid_hsx_H(sisl_files, sisl_tmp):
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.HSX'))
    si.read_hamiltonian(geometry=si_pdos_kgrid_geom())


def test_si_pdos_kgrid_hsx_H_fix_orbitals(sisl_files, sisl_tmp):
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.HSX'))
    si.read_hamiltonian(geometry=si_pdos_kgrid_geom(False))


def test_si_pdos_kgrid_hsx_H_no_geom(sisl_files, sisl_tmp):
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.HSX'))
    si.read_hamiltonian()


def test_si_pdos_kgrid_hsx_overlap(sisl_files, sisl_tmp):
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.HSX'))
    HS = si.read_hamiltonian(geometry=si_pdos_kgrid_geom())
    S = si.read_overlap(geometry=si_pdos_kgrid_geom(False))

    assert HS._csr.spsame(S._csr)
    assert np.allclose(HS._csr._D[:, HS.S_idx], S._csr._D[:, 0])


def test_h2o_dipole_hsx(sisl_files, sisl_tmp):
    HSX = sisl.get_sile(sisl_files(_dir, 'h2o_dipole.HSX'))
    geometry = sisl.get_sile(sisl_files(_dir, 'h2o_dipole.fdf')).read_geometry()
    # manually define this.
    # The fdf does not contain it, and the following test asserts it will fail
    # otherwise.
    geometry.set_nsc(a=5, b=1, c=3)
    # reading from hsx just requires atoms + coordinates + nsc
    HS = HSX.read_hamiltonian(geometry=geometry)
    S = HSX.read_overlap(geometry=geometry)

    assert HS._csr.spsame(S._csr)
    assert np.allclose(HS._csr._D[:, HS.S_idx], S._csr._D[:, 0])


def test_h2o_dipole_hsx_fail(sisl_files, sisl_tmp):
    HSX = sisl.get_sile(sisl_files(_dir, 'h2o_dipole.HSX'))
    with pytest.raises(ValueError, match=re.escape("xij(orb) -> xij(atom)")):
        HS = HSX.read_hamiltonian()
    with pytest.raises(ValueError, match=re.escape("xij(orb) -> xij(atom)")):
        S = HSX.read_overlap()
