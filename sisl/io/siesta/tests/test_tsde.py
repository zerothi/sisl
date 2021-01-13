""" pytest test configures """

import pytest
import os.path as osp
import numpy as np
import sisl


pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = osp.join('sisl', 'io', 'siesta')


def test_si_pdos_kgrid_tsde_dm(sisl_files):
    fdf = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.fdf'), base=sisl_files(_dir))

    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.TSDE'))

    DM1 = si.read_density_matrix(geometry=fdf.read_geometry())
    DM2 = fdf.read_density_matrix(order=['TSDE'])

    Ef1 = si.read_fermi_level()
    Ef2 = fdf.read_fermi_level()

    assert Ef1 == Ef2

    assert DM1._csr.spsame(DM2._csr)
    assert np.allclose(DM1._csr._D[:, :-1], DM2._csr._D[:, :-1])


def test_si_pdos_kgrid_tsde_edm(sisl_files):
    fdf = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.fdf'), base=sisl_files(_dir))

    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.TSDE'))

    EDM1 = si.read_energy_density_matrix(geometry=fdf.read_geometry())
    EDM2 = fdf.read_energy_density_matrix(order=['TSDE'])

    assert EDM1._csr.spsame(EDM2._csr)
    assert np.allclose(EDM1._csr._D[:, :-1], EDM2._csr._D[:, :-1])


@pytest.mark.filterwarnings("ignore:wrong sparse pattern")
def test_si_pdos_kgrid_tsde_dm_edm_rw(sisl_files, sisl_tmp):
    fdf = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.fdf'), base=sisl_files(_dir))
    geom = fdf.read_geometry()
    f1 = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.TSDE'))

    DM1 = f1.read_density_matrix(geometry=geom)
    EDM1 = f1.read_energy_density_matrix(geometry=geom)

    f2 = sisl.get_sile(sisl_tmp('noEf.TSDE', _dir))
    # by default everything gets sorted...
    f2.write_density_matrices(DM1, EDM1, sort=False)
    DM2 = f2.read_density_matrix(geometry=geom)
    EDM2 = f2.read_energy_density_matrix(geometry=geom)
    assert DM1._csr.spsame(DM2._csr)
    assert np.allclose(DM1._csr._D[:, :-1], DM2._csr._D[:, :-1])
    assert EDM1._csr.spsame(EDM2._csr)
    assert np.allclose(EDM1._csr._D[:, :-1], EDM2._csr._D[:, :-1])

    # Now the matrices ARE finalized, we don't have to do anything again
    EDM2 = EDM1.copy()
    EDM2.shift(-2., DM1)
    f3 = sisl.get_sile(sisl_tmp('Ef.TSDE', _dir))
    f3.write_density_matrices(DM1, EDM2, Ef=-2., sort=False)
    DM3 = f3.read_density_matrix(geometry=geom)
    EDM3 = f3.read_energy_density_matrix(geometry=geom)
    assert DM1._csr.spsame(DM3._csr)
    assert np.allclose(DM1._csr._D[:, :-1], DM3._csr._D[:, :-1])
    assert EDM1._csr.spsame(EDM3._csr)
    assert np.allclose(EDM1._csr._D[:, :-1], EDM3._csr._D[:, :-1])

    f3.write_density_matrices(DM1, EDM2, Ef=-2., sort=True)
    DM3 = f3.read_density_matrix(geometry=geom, sort=False)
    EDM3 = f3.read_energy_density_matrix(geometry=geom, sort=False)
    assert DM1._csr.spsame(DM3._csr)
    assert not np.allclose(DM1._csr._D[:, :-1], DM3._csr._D[:, :-1])
    DM3.finalize()
    assert np.allclose(DM1._csr._D[:, :-1], DM3._csr._D[:, :-1])
    assert EDM1._csr.spsame(EDM3._csr)
    assert not np.allclose(EDM1._csr._D[:, :-1], EDM3._csr._D[:, :-1])
