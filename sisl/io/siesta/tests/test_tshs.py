""" pytest test configures """

import pytest
import os.path as osp
import numpy as np
import sisl


pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = osp.join('sisl', 'io', 'siesta')


def test_tshs_si_pdos_kgrid(sisl_files, sisl_tmp):
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


def test_tshs_si_pdos_kgrid_tofromnc(sisl_files, sisl_tmp):
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.TSHS'))
    HS1 = si.read_hamiltonian()
    f = sisl_tmp('tmp.TSHS', _dir)
    fnc = sisl_tmp('tmp.nc', _dir)

    HS1.write(f)
    HS1.write(fnc)

    HS2 = sisl.get_sile(f).read_hamiltonian()
    HS2nc = sisl.get_sile(fnc).read_hamiltonian()
    assert HS1._csr.spsame(HS2._csr)
    assert HS1._csr.spsame(HS2nc._csr)
    HS1.finalize()
    HS2.finalize()
    assert np.allclose(HS1._csr._D, HS2._csr._D)
    HS2nc.finalize()
    assert np.allclose(HS1._csr._D, HS2nc._csr._D)


def test_tshs_si_pdos_kgrid_repeat_tile(sisl_files, sisl_tmp):
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.TSHS'))
    HS = si.read_hamiltonian()
    HSr = HS.repeat(3, 2).repeat(3, 0).repeat(3, 1)
    HSt = HS.tile(3, 2).tile(3, 0).tile(3, 1)
    assert np.allclose(HSr.eigh(), HSt.eigh())


def test_tshs_si_pdos_kgrid_repeat_tile_not_used(sisl_files, sisl_tmp):
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.TSHS'))
    HS = si.read_hamiltonian()
    for i in range(HS.no):
        HS._csr._extend_empty(i, 3 + i % 3)
    HSt = HS.tile(3, 2).tile(3, 0).tile(3, 1)
    HSr = HS.repeat(3, 2).repeat(3, 0).repeat(3, 1)
    assert np.allclose(HSr.eigh(), HSt.eigh())


def test_tshs_soc_pt2_xx(sisl_files, sisl_tmp):
    fdf = sisl.get_sile(sisl_files(_dir, 'SOC_Pt2_xx.fdf'), base=sisl_files(_dir))
    HS1 = fdf.read_hamiltonian()
    f = sisl_tmp('tmp.TSHS', _dir)
    HS1.write(f)
    si = sisl.get_sile(f)
    HS2 = si.read_hamiltonian()
    assert HS1._csr.spsame(HS2._csr)
    HS1.finalize()
    HS2.finalize()
    assert np.allclose(HS1._csr._D, HS2._csr._D)


def test_tshs_soc_pt2_xx_pdos(sisl_files):
    fdf = sisl.get_sile(sisl_files(_dir, 'SOC_Pt2_xx.fdf'), base=sisl_files(_dir))
    sc = fdf.read_supercell(order='TSHS')
    HS = fdf.read_hamiltonian()
    assert np.allclose(sc.cell, HS.geometry.sc.cell)
    HS.eigenstate().PDOS(np.linspace(-2, 2, 400))


def test_tshs_warn(sisl_files):
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


def test_tshs_error(sisl_files):
    # reading with a wrong geometry
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.TSHS'))

    # check number of orbitals
    geom = si.read_geometry()
    geom = sisl.Geometry(np.random.rand(geom.na + 1, 3))
    with pytest.raises(sisl.SileError):
        si.read_hamiltonian(geometry=geom)


def test_tshs_si_pdos_kgrid_overlap(sisl_files):
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.TSHS'))
    HS = si.read_hamiltonian()
    S = si.read_overlap()
    assert HS._csr.spsame(S._csr)
    HS.finalize()
    S.finalize()
    assert np.allclose(HS._csr._D[:, HS.S_idx], S._csr._D[:, 0])


def test_tshs_spin_orbit(sisl_tmp):
    H1 = sisl.Hamiltonian(sisl.geom.graphene(), spin=sisl.Spin('SO'))
    H1.construct(([0.1, 1.44],
                  [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                   [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]))

    f1 = sisl_tmp('tmp1.TSHS', _dir)
    f2 = sisl_tmp('tmp2.TSHS', _dir)
    H1.write(f1)
    H1.finalize()
    H2 = sisl.get_sile(f1).read_hamiltonian()
    H2.write(f2)
    H3 = sisl.get_sile(f2).read_hamiltonian()
    assert H1._csr.spsame(H2._csr)
    assert np.allclose(H1._csr._D, H2._csr._D)
    assert H1._csr.spsame(H3._csr)
    assert np.allclose(H1._csr._D, H3._csr._D)


def test_tshs_spin_orbit_tshs2nc2tshs(sisl_tmp):
    H1 = sisl.Hamiltonian(sisl.geom.graphene(), spin=sisl.Spin('SO'))
    H1.construct(([0.1, 1.44],
                  [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                   [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]))

    fdf_file = sisl_tmp('RUN.fdf', _dir)
    f1 = sisl_tmp('tmp1.TSHS', _dir)
    f2 = sisl_tmp('tmp1.nc', _dir)
    H1.write(f1)
    H1.finalize()
    H2 = sisl.get_sile(f1).read_hamiltonian()
    H2.write(f2)
    H3 = sisl.get_sile(f2).read_hamiltonian()
    open(fdf_file, 'w').writelines([
        "SystemLabel tmp1"
    ])
    fdf = sisl.get_sile(fdf_file)
    assert np.allclose(fdf.read_supercell(order='nc').cell,
                       fdf.read_supercell(order='TSHS').cell)
    assert H1._csr.spsame(H2._csr)
    assert np.allclose(H1._csr._D, H2._csr._D)
    assert H1._csr.spsame(H3._csr)
    assert np.allclose(H1._csr._D, H3._csr._D)
