from __future__ import print_function, division

import pytest

import numpy as np
import os

from tempfile import mkstemp
from sisl.io import *
from sisl import Geometry, Grid, Hamiltonian

import common as tc

_C = type('Temporary', (object, ), {})


def setup_module(module):
    tc.setup(module._C)


def teardown_module(module):
    tc.teardown(module._C)


gs = get_sile


def stdoutfile(f):
    with open(f, 'r') as fh:
        for line in fh:
            print(line.replace('\n', ''))


def _my_intersect(a, b):
    return list(set(get_siles(a)).intersection(get_siles(b)))


@pytest.mark.io
class TestObject(object):

    def test_siesta_sources(self):
        pytest.importorskip("sisl.io.siesta._siesta")

    def test_cube(self):
        sile1 = gs('test.cube')
        sile2 = gs('test.CUBE')
        for obj in [BaseSile, Sile, CUBESile]:
            assert isinstance(sile1, obj)
            assert isinstance(sile2, obj)

    def test_cube_gz(self):
        sile1 = gs('test.cube.gz')
        sile2 = gs('test.CUBE.gz')
        for obj in [BaseSile, Sile, CUBESile]:
            assert isinstance(sile1, obj)
            assert isinstance(sile2, obj)

    def test_bigdft_ascii(self):
        sile = gs('test.ascii')
        for obj in [BaseSile, Sile, SileBigDFT, ASCIISileBigDFT]:
            assert isinstance(sile, obj)
            assert isinstance(sile, obj)

    def test_bigdft_ascii_gz(self):
        sile = gs('test.ascii.gz')
        for obj in [BaseSile, Sile, SileBigDFT, ASCIISileBigDFT]:
            assert isinstance(sile, obj)
            assert isinstance(sile, obj)

    def test_fdf(self):
        sile1 = gs('test.fdf')
        sile2 = gs('test.FDF')
        for obj in [BaseSile, Sile, SileSiesta, fdfSileSiesta]:
            assert isinstance(sile1, obj)
            assert isinstance(sile2, obj)

    def test_fdf_gz(self):
        sile1 = gs('test.fdf.gz')
        sile2 = gs('test.FDF.gz')
        for obj in [BaseSile, Sile, SileSiesta, fdfSileSiesta]:
            assert isinstance(sile1, obj)
            assert isinstance(sile2, obj)

    def test_gout(self):
        sile = gs('test.gout')
        for obj in [BaseSile, Sile, SileGULP, gotSileGULP]:
            assert isinstance(sile, obj)

    def test_gout_gz(self):
        sile = gs('test.gout.gz')
        for obj in [BaseSile, Sile, SileGULP, gotSileGULP]:
            assert isinstance(sile, obj)

    def test_REF(self):
        for end in ['REF', 'REF.gz']:
            sile = gs('test.' + end)
            for obj in [BaseSile, Sile, SileScaleUp, REFSileScaleUp]:
                assert isinstance(sile, obj)

    def test_restart(self):
        for end in ['restart', 'restart.gz']:
            sile = gs('test.' + end)
            for obj in [BaseSile, Sile, SileScaleUp, restartSileScaleUp]:
                assert isinstance(sile, obj)

    def test_rham(self):
        for end in ['rham', 'rham.gz']:
            sile = gs('test.' + end)
            for obj in [BaseSile, Sile, SileScaleUp, rhamSileScaleUp]:
                assert isinstance(sile, obj)

    def test_out(self):
        for end in ['out', 'out.gz']:
            sile = gs('test.' + end)
            for obj in [BaseSile, Sile, SileSiesta, outSileSiesta]:
                assert isinstance(sile, obj)

    def test_nc(self):
        sile = gs('test.nc', _open=False)
        for obj in [BaseSile, SileCDF, SileCDFSiesta, ncSileSiesta]:
            assert isinstance(gs('test.nc', _open=False), obj)

    def test_grid_nc(self):
        sile = gs('test.grid.nc', _open=False)
        for obj in [BaseSile, SileCDF, SileCDFSiesta, gridncSileSiesta]:
            assert isinstance(sile, obj)

    def test_ham(self):
        sile1 = gs('test.ham')
        sile2 = gs('test.HAM')
        for obj in [BaseSile, Sile, HamiltonianSile]:
            assert isinstance(sile1, obj)
            assert isinstance(sile2, obj)

    def test_ham_gz(self):
        sile1 = gs('test.ham.gz')
        sile2 = gs('test.HAM.gz')
        for obj in [BaseSile, Sile, HamiltonianSile]:
            assert isinstance(sile1, obj)
            assert isinstance(sile2, obj)

    def test_tbtrans(self):
        sile = gs('test.TBT.nc', _open=False)
        for obj in [BaseSile, SileCDF, SileCDFTBtrans, tbtncSileTBtrans]:
            assert isinstance(sile, obj)

    def test_phtrans(self):
        sile = gs('test.PHT.nc', _open=False)
        for obj in [BaseSile, SileCDF, SileCDFTBtrans, tbtncSileTBtrans, phtncSileTBtrans]:
            assert isinstance(sile, obj)

    def test_vasp_contcar(self):
        sile = gs('CONTCAR')
        for obj in [BaseSile, Sile, SileVASP, CARSileVASP, CONTCARSileVASP]:
            assert isinstance(sile, obj)

    def test_vasp_poscar(self):
        sile = gs('POSCAR')
        for obj in [BaseSile, Sile, SileVASP, CARSileVASP, POSCARSileVASP]:
            assert isinstance(sile, obj)

    def test_vasp_contcar_gz(self):
        sile = gs('CONTCAR.gz')
        for obj in [BaseSile, Sile, SileVASP, CARSileVASP, CONTCARSileVASP]:
            assert isinstance(sile, obj)

    def test_vasp_poscar_gz(self):
        sile = gs('POSCAR.gz')
        for obj in [BaseSile, Sile, SileVASP, CARSileVASP, POSCARSileVASP]:
            assert isinstance(sile, obj)

    def test_xyz(self):
        sile1 = gs('test.xyz')
        sile2 = gs('test.XYZ')
        for obj in [BaseSile, Sile, XYZSile]:
            assert isinstance(sile1, obj)
            assert isinstance(sile2, obj)

    def test_xyz_gz(self):
        sile1 = gs('test.xyz.gz')
        sile2 = gs('test.XYZ.gz')
        for obj in [BaseSile, Sile, XYZSile]:
            assert isinstance(sile1, obj)
            assert isinstance(sile2, obj)

    def test_molf(self):
        sile1 = gs('test.molf')
        sile2 = gs('test.MOLF')
        for obj in [BaseSile, Sile, MoldenSile]:
            assert isinstance(sile1, obj)
            assert isinstance(sile2, obj)

    def test_molf_gz(self):
        sile1 = gs('test.molf.gz')
        sile2 = gs('test.MOLF.gz')
        for obj in [BaseSile, Sile, MoldenSile]:
            assert isinstance(sile1, obj)
            assert isinstance(sile2, obj)

    def test_xsf(self):
        sile1 = gs('test.xsf')
        sile2 = gs('test.XSF')
        for obj in [BaseSile, Sile, XSFSile]:
            assert isinstance(sile1, obj)
            assert isinstance(sile2, obj)

    def test_xsf_gz(self):
        sile1 = gs('test.xsf.gz')
        sile2 = gs('test.XSF.gz')
        for obj in [BaseSile, Sile, XSFSile]:
            assert isinstance(sile1, obj)
            assert isinstance(sile2, obj)

    def test_xv(self):
        sile = gs('test.XV')
        for obj in [BaseSile, Sile, SileSiesta, XVSileSiesta]:
            assert isinstance(sile, obj)

    def test_xv_gz(self):
        sile = gs('test.XV.gz')
        for obj in [BaseSile, Sile, SileSiesta, XVSileSiesta]:
            assert isinstance(sile, obj)

    def test_siesta(self):
        sile = gs('test.XV', cls=SileSiesta)
        for obj in [BaseSile, Sile, SileSiesta, XVSileSiesta]:
            assert isinstance(sile, obj)

    def test_wannier90_seed(self):
        sile = gs('test.win', cls=SileWannier90)
        for obj in [BaseSile, Sile, SileWannier90, winSileWannier90]:
            assert isinstance(sile, obj)

    def test_write(self):
        G = _C.g.rotatec(-30)
        G.set_nsc([1, 1, 1])
        f = mkstemp(dir=_C.d)[1]
        for sile in get_siles(['write_geometry']):
            # It is not yet an instance, hence issubclass
            if issubclass(sile, (HamiltonianSile, tbtncSileTBtrans, deltancSileTBtrans)):
                continue
            # Write
            sile(f, mode='w').write_geometry(G)

    @pytest.mark.parametrize("sile", _my_intersect(['read_geometry'], ['write_geometry']))
    def test_read_write_geom(self, sile):
        G = _C.g.rotatec(-30)
        G.set_nsc([1, 1, 1])
        f = mkstemp(dir=_C.d)[1] + '.win'
        # These files does not store the atomic species
        if issubclass(sile, (tbtncSileTBtrans, deltancSileTBtrans)):
            return
        # Write
        sile(f, mode='w').write_geometry(G)
        # Read 1
        try:
            g = sile(f, mode='r').read_geometry()
            assert g.equal(G, R=False)
        except UnicodeDecodeError as e:
            pass
        # Read 2
        try:
            g = Geometry.read(sile(f, mode='r'))
            assert g.equal(G, R=False)
        except UnicodeDecodeError as e:
            pass
        # Clean-up file
        os.remove(f)

    @pytest.mark.parametrize("sile", _my_intersect(['read_hamiltonian'], ['write_hamiltonian']))
    def test_read_write_hamiltonian(self, sile):
        G = _C.g.rotatec(-30)
        H = Hamiltonian(G)
        H.construct([[0.1, 1.45], [0.1, -2.7]])
        f = mkstemp(dir=_C.d)[1]
        # Write
        sile(f, mode='w').write_hamiltonian(H)
        # Read 1
        try:
            h = sile(f, mode='r').read_hamiltonian()
            assert H.spsame(h)
        except UnicodeDecodeError as e:
            pass
        # Read 2
        try:
            h = Hamiltonian.read(sile(f, mode='r'))
            assert H.spsame(h)
        except UnicodeDecodeError as e:
            pass
        # Clean-up file
        os.remove(f)

    @pytest.mark.parametrize("sile", _my_intersect(['read_grid'], ['write_grid']))
    def test_read_write_grid(self, sile):
        g = _C.g.rotatec(-30)
        G = Grid([10, 11, 12])
        G[:, :, :] = np.random.rand(10, 11, 12)

        f = mkstemp(dir=_C.d)[1]
        # Write
        try:
            sile(f, mode='w').write_grid(G)
        except SileError:
            sile(f, mode='wb').write_grid(G)
        # Read 1
        try:
            g = sile(f, mode='r').read_grid()
            assert np.allclose(g.grid, G.grid, atol=1e-5)
        except UnicodeDecodeError as e:
            pass
        # Read 2
        try:
            g = Grid.read(sile(f, mode='r'))
            assert np.allclose(g.grid, G.grid, atol=1e-5)
        except UnicodeDecodeError as e:
            pass
        # Clean-up file
        os.remove(f)

    def test_arg_parser1(self):
        f = mkstemp(dir=_C.d)[1]
        for sile in get_siles(['ArgumentParser']):
            try:
                sile(f).ArgumentParser()
            except:
                pass

    def test_arg_parser2(self):
        f = mkstemp(dir=_C.d)[1]
        for sile in get_siles(['ArgumentParser_out']):
            try:
                sile(f).ArgumentParser()
            except:
                pass
