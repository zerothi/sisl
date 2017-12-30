from __future__ import print_function, division

import pytest

import numpy as np
import os

from tempfile import mkstemp
from sisl.io import *
from sisl.io.tbtrans._cdf import *
from sisl import Geometry, Grid, Hamiltonian

import common as tc

_C = type('Temporary', (object, ), {})


def setup_module(module):
    tc.setup(module._C)


def teardown_module(module):
    tc.teardown(module._C)


gs = get_sile
gsc = get_sile_class


def stdoutfile(f):
    with open(f, 'r') as fh:
        for line in fh:
            print(line.replace('\n', ''))


def _my_intersect(a, b):
    return list(set(get_siles(a)).intersection(get_siles(b)))


def _fnames(base, variants):
    return [base + '.' + v if len(v) > 0 else base for v in variants]


@pytest.mark.io
def test_get_sile1():
    cls = gsc('test.xyz')
    assert issubclass(cls, XYZSile)

    cls = gsc('test.regardless{XYZ}')
    assert issubclass(cls, XYZSile)

    cls = gsc('test.fdf{XYZ}')
    assert issubclass(cls, XYZSile)

    cls = gsc('test.fdf{XYZ}')
    assert issubclass(cls, XYZSile)


@pytest.mark.io
@pytest.mark.xfail(raises=NotImplementedError)
def test_get_sile2():
    gsc('test.this_file_does_not_exist')


@pytest.mark.io
class TestObject(object):

    def test_siesta_sources(self):
        pytest.importorskip("sisl.io.siesta._siesta")

    @pytest.mark.parametrize("sile", _fnames('test', ['cube', 'CUBE', 'cube.gz', 'CUBE.gz']))
    def test_cube(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, CUBESile]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames('test', ['ascii', 'ascii.gz', 'ascii.gz']))
    def test_bigdft_ascii(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, SileBigDFT, ASCIISileBigDFT]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames('test', ['gout', 'gout.gz']))
    def test_gulp_gout(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, SileGULP, gotSileGULP]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames('test', ['REF', 'REF.gz']))
    def test_scaleup_REF(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, SileScaleUp, REFSileScaleUp]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames('test', ['restart', 'restart.gz']))
    def test_scaleup_restart(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, SileScaleUp, restartSileScaleUp]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames('test', ['rham', 'rham.gz']))
    def test_scaleup_rham(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, SileScaleUp, rhamSileScaleUp]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames('test', ['fdf', 'fdf.gz', 'FDF.gz', 'FDF']))
    def test_siesta_fdf(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, SileSiesta, fdfSileSiesta]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames('test', ['out', 'out.gz']))
    def test_siesta_out(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, SileSiesta, outSileSiesta]:
            assert isinstance(s, obj)

    def test_siesta_nc(self):
        s = gs('test.nc', _open=False)
        for obj in [BaseSile, SileCDF, SileCDFSiesta, ncSileSiesta]:
            assert isinstance(s, obj)

    def test_siesta_grid_nc(self):
        sile = gs('test.grid.nc', _open=False)
        for obj in [BaseSile, SileCDF, SileCDFSiesta, gridncSileSiesta]:
            assert isinstance(sile, obj)

    @pytest.mark.parametrize("sile", _fnames('test', ['XV', 'XV.gz']))
    def test_siesta_xv(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, SileSiesta, XVSileSiesta]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames('test', ['XV', 'XV.gz']))
    def test_siesta_xv_base(self, sile):
        s = gs(sile, cls=SileSiesta)
        for obj in [BaseSile, Sile, SileSiesta, XVSileSiesta]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames('test', ['PDOS.xml', 'pdos.xml', 'PDOS.xml.gz', 'pdos.xml.gz']))
    def test_siesta_pdos_xml(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, SileSiesta, pdosSileSiesta]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames('test', ['ham', 'HAM', 'HAM.gz']))
    def test_ham(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, HamiltonianSile]:
            assert isinstance(s, obj)

    def test_tbtrans_nc(self):
        s = gs('test.TBT.nc', _open=False)
        for obj in [BaseSile, SileCDF, SileCDFTBtrans, tbtncSileTBtrans]:
            assert isinstance(s, obj)

    def test_phtrans_nc(self):
        s = gs('test.PHT.nc', _open=False)
        for obj in [BaseSile, SileCDF, SileCDFTBtrans, tbtncSileTBtrans, phtncSileTBtrans]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames('CONTCAR', ['', 'gz']))
    def test_vasp_contcar(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, SileVASP, CARSileVASP, CONTCARSileVASP]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames('POSCAR', ['', 'gz']))
    def test_vasp_poscar(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, SileVASP, CARSileVASP, POSCARSileVASP]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames('test', ['xyz', 'XYZ', 'xyz.gz', 'XYZ.gz']))
    def test_xyz(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, XYZSile]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames('test', ['molf', 'MOLF', 'molf.gz', 'MOLF.gz']))
    def test_molf(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, MoldenSile]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames('test', ['xsf', 'XSF', 'xsf.gz', 'XSF.gz']))
    def test_xsf(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, XSFSile]:
            assert isinstance(s, obj)

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
            if issubclass(sile, (HamiltonianSile, _ncSileTBtrans, deltancSileTBtrans)):
                continue
            # Write
            sile(f, mode='w').write_geometry(G)

    @pytest.mark.parametrize("sile", _my_intersect(['read_geometry'], ['write_geometry']))
    def test_read_write_geom(self, sile):
        G = _C.g.rotatec(-30)
        G.set_nsc([1, 1, 1])
        f = mkstemp(dir=_C.d)[1] + '.win'
        # These files does not store the atomic species
        if issubclass(sile, (_ncSileTBtrans, deltancSileTBtrans)):
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

    @pytest.mark.parametrize("sile", _my_intersect(['read_hamiltonian'], ['write_hamiltonian']))
    def test_read_write_hamiltonian_overlap(self, sile):
        G = _C.g.rotatec(-30)
        H = Hamiltonian(G, orthogonal=False)
        H.construct([[0.1, 1.45], [(0.1, 1), (-2.7, 0.1)]])
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
