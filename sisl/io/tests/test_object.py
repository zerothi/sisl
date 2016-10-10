from __future__ import print_function, division

from nose.tools import *

import numpy as np

from tempfile import mkstemp
from sisl.io import *
from sisl import Geometry

import common as tc

gs = get_sile

class TestObject(object):
    # Base test class for MaskedArrays.
    setUp = tc.setUp
    tearDown = tc.tearDown

    def test_cube(self):
        sile1 = gs('test.cube')
        sile2 = gs('test.CUBE')
        for obj in [BaseSile, Sile, CUBESile]:
            assert_true(isinstance(sile1, obj))
            assert_true(isinstance(sile2, obj))

    def test_cube_gz(self):
        sile1 = gs('test.cube.gz')
        sile2 = gs('test.CUBE.gz')
        for obj in [BaseSile, Sile, CUBESile]:
            assert_true(isinstance(sile1, obj))
            assert_true(isinstance(sile2, obj))

    def test_bigdft_ascii(self):
        sile = gs('test.ascii')
        for obj in [BaseSile, Sile, SileBigDFT, ASCIISileBigDFT]:
            assert_true(isinstance(sile, obj))
            assert_true(isinstance(sile, obj))

    def test_bigdft_ascii_gz(self):
        sile = gs('test.ascii.gz')
        for obj in [BaseSile, Sile, SileBigDFT, ASCIISileBigDFT]:
            assert_true(isinstance(sile, obj))
            assert_true(isinstance(sile, obj))

    def test_fdf(self):
        sile1 = gs('test.fdf')
        sile2 = gs('test.FDF')
        for obj in [BaseSile, Sile, SileSiesta, fdfSileSiesta]:
            assert_true(isinstance(sile1, obj))
            assert_true(isinstance(sile2, obj))

    def test_fdf_gz(self):
        sile1 = gs('test.fdf.gz')
        sile2 = gs('test.FDF.gz')
        for obj in [BaseSile, Sile, SileSiesta, fdfSileSiesta]:
            assert_true(isinstance(sile1, obj))
            assert_true(isinstance(sile2, obj))

    def test_gout(self):
        sile = gs('test.gout')
        for obj in [BaseSile, Sile, SileGULP, gotSileGULP]:
            assert_true(isinstance(sile, obj))

    def test_gout_gz(self):
        sile = gs('test.gout.gz')
        for obj in [BaseSile, Sile, SileGULP, gotSileGULP]:
            assert_true(isinstance(sile, obj))

    def test_out(self):
        sile = gs('test.out')
        for obj in [BaseSile, Sile, SileSiesta, outSileSiesta]:
            assert_true(isinstance(sile, obj))

    def test_out_gz(self):
        sile = gs('test.out.gz')
        for obj in [BaseSile, Sile, SileSiesta, outSileSiesta]:
            assert_true(isinstance(sile, obj))

    def test_nc(self):
        sile = gs('test.nc', _open=False)
        for obj in [BaseSile, SileCDF, SileCDFSIESTA, ncSileSiesta]:
            assert_true(isinstance(gs('test.nc', _open=False), obj))

    def test_grid_nc(self):
        sile = gs('test.grid.nc', _open=False)
        for obj in [BaseSile, SileCDF, SileCDFSIESTA, gridncSileSiesta]:
            assert_true(isinstance(sile, obj))

    def test_ham(self):
        sile1 = gs('test.ham')
        sile2 = gs('test.HAM')
        for obj in [BaseSile, Sile, HamiltonianSile]:
            assert_true(isinstance(sile1, obj))
            assert_true(isinstance(sile2, obj))

    def test_ham_gz(self):
        sile1 = gs('test.ham.gz')
        sile2 = gs('test.HAM.gz')
        for obj in [BaseSile, Sile, HamiltonianSile]:
            assert_true(isinstance(sile1, obj))
            assert_true(isinstance(sile2, obj))

    def test_tbtrans(self):
        sile = gs('test.TBT.nc', _open=False)
        for obj in [BaseSile, SileCDF, SileCDFSIESTA, tbtncSileSiesta]:
            assert_true(isinstance(sile, obj))

    def test_phtrans(self):
        sile = gs('test.PHT.nc', _open=False)
        for obj in [BaseSile, SileCDF, SileCDFSIESTA, tbtncSileSiesta, phtncSileSiesta]:
            assert_true(isinstance(sile, obj))

    def test_vasp_contcar(self):
        sile = gs('CONTCAR')
        for obj in [BaseSile, Sile, SileVASP, CARSileVASP, CONTCARSileVASP]:
            assert_true(isinstance(sile, obj))

    def test_vasp_poscar(self):
        sile = gs('POSCAR')
        for obj in [BaseSile, Sile, SileVASP, CARSileVASP, POSCARSileVASP]:
            assert_true(isinstance(sile, obj))

    def test_vasp_contcar_gz(self):
        sile = gs('CONTCAR.gz')
        for obj in [BaseSile, Sile, SileVASP, CARSileVASP, CONTCARSileVASP]:
            assert_true(isinstance(sile, obj))

    def test_vasp_poscar_gz(self):
        sile = gs('POSCAR.gz')
        for obj in [BaseSile, Sile, SileVASP, CARSileVASP, POSCARSileVASP]:
            assert_true(isinstance(sile, obj))

    def test_xyz(self):
        sile1 = gs('test.xyz')
        sile2 = gs('test.XYZ')
        for obj in [BaseSile, Sile, XYZSile]:
            assert_true(isinstance(sile1, obj))
            assert_true(isinstance(sile2, obj))

    def test_xyz_gz(self):
        sile1 = gs('test.xyz.gz')
        sile2 = gs('test.XYZ.gz')
        for obj in [BaseSile, Sile, XYZSile]:
            assert_true(isinstance(sile1, obj))
            assert_true(isinstance(sile2, obj))

    def test_xsf(self):
        sile1 = gs('test.xsf')
        sile2 = gs('test.XSF')
        for obj in [BaseSile, Sile, XSFSile]:
            assert_true(isinstance(sile1, obj))
            assert_true(isinstance(sile2, obj))

    def test_xsf_gz(self):
        sile1 = gs('test.xsf.gz')
        sile2 = gs('test.XSF.gz')
        for obj in [BaseSile, Sile, XSFSile]:
            assert_true(isinstance(sile1, obj))
            assert_true(isinstance(sile2, obj))

    def test_xv(self):
        sile = gs('test.XV')
        for obj in [BaseSile, Sile, SileSiesta, XVSileSiesta]:
            assert_true(isinstance(sile, obj))
            
    def test_xv_gz(self):
        sile = gs('test.XV.gz')
        for obj in [BaseSile, Sile, SileSiesta, XVSileSiesta]:
            assert_true(isinstance(sile, obj))

    def test_siesta(self):
        sile = gs('test.XV', cls=SileSiesta)
        for obj in [BaseSile, Sile, SileSiesta, XVSileSiesta]:
            assert_true(isinstance(sile, obj))

    def test_wannier90_seed(self):
        sile = gs('test.win', cls=SileW90)
        for obj in [BaseSile, Sile, SileW90, winSileW90]:
            assert_true(isinstance(sile, obj))


    def test_read_write(self):
        G = self.g.rotatec(-30)
        G.set_nsc([1,1,1])
        f = mkstemp(dir=self.d)[1]
        read_geom = get_siles(['read_geom'])
        for sile in get_siles(['write_geom']):
            if not sile in read_geom:
                continue
            #if sile is CUBESile:
            #    continue
            if sile is HamiltonianSile:
                continue
            if sile in [tbtncSileSiesta, phtncSileSiesta, dHncSileSiesta]:
                continue
            # Write
            s = sile(f, mode='w').write_geom(G)
            # Read
            g = sile(f).read_geom()
            # Easy fix to run the ArgumentParser code...
            if hasattr(g, 'ArgumentParser'):
                g.ArgumentParser()
            # Assert
            assert_equal(g, G, sile)

