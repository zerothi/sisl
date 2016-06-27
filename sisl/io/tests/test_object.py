from __future__ import print_function, division

from nose.tools import *

from sisl.io import *

gs = get_sile

class TestObject(object):
    # Base test class for MaskedArrays.

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
        for obj in [BaseSile, Sile, SileBigDFT, BigDFTASCIISile]:
            assert_true(isinstance(sile, obj))
            assert_true(isinstance(sile, obj))

    def test_bigdft_ascii_gz(self):
        sile = gs('test.ascii.gz')
        for obj in [BaseSile, Sile, SileBigDFT, BigDFTASCIISile]:
            assert_true(isinstance(sile, obj))
            assert_true(isinstance(sile, obj))

    def test_fdf(self):
        sile1 = gs('test.fdf')
        sile2 = gs('test.FDF')
        for obj in [BaseSile, Sile, SileSIESTA, FDFSile]:
            assert_true(isinstance(sile1, obj))
            assert_true(isinstance(sile2, obj))

    def test_fdf_gz(self):
        sile1 = gs('test.fdf.gz')
        sile2 = gs('test.FDF.gz')
        for obj in [BaseSile, Sile, SileSIESTA, FDFSile]:
            assert_true(isinstance(sile1, obj))
            assert_true(isinstance(sile2, obj))

    def test_gout(self):
        sile = gs('test.gout')
        for obj in [BaseSile, Sile, SileGULP, GULPgoutSile]:
            assert_true(isinstance(sile, obj))

    def test_gout_gz(self):
        sile = gs('test.gout.gz')
        for obj in [BaseSile, Sile, SileGULP, GULPgoutSile]:
            assert_true(isinstance(sile, obj))

    def test_nc(self):
        sile = gs('test.nc')
        for obj in [BaseSile, NCSile, NCSileSIESTA, SIESTASile]:
            assert_true(isinstance(gs('test.nc'), obj))

    def test_grid_nc(self):
        sile = gs('test.grid.nc')
        for obj in [BaseSile, NCSile, NCSileSIESTA, SIESTAGridSile]:
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
        sile = gs('test.TBT.nc')
        for obj in [BaseSile, NCSile, NCSileSIESTA, TBtransSile]:
            assert_true(isinstance(sile, obj))

    def test_phtrans(self):
        sile = gs('test.PHT.nc')
        for obj in [BaseSile, NCSile, NCSileSIESTA, PHtransSile]:
            assert_true(isinstance(sile, obj))

    def test_vasp_contcar(self):
        sile = gs('CONTCAR')
        for obj in [BaseSile, Sile, SileVASP, CARSile, CONTCARSile]:
            assert_true(isinstance(sile, obj))

    def test_vasp_poscar(self):
        sile = gs('POSCAR')
        for obj in [BaseSile, Sile, SileVASP, CARSile, POSCARSile]:
            assert_true(isinstance(sile, obj))

    def test_vasp_contcar_gz(self):
        sile = gs('CONTCAR.gz')
        for obj in [BaseSile, Sile, SileVASP, CARSile, CONTCARSile]:
            assert_true(isinstance(sile, obj))

    def test_vasp_poscar_gz(self):
        sile = gs('POSCAR.gz')
        for obj in [BaseSile, Sile, SileVASP, CARSile, POSCARSile]:
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

    def test_xv(self):
        sile = gs('test.XV')
        for obj in [BaseSile, Sile, SileSIESTA, XVSile]:
            assert_true(isinstance(sile, obj))
            
    def test_xv_gz(self):
        sile = gs('test.XV.gz')
        for obj in [BaseSile, Sile, SileSIESTA, XVSile]:
            assert_true(isinstance(sile, obj))
