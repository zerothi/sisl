# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pytest
import sys
import os.path as osp
import numpy as np
import os
from pathlib import Path

from sisl.io import *
from sisl.io.siesta.binaries import _gfSileSiesta
from sisl.io.tbtrans._cdf import *
from sisl.io.vasp import chgSileVASP
from sisl import Lattice, Geometry, GeometryCollection, Grid, Hamiltonian
from sisl import DensityMatrix, EnergyDensityMatrix


pytestmark = pytest.mark.io
_dir = osp.join("sisl", "io")


gs = get_sile
gsc = get_sile_class


def _my_intersect(a, b):
    a = get_siles(a)
    b = set(get_siles(b))
    return [k for k in a if k in b]


def _fnames(base, variants):
    files = [base + "." + v if len(v) > 0 else base for v in variants]
    files.append(Path(files[0]))
    return files


def test_get_sile1():
    cls = gsc("test.xyz")
    assert issubclass(cls, xyzSile)

    cls = gsc("test.regardless{xyz}")
    assert issubclass(cls, xyzSile)

    cls = gsc("test.fdf{xyz}")
    assert issubclass(cls, xyzSile)

    cls = gsc(Path("test.fdf{xyz}"))
    assert issubclass(cls, xyzSile)

    cls = gsc("test.xyz{fdf}")
    assert issubclass(cls, fdfSileSiesta)
    cls = gsc("test.cube{fdf}")
    assert issubclass(cls, fdfSileSiesta)


def test_get_sile2():
    # requesting non implemented files
    with pytest.raises(NotImplementedError):
        gsc("test.this_file_does_not_exist")


class TestObject:

    def test_siesta_sources(self):
        pytest.importorskip("sisl.io.siesta._siesta")

    def test_direct_path_instantiation(self, sisl_tmp):
        fp = Path(sisl_tmp("shouldnotexist.1234567", _dir))
        if fp.exists():
            os.remove(str(fp))
        for Sile in get_siles():
            if issubclass(Sile, SileCDF):
                sile = Sile(fp, _open=False)
            else:
                sile = Sile(fp)
            assert isinstance(sile, Sile)
            assert not fp.exists()

    def test_direct_string_instantiation(self, sisl_tmp):
        fp = sisl_tmp("shouldnotexist.1234567", _dir)
        if os.path.exists(fp):
            os.remove(fp)
        for Sile in get_siles():
            if issubclass(Sile, SileCDF):
                sile = Sile(fp, _open=False)
            else:
                sile = Sile(fp)
            assert isinstance(sile, Sile)
            assert not os.path.exists(fp)

    @pytest.mark.parametrize("sile", _fnames("test", ["cube", "CUBE", "cube.gz", "CUBE.gz"]))
    def test_cube(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, cubeSile]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames("test", ["ascii", "ascii.gz", "ascii.gz"]))
    def test_bigdft_ascii(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, SileBigDFT, asciiSileBigDFT]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames("test", ["gout", "gout.gz"]))
    def test_gulp_gout(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, SileGULP, gotSileGULP]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames("test", ["REF", "REF.gz"]))
    def test_scaleup_REF(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, SileScaleUp, refSileScaleUp]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames("test", ["restart", "restart.gz"]))
    def test_scaleup_restart(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, SileScaleUp, restartSileScaleUp]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames("test", ["rham", "rham.gz"]))
    def test_scaleup_rham(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, SileScaleUp, rhamSileScaleUp]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames("test", ["fdf", "fdf.gz", "FDF.gz", "FDF"]))
    def test_siesta_fdf(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, SileSiesta, fdfSileSiesta]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames("test", ["out", "out.gz"]))
    def test_siesta_out(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, SileSiesta, outSileSiesta]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames("test", ["nc"]))
    def test_siesta_nc(self, sile):
        s = gs(sile, _open=False)
        for obj in [BaseSile, SileCDF, SileCDFSiesta, ncSileSiesta]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames("test", ["grid.nc"]))
    def test_siesta_grid_nc(self, sile):
        sile = gs(sile, _open=False)
        for obj in [BaseSile, SileCDF, SileCDFSiesta, gridncSileSiesta]:
            assert isinstance(sile, obj)

    @pytest.mark.parametrize("sile", _fnames("test", ["XV", "XV.gz"]))
    def test_siesta_xv(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, SileSiesta, xvSileSiesta]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames("test", ["XV", "XV.gz"]))
    def test_siesta_xv_base(self, sile):
        s = gs(sile, cls=SileSiesta)
        for obj in [BaseSile, Sile, SileSiesta, xvSileSiesta]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames("test", ["PDOS.xml", "PDOS.xml.gz"]))
    def test_siesta_pdos_xml(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, SileSiesta, pdosSileSiesta]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames("test", ["ham", "HAM", "HAM.gz"]))
    def test_ham(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, hamiltonianSile]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames("test", ["TBT.nc"]))
    def test_tbtrans_nc(self, sile):
        s = gs(sile, _open=False)
        for obj in [BaseSile, SileCDF, SileCDFTBtrans, tbtncSileTBtrans]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames("test", ["PHT.nc"]))
    def test_phtrans_nc(self, sile):
        s = gs(sile, _open=False)
        for obj in [BaseSile, SileCDF, SileCDFTBtrans, tbtncSileTBtrans, phtncSilePHtrans]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames("CONTCAR", ["", "gz"]))
    def test_vasp_contcar(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, SileVASP, carSileVASP]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames("POSCAR", ["", "gz"]))
    def test_vasp_poscar(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, SileVASP, carSileVASP]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames("test", ["xyz", "XYZ", "xyz.gz", "XYZ.gz"]))
    def test_xyz(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, xyzSile]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames("test", ["molf", "MOLF", "molf.gz", "MOLF.gz"]))
    def test_molf(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, moldenSile]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames("test", ["xsf", "XSF", "xsf.gz", "XSF.gz"]))
    def test_xsf(self, sile):
        s = gs(sile)
        for obj in [BaseSile, Sile, xsfSile]:
            assert isinstance(s, obj)

    @pytest.mark.parametrize("sile", _fnames("test", ["win"]))
    def test_wannier90_seed(self, sile):
        sile = gs(sile, cls=SileWannier90)
        for obj in [BaseSile, Sile, SileWannier90, winSileWannier90]:
            assert isinstance(sile, obj)

    def test_write(self, sisl_tmp, sisl_system):
        G = sisl_system.g.rotate(-30, sisl_system.g.cell[2, :], what="xyz+abc")
        G.set_nsc([1, 1, 1])
        f = sisl_tmp("test_write", _dir)
        for sile in get_siles(["write_geometry"]):
            # It is not yet an instance, hence issubclass
            if issubclass(sile, (hamiltonianSile, _ncSileTBtrans, deltancSileTBtrans)):
                continue
            # Write
            sile(f, mode="w").write_geometry(G)

    @pytest.mark.parametrize("sile", _my_intersect(["read_lattice"], ["write_lattice"]))
    def test_read_write_lattice(self, sisl_tmp, sisl_system, sile):
        L = sisl_system.g.rotate(-30, sisl_system.g.cell[2, :], what="xyz+abc").lattice
        L.set_nsc([1, 1, 1])
        f = sisl_tmp("test_read_write_geom.win", _dir)
        # These files does not store the atomic species
        if issubclass(sile, (_ncSileTBtrans, deltancSileTBtrans)):
            return
        if sys.platform.startswith("win") and issubclass(sile, chgSileVASP):
            pytest.xfail("Windows reading/writing supercell fails for some unknown reason")

        # Write
        sile(f, mode="w").write_lattice(L)
        # Read 1
        try:
            with sile(f, mode='r') as s:
                l = s.read_lattice()
            assert l.equal(L, tol=1e-3) # pdb files have 8.3 for atomic coordinates
        except UnicodeDecodeError as e:
            pass
        # Read 2
        try:
            with sile(f, mode='r') as s:
                l = Lattice.read(s)
            assert l.equal(L, tol=1e-3)
        except UnicodeDecodeError as e:
            pass

    @pytest.mark.parametrize("sile", _my_intersect(["read_lattice"], ["write_lattice"]))
    def test_read_write_supercell(self, sisl_tmp, sisl_system, sile):
        L = sisl_system.g.rotate(-30, sisl_system.g.cell[2, :], what="xyz+abc").lattice
        L.set_nsc([1, 1, 1])
        f = sisl_tmp("test_read_write_geom.win", _dir)
        # These files does not store the atomic species
        if issubclass(sile, (_ncSileTBtrans, deltancSileTBtrans)):
            return
        if sys.platform.startswith("win") and issubclass(sile, chgSileVASP):
            pytest.xfail("Windows reading/writing supercell fails for some unknown reason")

        # Write
        sile(f, mode="w").write_supercell(L)
        # Read 1
        try:
            with sile(f, mode='r') as s:
                l = s.read_supercell()
            assert l.equal(L, tol=1e-3) # pdb files have 8.3 for atomic coordinates
        except UnicodeDecodeError as e:
            pass

    @pytest.mark.parametrize("sile", _my_intersect(["read_geometry"], ["write_geometry"]))
    def test_read_write_geometry(self, sisl_tmp, sisl_system, sile):
        G = sisl_system.g.rotate(-30, sisl_system.g.cell[2, :], what="xyz+abc")
        G.set_nsc([1, 1, 1])
        f = sisl_tmp("test_read_write_geom.win", _dir)
        # These files does not store the atomic species
        if issubclass(sile, (_ncSileTBtrans, deltancSileTBtrans)):
            return
        if sys.platform.startswith("win") and issubclass(sile, chgSileVASP):
            pytest.xfail("Windows reading/writing supercell fails for some unknown reason")

        # Write
        sile(f, mode="w").write_geometry(G)
        # Read 1
        try:
            with sile(f, mode='r') as s:
                g = s.read_geometry()
                if isinstance(g, GeometryCollection):
                    g = g[0]
            assert g.equal(G, R=False, tol=1e-3) # pdb files have 8.3 for atomic coordinates
        except UnicodeDecodeError as e:
            pass
        # Read 2
        try:
            with sile(f, mode='r') as s:
                g = Geometry.read(s)
                if isinstance(g, GeometryCollection):
                    g = g[0]
            assert g.equal(G, R=False, tol=1e-3)
        except UnicodeDecodeError as e:
            pass

    @pytest.mark.parametrize("sile", _my_intersect(["read_hamiltonian"], ["write_hamiltonian"]))
    def test_read_write_hamiltonian(self, sisl_tmp, sisl_system, sile):
        if issubclass(sile, _gfSileSiesta):
            return

        G = sisl_system.g.rotate(-30, sisl_system.g.cell[2, :], what="xyz+abc")
        H = Hamiltonian(G)
        H.construct([[0.1, 1.45], [0.1, -2.7]])
        f = sisl_tmp("test_read_write_hamiltonian.win", _dir)
        # Write
        with sile(f, mode='w') as s:
            s.write_hamiltonian(H)
        # Read 1
        try:
            with sile(f, mode='r') as s:
                h = s.read_hamiltonian()
            assert H.spsame(h)
        except UnicodeDecodeError as e:
            pass
        # Read 2
        try:
            with sile(f, mode='r') as s:
                h = Hamiltonian.read(s)
            assert H.spsame(h)
        except UnicodeDecodeError as e:
            pass

    @pytest.mark.parametrize("sile", _my_intersect(["read_density_matrix"], ["write_density_matrix"]))
    def test_read_write_density_matrix(self, sisl_tmp, sisl_system, sile):
        G = sisl_system.g.rotate(-30, sisl_system.g.cell[2, :], what="xyz+abc")
        DM = DensityMatrix(G, orthogonal=True)
        DM.construct([[0.1, 1.45], [0.1, -2.7]])
        f = sisl_tmp("test_read_write_density_matrix.win", _dir)
        # Write
        with sile(f, mode='w') as s:
            s.write_density_matrix(DM)
        # Read 1
        try:
            with sile(f, mode='r') as s:
                dm = s.read_density_matrix(geometry=DM.geometry)
            assert DM.spsame(dm)
        except UnicodeDecodeError as e:
            pass
        # Read 2
        try:
            with sile(f, mode='r') as s:
                dm = DensityMatrix.read(s, geometry=DM.geometry)
            assert DM.spsame(dm)
        except UnicodeDecodeError as e:
            pass

    @pytest.mark.parametrize("sile", _my_intersect(["read_energy_density_matrix"], ["write_energy_density_matrix"]))
    def test_read_write_energy_density_matrix(self, sisl_tmp, sisl_system, sile):
        G = sisl_system.g.rotate(-30, sisl_system.g.cell[2, :], what="xyz+abc")
        EDM = EnergyDensityMatrix(G, orthogonal=True)
        EDM.construct([[0.1, 1.45], [0.1, -2.7]])
        f = sisl_tmp("test_read_write_energy_density_matrix.win", _dir)
        # Write
        with sile(f, mode='w') as s:
            s.write_energy_density_matrix(EDM)
        # Read 1
        try:
            with sile(f, mode='r') as s:
                edm = s.read_energy_density_matrix(geometry=EDM.geometry)
            assert EDM.spsame(edm)
        except UnicodeDecodeError as e:
            pass
        # Read 2
        try:
            with sile(f, mode='r') as s:
                edm = EnergyDensityMatrix.read(s, geometry=EDM.geometry)
            assert EDM.spsame(edm)
        except UnicodeDecodeError as e:
            pass

    @pytest.mark.parametrize("sile", _my_intersect(["read_hamiltonian"], ["write_hamiltonian"]))
    def test_read_write_hamiltonian_overlap(self, sisl_tmp, sisl_system, sile):
        if issubclass(sile, _gfSileSiesta):
            return

        G = sisl_system.g.rotate(-30, sisl_system.g.cell[2, :], what="xyz+abc")
        H = Hamiltonian(G, orthogonal=False)
        H.construct([[0.1, 1.45], [(0.1, 1), (-2.7, 0.1)]])
        f = sisl_tmp("test_read_write_hamiltonian_overlap.win", _dir)
        # Write
        with sile(f, mode='w') as s:
            s.write_hamiltonian(H)
        # Read 1
        try:
            with sile(f, mode='r') as s:
                h = s.read_hamiltonian()
            assert H.spsame(h)
        except UnicodeDecodeError as e:
            pass
        # Read 2
        try:
            with sile(f, mode='r') as s:
                h = Hamiltonian.read(s)
            assert H.spsame(h)
        except UnicodeDecodeError as e:
            pass

    @pytest.mark.filterwarnings("ignore", message="*gridncSileSiesta.read_grid cannot determine")
    @pytest.mark.parametrize("sile", _my_intersect(["read_grid"], ["write_grid"]))
    def test_read_write_grid(self, sisl_tmp, sisl_system, sile):
        g = sisl_system.g.rotate(-30, sisl_system.g.cell[2, :], what="xyz+abc")
        G = Grid([10, 11, 12])
        G[:, :, :] = np.random.rand(10, 11, 12)

        f = sisl_tmp("test_read_write_grid.win", _dir)
        # Write
        try:
            with sile(f, mode="w") as s:
                s.write_grid(G)
        except SileError:
            with sile(f, mode="wb") as s:
                s.write_grid(G)
        # Read 1
        try:
            with sile(f, mode="r") as s:
                g = s.read_grid()
            assert np.allclose(g.grid, G.grid, atol=1e-5)
        except UnicodeDecodeError as e:
            pass
        # Read 2
        try:
            with sile(f, mode='r') as s:
                g = Grid.read(s)
            assert np.allclose(g.grid, G.grid, atol=1e-5)
        except UnicodeDecodeError as e:
            pass

    def test_arg_parser1(self, sisl_tmp):
        f = sisl_tmp("something", _dir)
        for sile in get_siles(["ArgumentParser"]):
            try:
                sile(f).ArgumentParser()
            except Exception:
                pass

    def test_arg_parser2(self, sisl_tmp):
        f = sisl_tmp("something", _dir)
        for sile in get_siles(["ArgumentParser_out"]):
            try:
                sile(f).ArgumentParser()
            except Exception:
                pass


def test_buffer_cls():
    # Check that buffer_cls argument for subclassing classes
    # works correctly
    class customBuffer(BufferSile):
        pass

    class tmpSile(xyzSile, buffer_cls=customBuffer):
        pass

    assert tmpSile._buffer_cls.__name__ == "customBuffer"

    with pytest.raises(TypeError):
        class tmpSile(xyzSile, buffer_cls=object):
            pass
