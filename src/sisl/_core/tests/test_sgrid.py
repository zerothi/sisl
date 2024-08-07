# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl import Atom, Geometry, Grid, Lattice, get_sile
from sisl._core.grid import sgrid


@pytest.fixture
def setup():
    class t:
        def __init__(self):
            bond = 1.42
            sq3h = 3.0**0.5 * 0.5
            self.lattice = Lattice(
                np.array(
                    [[1.5, sq3h, 0.0], [1.5, -sq3h, 0.0], [0.0, 0.0, 10.0]], np.float64
                )
                * bond,
                nsc=[3, 3, 1],
            )
            C = Atom(Z=6, R=[bond * 1.01] * 2)
            self.g = Geometry(
                np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float64) * bond,
                atoms=C,
                lattice=self.lattice,
            )
            self.grid = Grid(0.2, geometry=self.g)
            self.grid.grid[:, :, :] = np.random.rand(*self.grid.shape)

            self.mol = Geometry([[i, 0, 0] for i in range(10)], lattice=[50])

            self.grid_mol = Grid(0.2, geometry=self.mol)
            self.grid_mol.grid[:, :, :] = np.random.rand(*self.grid_mol.shape)

            def sg_g(**kwargs):
                kwargs["ret_grid"] = True
                if "grid" not in kwargs:
                    kwargs["grid"] = self.grid
                return sgrid(**kwargs)

            self.sg_g = sg_g

            def sg_mol(**kwargs):
                kwargs["ret_grid"] = True
                if "grid" not in kwargs:
                    kwargs["grid"] = self.grid_mol
                return sgrid(**kwargs)

            self.sg_mol = sg_mol

    return t()


@pytest.mark.sgrid
class TestsGrid:
    def test_help(self):
        with pytest.raises(SystemExit):
            sgrid(argv=["--help"])

    def test_version(self):
        sgrid(argv=["--version"])

    def test_cite(self):
        sgrid(argv=["--cite"])

    def test_average1(self, setup):
        g = setup.grid.copy()
        gavg = g.average(0)
        for avg in ["average x", "average 0"]:
            G = setup.sg_g(argv=("--" + avg).split())
            assert np.allclose(G.grid, gavg.grid)
        gavg = g.average(1)
        for avg in ["average y", "average 1"]:
            G = setup.sg_g(argv=("--" + avg).split())
            assert np.allclose(G.grid, gavg.grid)
        gavg = g.average(2)
        for avg in ["average z", "average 2"]:
            G = setup.sg_g(argv=("--" + avg).split())
            assert np.allclose(G.grid, gavg.grid)

    def test_average2(self, setup):
        g = setup.grid.copy()
        gavg = g.average(0).average(1)
        for avg in ["average x --average 1", "average 0 --average y"]:
            G = setup.sg_g(argv=("--" + avg).split())
            assert np.allclose(G.grid, gavg.grid)

    def test_sum1(self, setup):
        g = setup.grid.copy()
        gavg = g.sum(0)
        for avg in ["sum x", "sum 0"]:
            G = setup.sg_g(argv=("--" + avg).split())
            assert np.allclose(G.grid, gavg.grid)
        gavg = g.sum(1)
        for avg in ["sum y", "sum 1"]:
            G = setup.sg_g(argv=("--" + avg).split())
            assert np.allclose(G.grid, gavg.grid)
        gavg = g.sum(2)
        for avg in ["sum z", "sum 2"]:
            G = setup.sg_g(argv=("--" + avg).split())
            assert np.allclose(G.grid, gavg.grid)

    def test_sum2(self, setup):
        g = setup.grid.copy()
        gavg = g.sum(0).sum(1)
        for avg in ["sum x --sum 1", "sum 0 --sum y"]:
            G = setup.sg_g(argv=("--" + avg).split())
            assert np.allclose(G.grid, gavg.grid)

    def test_print1(self, setup):
        setup.sg_g(argv=["--info"])

    def test_interp(self, setup):
        g1 = setup.sg_g(argv="--interp 10 10 10".split())
        # last argument is default
        g2 = setup.sg_g(argv="--interp 10 10 10 1".split())
        assert np.allclose(g1.grid, g2.grid)
        g2 = setup.sg_g(argv="--interp 10 10 10 3".split())
        assert not np.allclose(g1.grid, g2.grid)
        g3 = setup.sg_g(argv="--interp 0.01 0.1 1. 3".split())

    def test_smooth(self, setup):
        g1 = setup.sg_g(argv=["--smooth"])
        g2 = setup.sg_g(argv="--smooth 0.7".split())
        assert np.allclose(g1.grid, g2.grid)
        g2 = setup.sg_g(argv="--smooth 1.".split())
        assert not np.allclose(g1.grid, g2.grid)

    def test_sub1(self, setup):
        g = setup.grid.copy()
        idx = g.index(1.0, 0)
        gs = g.sub_part(idx, 0, True)
        for sub in ["sub 1.: a", "sub 1.: 0"]:
            G = setup.sg_g(argv=("--" + sub).split())
            assert np.allclose(G.grid, gs.grid)

        idx = g.index(1.2, 1)
        gs = g.sub_part(idx, 1, True)
        for sub in ["sub 1.2: b", "sub 1.2: y"]:
            G = setup.sg_g(argv=("--" + sub).split())
            assert np.allclose(G.grid, gs.grid)

        idx = g.index(0.8, 2)
        gs = g.sub_part(idx, 2, False)
        for sub in ["sub :.8 2", "sub :0.8 z"]:
            G = setup.sg_g(argv=("--" + sub).split())
            assert np.allclose(G.grid, gs.grid)

    def test_remove1(self, setup):
        g = setup.grid.copy()
        idx = g.index(1.0, 0)
        gs = g.remove_part(idx, 0, True)
        for remove in ["remove 1.: a", "remove 1.: 0"]:
            G = setup.sg_g(argv=("--" + remove).split())
            assert np.allclose(G.grid, gs.grid)

        idx = g.index(1.2, 1)
        gs = g.remove_part(idx, 1, False)
        for remove in ["remove :1.2 b", "remove :1.2 y"]:
            G = setup.sg_g(argv=("--" + remove).split())
            assert np.allclose(G.grid, gs.grid)

        idx = g.index(0.8, 2)
        gs = g.remove_part(idx, 2, False)
        for remove in ["remove :.8 2", "remove :0.8 z"]:
            G = setup.sg_g(argv=("--" + remove).split())
            assert np.allclose(G.grid, gs.grid)

    def test_tile1(self, setup):
        g = setup.grid.copy()

        g2 = g.tile(2, 0)
        G = setup.sg_g(argv="--tile 2 a".split())
        assert np.allclose(G.grid, g2.grid)

        g2 = g.tile(2, 1)
        G = setup.sg_g(argv="--tile 2 y".split())
        assert np.allclose(G.grid, g2.grid)

        g2 = g.tile(2, 2)
        G = setup.sg_g(argv="--tile 2 c".split())
        assert np.allclose(G.grid, g2.grid)

    def test_write_data(self, setup, sisl_tmp):
        out = sisl_tmp("table.dat")
        G = setup.sg_g(argv=f"--sum 0 --average 1 --out {out}".split())
        dat = get_sile(out).read_data()
        assert np.allclose(dat[1, :], G.grid.ravel())

    def test_write_grid(self, setup, sisl_tmp):
        out = sisl_tmp("table.cube")
        G = setup.sg_g(argv=f"--sum 0 --out {out}".split())
