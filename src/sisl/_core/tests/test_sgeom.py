# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl import Atom, Geometry, Lattice
from sisl._core.geometry import sgeom


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

            self.mol = Geometry([[i, 0, 0] for i in range(10)], lattice=[50])

            def sg_g(**kwargs):
                kwargs["ret_geometry"] = True
                if "geometry" not in kwargs:
                    kwargs["geometry"] = self.g.copy()
                return sgeom(**kwargs)

            self.sg_g = sg_g

            def sg_mol(**kwargs):
                kwargs["ret_geometry"] = True
                if "geometry" not in kwargs:
                    kwargs["geometry"] = self.mol.copy()
                return sgeom(**kwargs)

            self.sg_mol = sg_mol

    return t()


@pytest.mark.geometry
class TestGeometry:
    def test_help(self):
        with pytest.raises(SystemExit):
            sgeom(argv=["--help"])

    def test_version(self):
        sgeom(argv=["--version"])

    def test_cite(self):
        sgeom(argv=["--cite"])

    def test_tile1(self, setup):
        cell = np.copy(setup.g.lattice.cell)
        cell[0, :] *= 2
        for tile in ["tile 2 x", "tile-x 2"]:
            tx = setup.sg_g(argv=("--" + tile).split())
            assert np.allclose(cell, tx.lattice.cell)
        cell[1, :] *= 2
        for tile in ["tile 2 y", "tile-y 2"]:
            ty = setup.sg_g(geometry=tx, argv=("--" + tile).split())
            assert np.allclose(cell, ty.lattice.cell)
        cell[2, :] *= 2
        for tile in ["tile 2 z", "tile-z 2"]:
            tz = setup.sg_g(geometry=ty, argv=("--" + tile).split())
            assert np.allclose(cell, tz.lattice.cell)

    def test_tile2(self, setup):
        cell = np.copy(setup.g.lattice.cell)
        cell[:, :] *= 2
        for xt in ["tile 2 x", "tile-x 2"]:
            xt = "--" + xt
            for yt in ["tile 2 y", "tile-y 2"]:
                yt = "--" + yt
                for zt in ["tile 2 z", "tile-z 2"]:
                    zt = "--" + zt
                    argv = " ".join([xt, yt, zt]).split()
                    t = setup.sg_g(argv=argv)
                    assert np.allclose(cell, t.lattice.cell)

    def test_repeat1(self, setup):
        cell = np.copy(setup.g.lattice.cell)
        cell[0, :] *= 2
        for repeat in ["repeat 2 x", "repeat-x 2"]:
            tx = setup.sg_g(argv=("--" + repeat).split())
            assert np.allclose(cell, tx.lattice.cell)
        cell[1, :] *= 2
        for repeat in ["repeat 2 y", "repeat-y 2"]:
            ty = setup.sg_g(geometry=tx, argv=("--" + repeat).split())
            assert np.allclose(cell, ty.lattice.cell)
        cell[2, :] *= 2
        for repeat in ["repeat 2 z", "repeat-z 2"]:
            tz = setup.sg_g(geometry=ty, argv=("--" + repeat).split())
            assert np.allclose(cell, tz.lattice.cell)

    def test_repeat2(self, setup):
        cell = np.copy(setup.g.lattice.cell)
        cell[:, :] *= 2
        for xt in ["repeat 2 x", "repeat-x 2"]:
            xt = "--" + xt
            for yt in ["repeat 2 y", "repeat-y 2"]:
                yt = "--" + yt
                for zt in ["repeat 2 z", "repeat-z 2"]:
                    zt = "--" + zt
                    argv = " ".join([xt, yt, zt]).split()
                    t = setup.sg_g(argv=argv)
                    assert np.allclose(cell, t.lattice.cell)

    def test_sub(self, setup):
        for a, l in [("0", 1), ("0,1", 2), ("0-1", 2)]:
            g = setup.sg_g(argv=["--sub", a])
            assert len(g) == l

    def test_remove(self, setup):
        geom = setup.g.tile(2, 0).tile(2, 1)
        N = len(geom)
        for a, l in [("0", 1), ("0,1", 2), ("0-1", 2)]:
            g = setup.sg_g(geometry=geom.copy(), argv=["--remove", a])
            assert len(g) == N - l

    def test_rotation1(self, setup):
        rot = setup.sg_g(argv="--rotate 180 z".split())
        rot.lattice.cell[2, 2] *= -1
        assert np.allclose(-rot.lattice.cell, setup.g.lattice.cell)
        assert np.allclose(-rot.xyz, setup.g.xyz)

        rot = setup.sg_g(argv="--rotate-z 180".split())
        rot.lattice.cell[2, 2] *= -1
        assert np.allclose(-rot.lattice.cell, setup.g.lattice.cell)
        assert np.allclose(-rot.xyz, setup.g.xyz)

        rot = setup.sg_g(argv="--rotate rpi z".split())
        rot.lattice.cell[2, 2] *= -1
        assert np.allclose(-rot.lattice.cell, setup.g.lattice.cell)
        assert np.allclose(-rot.xyz, setup.g.xyz)

        rot = setup.sg_g(argv="--rotate-z rpi".split())
        rot.lattice.cell[2, 2] *= -1
        assert np.allclose(-rot.lattice.cell, setup.g.lattice.cell)
        assert np.allclose(-rot.xyz, setup.g.xyz)

    def test_swap(self, setup):
        s = setup.sg_g(argv="--swap 0 1".split())
        for i in [0, 1, 2]:
            assert np.allclose(setup.g.xyz[::-1, i], s.xyz[:, i])
