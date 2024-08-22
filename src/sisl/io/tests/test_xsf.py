# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl import Geometry, Grid
from sisl.io.xsf import *

pytestmark = [pytest.mark.io, pytest.mark.generic]


def test_xsf_default(sisl_tmp):
    f = sisl_tmp("GRID_default.xsf")
    grid = Grid(0.2)
    grid.grid = np.random.rand(*grid.shape)
    grid.write(f)
    assert grid.geometry is None


@pytest.mark.parametrize(
    "pbc",
    [
        (True, True, True),
        (True, True, False),
        (True, False, False),
        (False, False, False),
    ],
)
def test_xsf_pbc(sisl_tmp, pbc):
    f = sisl_tmp("GRID_default.xsf")
    geom = Geometry(
        np.random.rand(10, 3),
        np.random.randint(1, 70, 10),
        lattice=[10, 10, 10, 45, 60, 90],
    )
    geom.lattice.pbc = pbc
    geom.write(f)
    geom2 = geom.read(f)
    assert all(geom.pbc == geom2.pbc)


def test_xsf_default_size(sisl_tmp):
    f = sisl_tmp("GRID_default_size.xsf")
    grid = Grid(0.2, lattice=2.0)
    grid.grid = np.random.rand(*grid.shape)
    grid.write(f)
    assert grid.geometry is None


def test_xsf_geometry(sisl_tmp):
    f = sisl_tmp("GRID_geometry.xsf")
    geom = Geometry(
        np.random.rand(10, 3),
        np.random.randint(1, 70, 10),
        lattice=[10, 10, 10, 45, 60, 90],
    )
    grid = Grid(0.2, geometry=geom)
    grid.grid = np.random.rand(*grid.shape)
    grid.write(f)
    assert not grid.geometry is None


def test_xsf_imaginary(sisl_tmp):
    f = sisl_tmp("GRID_imag.xsf")
    geom = Geometry(
        np.random.rand(10, 3),
        np.random.randint(1, 70, 10),
        lattice=[10, 10, 10, 45, 60, 90],
    )
    grid = Grid(0.2, geometry=geom, dtype=np.complex128)
    grid.grid = np.random.rand(*grid.shape) + 1j * np.random.rand(*grid.shape)
    grid.write(f)
    assert not grid.geometry is None


def test_axsf_geoms(sisl_tmp):
    f = sisl_tmp("multigeom_nodata.axsf")
    geom = Geometry(
        np.random.rand(10, 3),
        np.random.randint(1, 70, 10),
        lattice=[10, 10, 10, 45, 60, 90],
    )
    geoms = [geom.move((i / 10, i / 10, i / 10)) for i in range(3)]

    with xsfSile(f, "w", steps=3) as s:
        for g in geoms:
            s.write_geometry(g)

    with xsfSile(f) as s:
        rgeoms = s.read_geometry[0:2]()
        assert len(rgeoms) == 2
        assert all(isinstance(rg, Geometry) for rg in rgeoms)
        assert all(g.equal(rg) for g, rg in zip(geoms, rgeoms))

    with xsfSile(f) as s:
        rgeoms = s.read_geometry[1]()
        assert isinstance(rgeoms, Geometry)
        assert geoms[1].equal(rgeoms)

    with xsfSile(f) as s:
        rgeoms = s.read_geometry[:]()
        assert len(rgeoms) == len(geoms)
        assert all(isinstance(rg, Geometry) for rg in rgeoms)
        assert all(g.equal(rg) for g, rg in zip(geoms, rgeoms))

    with xsfSile(f) as s:
        rgeoms, rdata = s.read_geometry[:](ret_data=True)
        assert len(rgeoms) == 3
        assert all(g.equal(rg) for g, rg in zip(geoms, rgeoms))
        for dat in rdata:
            assert dat.size == 0


def test_axsf_data(sisl_tmp):
    f = sisl_tmp("multigeom_data.axsf")
    geom = Geometry(
        np.random.rand(10, 3),
        np.random.randint(1, 70, 10),
        lattice=[10, 10, 10, 45, 60, 90],
    )
    geoms = [geom.move((i / 10, i / 10, i / 10)) for i in range(3)]
    data = np.random.rand(3, 10, 3)

    with xsfSile(f, "w", steps=3) as s:
        for g, dat in zip(geoms, data):
            s.write_geometry(g, data=dat)

    with xsfSile(f) as s:
        rgeoms, rdata = s.read_geometry[:](ret_data=True)
        assert len(rgeoms) == len(geoms)
        assert all(g.equal(rg) for g, rg in zip(geoms, rgeoms))
        assert all(np.allclose(d0, d1) for d0, d1 in zip(rdata, data))

    with xsfSile(f) as s:
        rgeoms, rdata = s.read_geometry[0](ret_data=True)
        assert geoms[0].equal(rgeoms)
        assert np.allclose(rdata, data[0])
