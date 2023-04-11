# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pytest
import os.path as osp
from sisl import Geometry, Atom, Grid
from sisl.io.xsf import *
import numpy as np
from itertools import zip_longest


pytestmark = [pytest.mark.io, pytest.mark.generic]
_dir = osp.join('sisl', 'io')


def test_default(sisl_tmp):
    f = sisl_tmp('GRID_default.xsf', _dir)
    grid = Grid(0.2)
    grid.grid = np.random.rand(*grid.shape)
    grid.write(f)
    assert grid.geometry is None


def test_default_size(sisl_tmp):
    f = sisl_tmp('GRID_default_size.xsf', _dir)
    grid = Grid(0.2, lattice=2.0)
    grid.grid = np.random.rand(*grid.shape)
    grid.write(f)
    assert grid.geometry is None


def test_geometry(sisl_tmp):
    f = sisl_tmp('GRID_geometry.xsf', _dir)
    geom = Geometry(np.random.rand(10, 3), np.random.randint(1, 70, 10), lattice=[10, 10, 10, 45, 60, 90])
    grid = Grid(0.2, geometry=geom)
    grid.grid = np.random.rand(*grid.shape)
    grid.write(f)
    assert not grid.geometry is None


def test_imaginary(sisl_tmp):
    f = sisl_tmp('GRID_imag.xsf', _dir)
    geom = Geometry(np.random.rand(10, 3), np.random.randint(1, 70, 10), lattice=[10, 10, 10, 45, 60, 90])
    grid = Grid(0.2, geometry=geom, dtype=np.complex128)
    grid.grid = np.random.rand(*grid.shape) + 1j*np.random.rand(*grid.shape)
    grid.write(f)
    assert not grid.geometry is None


def test_axsf_geoms(sisl_tmp):
    f = sisl_tmp('multigeom_nodata.axsf', _dir)
    geom = Geometry(np.random.rand(10, 3), np.random.randint(1, 70, 10), lattice=[10, 10, 10, 45, 60, 90])
    geoms = [geom.move((i/10, i/10, i/10)) for i in range(3)]

    with xsfSile(f, "w", steps=3) as s:
        for g in geoms:
            s.write_geometry(g)

    with xsfSile(f) as s:
        rgeoms = s.read_geometry(start=0, stop=2)
        assert len(rgeoms) == 2
        assert all(isinstance(rg, Geometry) for rg in rgeoms)
        print()
        #for g, rg in zip(geoms, rgeoms):
        #    print(g)
        #    print(rg)
        assert all(g.equal(rg) for g, rg in zip(geoms, rgeoms))

    with xsfSile(f) as s:
        rgeoms = s.read_geometry(start=1)
        assert isinstance(rgeoms, Geometry)
        assert geoms[1].equal(rgeoms)

    with xsfSile(f) as s:
        rgeoms = s.read_geometry(all=True)
        assert len(rgeoms) == len(geoms)
        assert all(isinstance(rg, Geometry) for rg in rgeoms)
        assert all(g.equal(rg) for g, rg in zip(geoms, rgeoms))

    with xsfSile(f) as s:
        rgeoms, rdata = s.read_geometry(all=True, ret_data=True)
        assert len(rgeoms) == 3
        assert all(g.equal(rg) for g, rg in zip(geoms, rgeoms))
        for dat in rdata:
            assert dat.size == 0


def test_axsf_data(sisl_tmp):
    f = sisl_tmp('multigeom_data.axsf', _dir)
    geom = Geometry(np.random.rand(10, 3), np.random.randint(1, 70, 10), lattice=[10, 10, 10, 45, 60, 90])
    geoms = [geom.move((i/10, i/10, i/10)) for i in range(3)]
    data = np.random.rand(3, 10, 3)

    with xsfSile(f, "w", steps=3) as s:
        for g, dat in zip(geoms, data):
            s.write_geometry(g, data=dat)

    with xsfSile(f) as s:
        rgeoms, rdata = s.read_geometry(all=True, ret_data=True)
        assert len(rgeoms) == len(geoms)
        assert all(g.equal(rg) for g, rg in zip(geoms, rgeoms))
        assert all(np.allclose(d0, d1) for d0, d1 in zip(rdata, data))

    with xsfSile(f) as s:
        rgeoms, rdata = s.read_geometry(start=0, ret_data=True)
        assert geoms[0].equal(rgeoms)
        assert np.allclose(rdata, data[0])

