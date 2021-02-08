import pytest
import os.path as osp
from sisl import Geometry, Atom, Grid
from sisl.io.xsf import *
import numpy as np
from itertools import zip_longest


pytestmark = [pytest.mark.io, pytest.mark.generic]
_dir = osp.join('sisl', 'io')


def test_default(sisl_tmp):
    f = sisl_tmp('GRID.xsf', _dir)
    grid = Grid(0.2)
    grid.grid = np.random.rand(*grid.shape)
    grid.write(f)
    assert grid.geometry is None


def test_default_size(sisl_tmp):
    f = sisl_tmp('GRID.xsf', _dir)
    grid = Grid(0.2, sc=2.0)
    grid.grid = np.random.rand(*grid.shape)
    grid.write(f)
    assert grid.geometry is None


def test_geometry(sisl_tmp):
    f = sisl_tmp('GRID.xsf', _dir)
    geom = Geometry(np.random.rand(10, 3), np.random.randint(1, 70, 10), sc=[10, 10, 10, 45, 60, 90])
    grid = Grid(0.2, geometry=geom)
    grid.grid = np.random.rand(*grid.shape)
    grid.write(f)
    assert not grid.geometry is None


def test_imaginary(sisl_tmp):
    f = sisl_tmp('GRID.xsf', _dir)
    geom = Geometry(np.random.rand(10, 3), np.random.randint(1, 70, 10), sc=[10, 10, 10, 45, 60, 90])
    grid = Grid(0.2, geometry=geom, dtype=np.complex128)
    grid.grid = np.random.rand(*grid.shape) + 1j*np.random.rand(*grid.shape)
    grid.write(f)
    assert not grid.geometry is None


def test_axsf_geoms(sisl_tmp):
    f = sisl_tmp('multigeom.axsf', _dir)
    geom = Geometry(np.random.rand(10, 3), np.random.randint(1, 70, 10), sc=[10, 10, 10, 45, 60, 90])
    geoms = [geom.move((i/10, i/10, i/10)) for i in range(3)]

    with axsfSile(f, "w", steps=3) as s:
        for i in range(3):
            s.write_geometry(geoms[i])

    with axsfSile(f) as s:
        rgeoms = s.read_geometry(index=(0, 2))
        assert all(isinstance(rg, Geometry) for rg in rgeoms)
        assert all(g.equal(rg) for g, rg in zip_longest([geoms[0], geoms[2]], rgeoms))

    with axsfSile(f) as s:
        rgeoms = s.read_geometry(index=1)
        assert isinstance(rgeoms, Geometry)
        assert geoms[1].equal(rgeoms)

    with axsfSile(f) as s:
        rgeoms = s.read_geometry(index=None)
        assert all(isinstance(rg, Geometry) for rg in rgeoms)
        assert all(g.equal(rg) for g, rg in zip_longest(geoms[:3], rgeoms))

    with axsfSile(f) as s:
        rgeoms, rdata = s.read_geometry(index=None, ret_data=True)
        assert all(g.equal(rg) for g, rg in zip_longest(geoms, rgeoms))
        assert rdata.shape == (3, 10, 0)


def test_axsf_data(sisl_tmp):
    f = sisl_tmp('multigeom.axsf', _dir)
    geom = Geometry(np.random.rand(10, 3), np.random.randint(1, 70, 10), sc=[10, 10, 10, 45, 60, 90])
    geoms = [geom.move((i/10, i/10, i/10)) for i in range(3)]
    data = np.random.rand(3, 10, 3)

    with axsfSile(f, "w", steps=3) as s:
        for i in range(3):
            s.write_geometry(geoms[i], data=data[i])

    with axsfSile(f) as s:
        rgeoms, rdata = s.read_geometry(index=None, ret_data=True)
        assert all(g.equal(rg) for g, rg in zip_longest(geoms, rgeoms))
        assert np.allclose(rdata, data)

    with axsfSile(f) as s:
        rgeoms, rdata = s.read_geometry(index=0, ret_data=True)
        assert geoms[0].equal(rgeoms)
        assert np.allclose(rdata, data[0, :, :])
