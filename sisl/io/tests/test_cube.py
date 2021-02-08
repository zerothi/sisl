import pytest
import os.path as osp
from sisl import Geometry, Atom, Grid, SislError
from sisl.io.cube import *
import numpy as np


pytestmark = [pytest.mark.io, pytest.mark.generic]
_dir = osp.join('sisl', 'io')


def test_default(sisl_tmp):
    f = sisl_tmp('GRID.cube', _dir)
    grid = Grid(0.2)
    grid.grid = np.random.rand(*grid.shape)
    grid.write(f)
    read = grid.read(f)
    assert np.allclose(grid.grid, read.grid)
    assert grid.geometry is None
    assert len(read.geometry) == 1


def test_default_size(sisl_tmp):
    f = sisl_tmp('GRID.cube', _dir)
    grid = Grid(0.2, sc=2.0)
    grid.grid = np.random.rand(*grid.shape)
    grid.write(f)
    read = grid.read(f)
    assert np.allclose(grid.grid, read.grid)
    assert grid.geometry is None
    assert len(read.geometry) == 1


def test_geometry(sisl_tmp):
    f = sisl_tmp('GRID.cube', _dir)
    geom = Geometry(np.random.rand(10, 3), np.random.randint(1, 70, 10), sc=[10, 10, 10, 45, 60, 90])
    grid = Grid(0.2, geometry=geom)
    grid.grid = np.random.rand(*grid.shape)
    grid.write(f)
    read = grid.read(f)
    assert np.allclose(grid.grid, read.grid)
    assert not grid.geometry is None
    assert not read.geometry is None
    assert grid.geometry == read.geometry


def test_imaginary(sisl_tmp):
    fr = sisl_tmp('GRID_real.cube', _dir)
    fi = sisl_tmp('GRID_imag.cube', _dir)
    geom = Geometry(np.random.rand(10, 3), np.random.randint(1, 70, 10), sc=[10, 10, 10, 45, 60, 90])
    grid = Grid(0.2, geometry=geom, dtype=np.complex128)
    grid.grid = np.random.rand(*grid.shape) + 1j*np.random.rand(*grid.shape)
    grid.write(fr)
    grid.write(fi, imag=True)
    read = grid.read(fr)
    read_i = grid.read(fi)
    read.grid = read.grid + 1j*read_i.grid
    assert np.allclose(grid.grid, read.grid)
    assert not grid.geometry is None
    assert not read.geometry is None
    assert grid.geometry == read.geometry

    read = grid.read(fr, imag=fi)
    assert np.allclose(grid.grid, read.grid)

    read = grid.read(fr, imag=read_i)
    assert np.allclose(grid.grid, read.grid)


def test_imaginary_fail_shape(sisl_tmp):
    fr = sisl_tmp('GRID_real.cube', _dir)
    fi = sisl_tmp('GRID_imag.cube', _dir)
    geom = Geometry(np.random.rand(10, 3), np.random.randint(1, 70, 10), sc=[10, 10, 10, 45, 60, 90])
    grid = Grid(0.2, geometry=geom, dtype=np.complex128)
    grid.grid = np.random.rand(*grid.shape) + 1j*np.random.rand(*grid.shape)
    grid.write(fr)

    # Assert it fails on shape
    grid2 = Grid(0.3, geometry=geom, dtype=np.complex128)
    grid2.write(fi, imag=True)
    with pytest.raises(SislError):
        grid.read(fr, imag=fi)


def test_imaginary_fail_geometry(sisl_tmp):
    fr = sisl_tmp('GRID_real.cube', _dir)
    fi = sisl_tmp('GRID_imag.cube', _dir)
    geom = Geometry(np.random.rand(10, 3), np.random.randint(1, 70, 10), sc=[10, 10, 10, 45, 60, 90])
    grid = Grid(0.2, geometry=geom, dtype=np.complex128)
    grid.grid = np.random.rand(*grid.shape) + 1j*np.random.rand(*grid.shape)
    grid.write(fr)

    # Assert it fails on geometry
    grid2 = Grid(0.3, dtype=np.complex128)
    grid2.write(fi, imag=True)
    with pytest.raises(SislError):
        grid.read(fr, imag=fi)
