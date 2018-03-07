from __future__ import print_function, division

import pytest

from sisl import Geometry, Atom, Grid
from sisl.io.cube import *

import numpy as np

pytestmark = pytest.mark.io


def test_default(dir_test):
    f = dir_test.file('GRID.cube')
    grid = Grid(0.2)
    grid.grid = np.random.rand(*grid.shape)
    grid.write(f)
    read = grid.read(f)
    assert np.allclose(grid.grid, read.grid)
    assert grid.geometry is None
    assert read.geometry is None


def test_default_size(dir_test):
    f = dir_test.file('GRID.cube')
    grid = Grid(0.2, sc=2.0)
    grid.grid = np.random.rand(*grid.shape)
    grid.write(f)
    read = grid.read(f)
    assert np.allclose(grid.grid, read.grid)
    assert grid.geometry is None
    assert read.geometry is None


def test_geometry(dir_test):
    f = dir_test.file('GRID.cube')
    geom = Geometry(np.random.rand(10, 3), np.random.randint(1, 70, 10), sc=[10, 10, 10, 45, 60, 90])
    grid = Grid(0.2, geom=geom)
    grid.grid = np.random.rand(*grid.shape)
    grid.write(f)
    read = grid.read(f)
    assert np.allclose(grid.grid, read.grid)
    assert not grid.geometry is None
    assert not read.geometry is None
    assert grid.geometry == read.geometry


def test_imaginary(dir_test):
    fr = dir_test.file('GRID_real.cube')
    fi = dir_test.file('GRID_imag.cube')
    geom = Geometry(np.random.rand(10, 3), np.random.randint(1, 70, 10), sc=[10, 10, 10, 45, 60, 90])
    grid = Grid(0.2, geom=geom, dtype=np.complex128)
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
