from __future__ import print_function, division

import pytest

from sisl import Geometry, Atom, Grid
from sisl.io.xsf import *

import numpy as np

pytestmark = pytest.mark.only


def test_default(dir_test):
    f = dir_test.file('GRID.xsf')
    print(f)
    grid = Grid(0.2)
    grid.grid = np.random.rand(*grid.shape)
    grid.write(f)
    assert grid.geometry is None


def test_default_size(dir_test):
    f = dir_test.file('GRID.xsf')
    grid = Grid(0.2, sc=2.0)
    grid.grid = np.random.rand(*grid.shape)
    grid.write(f)
    assert grid.geometry is None


def test_geometry(dir_test):
    f = dir_test.file('GRID.xsf')
    geom = Geometry(np.random.rand(10, 3), np.random.randint(1, 70, 10), sc=[10, 10, 10, 45, 60, 90])
    grid = Grid(0.2, geom=geom)
    grid.grid = np.random.rand(*grid.shape)
    grid.write(f)
    assert not grid.geometry is None


def test_imaginary(dir_test):
    f = dir_test.file('GRID.xsf')
    geom = Geometry(np.random.rand(10, 3), np.random.randint(1, 70, 10), sc=[10, 10, 10, 45, 60, 90])
    grid = Grid(0.2, geom=geom, dtype=np.complex128)
    grid.grid = np.random.rand(*grid.shape) + 1j*np.random.rand(*grid.shape)
    grid.write(f)
    assert not grid.geometry is None
