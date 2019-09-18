from __future__ import print_function, division

import pytest
import os.path as osp
from sisl import Geometry, Atom, Grid
from sisl.io.xsf import *
import numpy as np


pytestmark = pytest.mark.io
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
