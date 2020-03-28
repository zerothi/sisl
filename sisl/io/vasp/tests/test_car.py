import pytest
import os.path as osp
from sisl import Geometry, Atom
from sisl.io.vasp.car import *
import numpy as np


pytestmark = [pytest.mark.io, pytest.mark.vasp]
_dir = osp.join('sisl', 'io', 'vasp')


def test_geometry_car_mixed(sisl_tmp):
    f = sisl_tmp('test_read_write.POSCAR', _dir)

    atoms = [Atom[1],
             Atom[2],
             Atom[2],
             Atom[1],
             Atom[1],
             Atom[2],
             Atom[3]]
    xyz = np.random.rand(len(atoms), 3)
    geom = Geometry(xyz, atoms, 100)

    geom.write(carSileVASP(f, 'w'))

    assert carSileVASP(f).read_geometry() == geom


def test_geometry_car_allsame(sisl_tmp):
    f = sisl_tmp('test_read_write.POSCAR', _dir)

    atoms = Atom[1]
    xyz = np.random.rand(10, 3)
    geom = Geometry(xyz, atoms, 100)

    geom.write(carSileVASP(f, 'w'))

    assert carSileVASP(f).read_geometry() == geom


def test_geometry_car_allsame(sisl_tmp):
    f = sisl_tmp('test_read_write.POSCAR', _dir)

    atoms = Atom[1]
    xyz = np.random.rand(10, 3)
    geom = Geometry(xyz, atoms, 100)

    geom.write(carSileVASP(f, 'w'))

    assert carSileVASP(f).read_geometry() == geom


def test_geometry_car_fixed(sisl_tmp):
    f = sisl_tmp('test_fixed.POSCAR', _dir)

    atoms = Atom[1]
    xyz = np.random.rand(10, 3)
    geom = Geometry(xyz, atoms, 100)

    read = carSileVASP(f)

    geom.write(carSileVASP(f, 'w'))
    g, fix = read.read_geometry(ret_fixed=True)
    assert not np.any(fix)

    geom.write(carSileVASP(f, 'w'), fixed=True)
    g, fix = read.read_geometry(ret_fixed=True)
    assert np.all(fix)

    fixed = [False] * len(geom)
    fixed[0] = [True, False, True]
    geom.write(carSileVASP(f, 'w'), fixed=fixed)
    g, fix = read.read_geometry(ret_fixed=True)
    assert np.array_equal(fixed[0], fix[0])
    assert not np.any(fix[1:])
