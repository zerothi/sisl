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


def test_geometry_car_group(sisl_tmp):
    f = sisl_tmp('test_sort.POSCAR', _dir)

    atoms = [Atom[1],
             Atom[2],
             Atom[2],
             Atom[1],
             Atom[1],
             Atom[2],
             Atom[3]]
    xyz = np.random.rand(len(atoms), 3)
    geom = Geometry(xyz, atoms, 100)

    geom.write(carSileVASP(f, 'w'), group_species=True)

    assert carSileVASP(f).read_geometry() != geom
    geom = carSileVASP(f).geometry_group(geom)
    assert carSileVASP(f).read_geometry() == geom


def test_geometry_car_allsame(sisl_tmp):
    f = sisl_tmp('test_read_write.POSCAR', _dir)

    atoms = Atom[1]
    xyz = np.random.rand(10, 3)
    geom = Geometry(xyz, atoms, 100)

    geom.write(carSileVASP(f, 'w'))

    assert carSileVASP(f).read_geometry() == geom


def test_geometry_car_dynamic(sisl_tmp):
    f = sisl_tmp('test_dynamic.POSCAR', _dir)

    atoms = Atom[1]
    xyz = np.random.rand(10, 3)
    geom = Geometry(xyz, atoms, 100)

    read = carSileVASP(f)

    # no dynamic (direct geometry)
    geom.write(carSileVASP(f, 'w'), dynamic=None)
    g, dyn = read.read_geometry(ret_dynamic=True)
    assert dyn is None

    geom.write(carSileVASP(f, 'w'), dynamic=False)
    g, dyn = read.read_geometry(ret_dynamic=True)
    assert not np.any(dyn)

    geom.write(carSileVASP(f, 'w'), dynamic=True)
    g, dyn = read.read_geometry(ret_dynamic=True)
    assert np.all(dyn)

    dynamic = [False] * len(geom)
    dynamic[0] = [True, False, True]
    geom.write(carSileVASP(f, 'w'), dynamic=dynamic)
    g, dyn = read.read_geometry(ret_dynamic=True)
    assert np.array_equal(dynamic[0], dyn[0])
    assert not np.any(dyn[1:])
