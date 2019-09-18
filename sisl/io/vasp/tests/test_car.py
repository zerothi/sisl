from __future__ import print_function, division

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
