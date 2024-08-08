# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl import Atom, Geometry
from sisl.io.vasp.car import *

pytestmark = [pytest.mark.io, pytest.mark.vasp]


def test_geometry_car_mixed(sisl_tmp):
    f = sisl_tmp("test_read_write.POSCAR")

    atoms = [Atom[1], Atom[2], Atom[2], Atom[1], Atom[1], Atom[2], Atom[3]]
    xyz = np.random.rand(len(atoms), 3)
    geom = Geometry(xyz, atoms, 100)

    geom.write(carSileVASP(f, "w"))

    assert carSileVASP(f).read_geometry() == geom


def test_geometry_car_group(sisl_tmp):
    f = sisl_tmp("test_sort.POSCAR")

    atoms = [Atom[1], Atom[2], Atom[2], Atom[1], Atom[1], Atom[2], Atom[3]]
    xyz = np.random.rand(len(atoms), 3)
    geom = Geometry(xyz, atoms, 100)

    geom.write(carSileVASP(f, "w"), group_species=True)

    assert carSileVASP(f).read_geometry() != geom
    geom = carSileVASP(f).geometry_group(geom)
    assert carSileVASP(f).read_geometry() == geom


def test_geometry_car_allsame(sisl_tmp):
    f = sisl_tmp("test_read_write.POSCAR")

    atoms = Atom[1]
    xyz = np.random.rand(10, 3)
    geom = Geometry(xyz, atoms, 100)

    geom.write(carSileVASP(f, "w"))

    assert carSileVASP(f).read_geometry() == geom


def test_geometry_car_dynamic(sisl_tmp):
    f = sisl_tmp("test_dynamic.POSCAR")

    atoms = Atom[1]
    xyz = np.random.rand(10, 3)
    geom = Geometry(xyz, atoms, 100)

    read = carSileVASP(f)

    # no dynamic (direct geometry)
    geom.write(carSileVASP(f, "w"), dynamic=None)
    g, dyn = read.read_geometry(ret_dynamic=True)
    assert dyn is None

    geom.write(carSileVASP(f, "w"), dynamic=False)
    g, dyn = read.read_geometry(ret_dynamic=True)
    assert not np.any(dyn)

    geom.write(carSileVASP(f, "w"), dynamic=True)
    g, dyn = read.read_geometry(ret_dynamic=True)
    assert np.all(dyn)

    dynamic = [False] * len(geom)
    dynamic[0] = [True, False, True]
    geom.write(carSileVASP(f, "w"), dynamic=dynamic)
    g, dyn = read.read_geometry(ret_dynamic=True)
    assert np.array_equal(dynamic[0], dyn[0])
    assert not np.any(dyn[1:])
