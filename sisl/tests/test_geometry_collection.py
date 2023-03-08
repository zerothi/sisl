# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pytest

import math as m
import numpy as np

import sisl

pytestmark = [pytest.mark.geometry, pytest.mark.collection]


def get_n_geoms(N, methods=["graphene", "nanotube"]):
    """ creates n geometries cycling through `methods` """
    if isinstance(methods, str):
        methods = [methods]

    nmethods = len(methods)
    
    g = []
    while len(g) < N:
        method = methods[len(g) % nmethods]
        g.append(getattr(sisl.geom, method)(4.))
    return g


def test_gc_creation():
    gs = sisl.GeometryCollection(get_n_geoms(10))
    assert len(gs) == 10


def test_gc_applymap():
    gs = sisl.GeometryCollection(get_n_geoms(10))
    lens = [len(g) for g in gs]
    gstiled = gs.applymap(sisl.Geometry.tile, reps=3, axis=2)
    assert lens == [len(g) // 3 for g in gstiled]
