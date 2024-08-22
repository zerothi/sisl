# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl.shape._cylinder import *
from sisl.utils.mathematics import fnorm

pytestmark = pytest.mark.shape


@pytest.mark.filterwarnings("ignore", message="*orthogonalizes the vectors")
def test_create_ellipticalcylinder():
    el = EllipticalCylinder(1.0, 1.0)
    el = EllipticalCylinder([1.0, 1.0], 1.0)
    v0 = [1.0, 0.2, 1.0]
    v1 = [1.0, -0.2, 1.0]
    el = EllipticalCylinder([v0, v1], 1.0)
    v0 = el.radial_vector[0]
    v1 = el.radial_vector[1]
    v0 /= fnorm(v0)
    v1 /= fnorm(v1)
    el = EllipticalCylinder([v0, v1], 1.0)
    e2 = el.scale(1.1)
    assert np.allclose(el.radius + 0.1, e2.radius)
    assert np.allclose(el.height + 0.1, e2.height)
    e2 = el.scale([1.1] * 3)
    assert np.allclose(el.radius + 0.1, e2.radius)
    assert np.allclose(el.height + 0.1, e2.height)
    e2 = el.scale([1.1, 2.1, 3.1])
    assert np.allclose(el.radius + [0.1, 1.1], e2.radius)
    assert np.allclose(el.height + 2.1, e2.height)
    str(el)
    e2.volume


def test_ellipticalcylinder_within():
    el = EllipticalCylinder(1.0, 1.0)
    # center of cylinder
    assert el.within_index([0, 0, 0])[0] == 0
    # should not be in a circle
    assert el.within_index([0.2, 0.2, 0.9])[0] == 0


def test_tosphere():
    el = EllipticalCylinder([1.0, 1.0], 1.0)
    el.to.Sphere()


def test_tocuboid():
    el = EllipticalCylinder([1.0, 1.0], 1.0)
    el.to.Cuboid()


def test_translate():
    el = EllipticalCylinder([1.0, 1.0], 1.0)
    el2 = el.translate([0, 1, 2])
    assert not np.allclose(el.center, el2.center)
