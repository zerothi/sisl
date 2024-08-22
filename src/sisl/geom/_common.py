# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np

__all__ = ["geometry_define_nsc", "geometry2uc"]


def geometry_define_nsc(geometry, periodic=(True, True, True)):
    """Define the number of supercells for a geometry based on the periodicity"""

    # Pre-set nsc, PBC and the axes
    nsc, pbc, axes = {}, {}, []

    for ax, (d, per) in enumerate(zip("abc", periodic)):
        if per:
            nsc[d] = 3
            pbc[d] = "Periodic"
            axes.append(ax)
        else:
            nsc[d] = 1
            pbc[d] = "Unknown"

    geometry.set_nsc(**nsc)

    if np.all(geometry.maxR(True) > 0.0):
        nsc = geometry.find_nsc(axes)
        geometry.set_nsc(nsc)

    geometry.lattice.set_boundary_condition(**pbc)


def geometry2uc(geometry, dx=1e-8):
    """Translate the geometry to the unit cell by first shifting `dx`"""
    geometry = geometry.move(dx).translate2uc().move(-dx)
    geometry.xyz[geometry.xyz < 0] = 0
    return geometry
