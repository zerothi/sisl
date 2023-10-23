# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np

__all__ = ["geometry_define_nsc", "geometry2uc"]


def geometry_define_nsc(geometry, periodic=(True, True, True)):
    """Define the number of supercells for a geometry based on the periodicity """
    if np.all(geometry.maxR(True) > 0.):
        geometry.optimize_nsc()
        for i, d, per in zip(range(3), "abc", periodic):
            if per:
                geometry.lattice.set_boundary_condition(**{d: "Periodic"})
            else:
                geometry.set_nsc(**{d: 1})
    else:
        nsc = [3 if p else 1 for p in periodic]
        geometry.set_nsc(nsc)

    for name, per in zip("abc", periodic):
        if per:
            per = "Periodic"
        else:
            per = "Unknown"
        geometry.lattice.set_boundary_condition(**{name: per})


def geometry2uc(geometry, dx=1e-8):
    """ Translate the geometry to the unit cell by first shifting `dx` """
    geometry = geometry.move(dx).translate2uc().move(-dx)
    geometry.xyz[geometry.xyz < 0] = 0
    return geometry
