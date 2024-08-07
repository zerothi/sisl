# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from sisl import Atom, Geometry, Lattice
from sisl._internal import set_module
from sisl.typing import AtomsLike

from ._common import geometry_define_nsc

__all__ = ["nanotube"]

FloatOrFloat2 = Union[float, tuple[float, float]]


@set_module("sisl.geom")
def nanotube(
    bond: float,
    atoms: Optional[AtomsLike] = None,
    chirality: tuple[int, int] = (1, 1),
    vacuum: FloatOrFloat2 = 20.0,
) -> Geometry:
    """Nanotube with user-defined chirality.

    This routine is implemented as in `ASE`_ with some cosmetic changes.

    Parameters
    ----------
    bond :
       length between atoms in nano-tube
    atoms :
       nanotube atoms
    chirality :
       chirality of nanotube (n, m)
    """
    if atoms is None:
        atoms = Atom(Z=6, R=bond * 1.01)

    # Correct the input...
    n, m = chirality
    if n < m:
        m, n = n, m
        sign = -1
    else:
        sign = 1

    sq3 = 3.0**0.5
    a = sq3 * bond
    l2 = n * n + m * m + n * m
    l = l2**0.5

    def gcd(a, b):
        while a != 0:
            a, b = b % a, a
        return b

    nd = gcd(n, m)
    if (n - m) % (3 * nd) == 0:
        ndr = 3 * nd
    else:
        ndr = nd

    nr = (2 * m + n) // ndr
    ns = -(2 * n + m) // ndr
    nn = 2 * l2 // ndr

    ichk = 0
    if nr == 0:
        n60 = 1
    else:
        n60 = nr * 4

    absn = abs(n60)
    nnp = []
    nnq = []
    for i in range(-absn, absn + 1):
        for j in range(-absn, absn + 1):
            j2 = nr * j - ns * i
            if j2 == 1:
                j1 = m * i - n * j
                if j1 > 0 and j1 < nn:
                    ichk += 1
                    nnp.append(i)
                    nnq.append(j)

    if ichk == 0:
        raise RuntimeError("not found p, q strange!!")
    if ichk >= 2:
        raise RuntimeError("more than 1 pair p, q strange!!")

    nnnp = nnp[0]
    nnnq = nnq[0]

    lp = nnnp * nnnp + nnnq * nnnq + nnnp * nnnq
    r = a * lp**0.5
    c = a * l
    t = sq3 * c / ndr

    rs = c / (2.0 * np.pi)

    q1 = np.arctan((sq3 * m) / (2 * n + m))
    q2 = np.arctan((sq3 * nnnq) / (2 * nnnp + nnnq))
    q3 = q1 - q2

    q4 = 2.0 * np.pi / nn
    q5 = bond * np.cos((np.pi / 6.0) - q1) / c * 2.0 * np.pi

    h1 = abs(t) / abs(np.sin(q3))
    h2 = bond * np.sin((np.pi / 6.0) - q1)

    xyz = np.empty([nn * 2, 3], np.float64)
    for i in range(nn):
        ix = i * 2

        k = np.floor(i * abs(r) / h1)
        xyz[ix, 0] = rs * np.cos(i * q4)
        xyz[ix, 1] = rs * np.sin(i * q4)
        z = (i * abs(r) - k * h1) * np.sin(q3)
        kk2 = abs(np.floor((z + 0.0001) / t))
        if z >= t - 0.0001:
            z -= t * kk2
        elif z < 0:
            z += t * kk2
        xyz[ix, 2] = z * sign

        # Next
        ix += 1
        xyz[ix, 0] = rs * np.cos(i * q4 + q5)
        xyz[ix, 1] = rs * np.sin(i * q4 + q5)
        z = (i * abs(r) - k * h1) * np.sin(q3) - h2
        if z >= 0 and z < t:
            pass
        else:
            z -= h1 * np.sin(q3)
            kk = abs(np.floor(z / t))
            if z >= t - 0.0001:
                z -= t * kk
            elif z < 0:
                z += t * kk
        xyz[ix, 2] = z * sign

    # Sort the atomic coordinates according to z
    idx = np.argsort(xyz[:, 2])
    xyz = xyz[idx, :]
    xyz_min, xyz_max = xyz.min(0), xyz.max(0)

    cell = xyz_max - xyz_min
    cell[:2] += vacuum
    cell[2] = t
    lattice = Lattice(cell)

    geom = Geometry(xyz, atoms, lattice=lattice)
    geom = geom.translate(-xyz_min)

    geometry_define_nsc(geom, [False, False, True])

    return geom
