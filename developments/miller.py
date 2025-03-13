# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from fractions import gcd

import numpy as np

import sisl as si

ortho = si.utils.math.orthogonalize

# The intent of this small routine is to generate a lattice given by
# providing the Miller indices.
# This is not functional, yet, but could be useful to generate
# lattices etc.


def get_miller(rcell, hkl):
    # Get direction normal to the plane (Miller direction
    hkl = np.array(hkl, np.int32).reshape(3)
    if np.all(hkl == 0):
        return rcell
    d = gcd(gcd(hkl[0], hkl[1]), hkl[2])
    if d != 1:
        # Remove the common denominator
        hkl //= d
    if (hkl < 0).sum() > 1:
        # Rotate to have the majority as positive numbers
        hkl *= -1
    v0 = np.dot(hkl, rcell)
    # Now create the other directions which are orthogonal to the
    # Miller direction
    # If any of the Miller indices are negative,

    ortho = si.utils.math.orthogonalize
    # Figure out which directions we can use
    # decide on a lattice vector
    # We start by finding one zero (Miller-plane is parallel)
    idx = (hkl == 0).nonzero()[0]
    if len(idx) == 2:
        # We already now have the two vectors
        v1 = rcell[idx[0], :]
        v2 = rcell[idx[1], :]

    elif len(idx) == 1:
        v1 = rcell[idx[0], :]
        hkl2 = np.delete(hkl, idx[0])
        v2 = rcell[hkl2[0], :] - rcell[hkl2[1], :]

    else:
        # We simply have to decide on "some" vectors
        v1 = hkl[0] * rcell[0, :] + hkl[1] * rcell[1, :] - hkl[2] * rcell[2, :]
        v2 = hkl[0] * rcell[0, :] - hkl[1] * rcell[1, :] - hkl[2] * rcell[2, :]

    v1 = ortho(v0, v1)
    v2 = ortho(v0, v2)
    rv = np.array([v0, v1, v2])

    # Now we should align the vectors such that v0 points along the x-direction
    r0, t0, p0 = si.utils.math.cart2spher(v0)
    # Create a rotation matrix that rotates the first vector to be along the
    # first lattice.
    q0 = si.Quaternion(-p0, [0, 0, 1.0], rad=True)
    q1 = si.Quaternion(-t0, [1.0, 0, 0], rad=True)
    q = q0 * q1
    rv = q.rotate(rv)
    # Remove too small numbers
    rv = np.where(np.abs(rv) < 1e-10, 0, rv)
    v = np.linalg.inv(rv) * 2.0 * np.pi
    print(v)

    return q.rotate(v)
