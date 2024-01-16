# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import math

import numpy as np

from sisl._ufuncs import register_sisl_dispatch

from .lattice import Lattice

# Nothing gets exposed here
__all__ = []


@register_sisl_dispatch(module="sisl")
def tile(lattice: Lattice, reps: int, axis: int) -> Lattice:
    """Extend the unit-cell `reps` times along the `axis` lattice vector

    Notes
    -----
    This is *exactly* equivalent to the `repeat` routine.

    Parameters
    ----------
    lattice: Lattice
        the lattice object to act on
    reps :
        number of times the unit-cell is repeated along the specified lattice vector
    axis :
        the lattice vector along which the repetition is performed
    """
    cell = np.copy(lattice.cell)
    nsc = np.copy(lattice.nsc)
    origin = np.copy(lattice.origin)
    cell[axis] *= reps
    # Only reduce the size if it is larger than 5
    if nsc[axis] > 3 and reps > 1:
        # This is number of connections for the primary cell
        h_nsc = nsc[axis] // 2
        # The new number of supercells will then be
        nsc[axis] = max(1, int(math.ceil(h_nsc / reps))) * 2 + 1
    return lattice.__class__(cell, nsc=nsc, origin=origin)
