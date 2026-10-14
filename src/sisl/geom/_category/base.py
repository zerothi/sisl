# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from functools import wraps

import numpy as np

from sisl._category import NullCategory
from sisl._core import AtomCategory, Geometry
from sisl.typing import AtomsIndex

__all__ = ["NullCategory", "AtomCategory"]


def _sanitize_loop(func):
    @wraps(func)
    def loop_func(self, geometry: Geometry, atoms: AtomsIndex = None):
        if atoms is None:
            return [func(self, geometry, ia) for ia in geometry]

        # extract based on atoms selection
        atoms = geometry._sanitize_atoms(atoms)
        if atoms.ndim == 0:
            return func(self, geometry, atoms)
        return [func(self, geometry, ia) for ia in atoms]

    return loop_func


def _mask_subset(func):
    """Wrap ``func(self, geometry, cache)``, which returns the mask of all atoms"""

    @wraps(func)
    def mask_func(self, geometry: Geometry, atoms: AtomsIndex = None, *, cache: dict):
        mask = func(self, geometry, cache)
        if atoms is None:
            return mask
        return mask[geometry._sanitize_atoms(atoms).ravel()]

    return mask_func


def _find_pairs(geometry: Geometry, R: float):
    from sisl.geom._neighbors import NeighborFinder

    # Search a periodic copy with all atoms in the unit cell, then go back
    # to the supercell indices of `geometry` and keep those in nsc.
    shift = np.floor(geometry.fxyz).astype(np.int64)
    search = geometry.copy()
    search.xyz = geometry.xyz - shift @ geometry.cell
    search.lattice.pbc = [True] * 3

    finder = NeighborFinder(search, R=max(R, 0.01), overlap=False)
    pairs = finder.find_neighbors(self_interaction=True)
    I = pairs.I
    J = pairs.J
    isc = pairs.isc - shift[J] + shift[I]
    keep = np.all(np.abs(isc) <= geometry.nsc // 2, axis=1)
    I = I[keep]
    J = J[keep]
    isc = isc[keep]
    xyz = geometry.xyz
    rij = xyz[J] + isc @ geometry.cell - xyz[I]
    d = np.sqrt((rij**2).sum(1))
    keep = d <= R
    return I[keep], J[keep], isc[keep], rij[keep], d[keep]


def _neighbor_pairs(geometry: Geometry, R: float, cache: dict):
    """Atom pairs ``(I, J, isc, rij, d)`` with distance ``d <= R``

    Equivalent to `Geometry.close`: only the supercells in ``nsc`` are searched,
    and ``I`` is its own neighbor at ``d == 0``.
    The pairs of the largest `R` are stored in `cache` and filtered for smaller `R`.
    """
    if R < 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty((0, 3), np.int64), np.empty((0, 3)), np.empty(0)

    key = ("neighbor_pairs", id(geometry))
    cached = cache.get(key)
    if cached is None or cached[0] < R:
        cached = (R, _find_pairs(geometry, R))
        cache[key] = cached

    pairs = cached[1]
    if cached[0] > R:
        keep = pairs[-1] <= R
        pairs = tuple(p[keep] for p in pairs)
    return pairs
