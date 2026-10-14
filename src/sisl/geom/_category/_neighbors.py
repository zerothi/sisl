# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
from numpy import ndarray

from sisl._core import Geometry
from sisl._internal import set_module
from sisl.typing import AtomsIndex

from .base import AtomCategory, NullCategory, _mask_subset, _neighbor_pairs

__all__ = ["AtomNeighbors"]


@set_module("sisl.geom")
class AtomNeighbors(AtomCategory, aliases=("neighbours",)):
    r"""Classify atoms based on number of neighbors

    Parameters
    ----------
    min : int, optional
       minimum number of neighbors
    max : int
       maximum number of neighbors
    neighbor : Category, optional
       a category the neighbor must be in to be counted
    R : tuple, float, callable or None, optional
       Value passed to `Geometry.close`.
       ``tuple``, directly passed and thus only neigbors within
       the tuple range are considered
       ``float``, this will pass ``(0.01, R)`` and thus *not* count the
       atom itself.
       ``callable``, the return value of this will be directly passed.
       If the callable returns a single float it will count the atom itself.

    Examples
    --------
    >>> AtomNeighbors(4) # 4 neighbors within (0.01, Geometry.maxR())
    >>> AtomNeighbors(4, R=1.44) # 4 neighbors within (0.01, 1.44)
    >>> AtomNeighbors(4, R=(1, 1.44)) # 4 neighbors within (1, Geometry.maxR())
    >>> AtomNeighbors(4, R=lambda atom: (0.01, PeriodicTable().radius(atom.Z))) # 4 neighbors within (0.01, <>)
    """

    __slots__ = ("_min", "_max", "_in", "_R")

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            if isinstance(args[-1], AtomCategory):
                *args, kwargs["neighbor"] = args

        _min = 0
        _max = 2**31

        if len(args) == 1:
            _min = args[0]
            _max = args[0]

        elif len(args) == 2:
            _min = args[0]
            _max = args[1]

        _min = kwargs.pop("min", _min)
        _max = kwargs.pop("max", _max)

        if _min is None:
            _min = 0
        if _max is None:
            _max = 2**31

        self._min = _min
        self._max = _max

        if self._min == self._max:
            name = f"={self._max}"
        elif self._max == 2**31:
            name = f" ∈ [{self._min};∞["
        else:
            name = f" ∈ [{self._min};{self._max}]"

        self._in = kwargs.get("neighbor", None)
        if isinstance(self._in, dict):
            self._in = AtomCategory(**self._in)
        self._R = kwargs.get("R", None)

        # Determine name. If there are requirements for the neighbors
        # then the name changes

        if self._in is None:
            name = f"neighbors{name}"
        else:
            name = f"neighbors({self._in}){name}"
        super().__init__(name)

    def R(self, atom):
        if self._R is None:
            return (0.01, atom.maxR())
        if callable(self._R):
            return self._R(atom)
        if isinstance(self._R, (tuple, list, ndarray)):
            return self._R
        return (0.01, self._R)

    def _R_window(self, geometry: Geometry) -> tuple[np.ndarray, np.ndarray]:
        """Per atom ``(lo, hi)`` so that a neighbor at distance ``d`` has ``lo < d <= hi``"""
        lo = []
        hi = []
        for atom in geometry.atoms.atom:
            R = np.ravel(self.R(atom))
            hi.append(R[-1])
            # A single radius counts the atom itself (d == 0)
            lo.append(R[-2] if R.size > 1 else -1.0)
        species = geometry.atoms.species
        return np.asarray(lo)[species], np.asarray(hi)[species]

    @_mask_subset
    def _mask(self, geometry: Geometry, cache: dict) -> np.ndarray:
        lo, hi = self._R_window(geometry)
        I, J, _, _, d = _neighbor_pairs(geometry, hi.max(initial=-1.0), cache)
        keep = (lo[I] < d) & (d <= hi[I])
        if self._in is not None:
            keep &= self._in._mask(geometry, cache=cache)[J]
        n = np.bincount(I[keep], minlength=geometry.na)
        return (self._min <= n) & (n <= self._max)

    def categorize(self, geometry: Geometry, atoms: AtomsIndex = None):
        """Check if geometry and atoms matches the neighbor criteria"""
        if atoms is not None:
            atoms = geometry._sanitize_atoms(atoms)
        null = NullCategory()
        ret = [self if m else null for m in self.mask(geometry, atoms)]
        if atoms is not None and atoms.ndim == 0:
            return ret[0]
        return ret

    def __eq__(self, other):
        eq = self.__class__ is other.__class__
        if eq:
            return (
                self._min == other._min
                and self._max == other._max
                and self._in == other._in
            )
        return False
