# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from numpy import ndarray

from sisl._core import Geometry
from sisl._internal import set_module
from sisl.typing import AtomsIndex

from .base import AtomCategory, NullCategory, _sanitize_loop

__all__ = ["AtomNeighbors"]


@set_module("sisl.geom")
class AtomNeighbors(AtomCategory):
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
       - ``tuple``, directly passed and thus only neigbours within
         the tuple range are considered
       - ``float``, this will pass ``(0.01, R)`` and thus *not* count the
         atom itself.
       - ``callable``, the return value of this will be directly passed.
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

    @_sanitize_loop
    def categorize(self, geometry: Geometry, atoms: AtomsIndex = None):
        """Check if geometry and atoms matches the neighbor criteria"""
        idx = geometry.close(atoms, R=self.R(geometry.atoms[atoms]))[-1]
        # quick escape the lower bound, in case we have more than max, they could
        # be limited by the self._in type
        n = len(idx)
        if n < self._min:
            return NullCategory()

        # Check if we have a condition
        if not self._in is None:
            # Get category of neighbors
            cat = self._in.categorize(geometry, geometry.asc2uc(idx))
            idx = [i for i, c in zip(idx, cat) if not isinstance(c, NullCategory)]
            n = len(idx)

        if self._min <= n <= self._max:
            return self
        return NullCategory()

    def __eq__(self, other):
        eq = self.__class__ is other.__class__
        if eq:
            return (
                self._min == other._min
                and self._max == other._max
                and self._in == other._in
            )
        return False
