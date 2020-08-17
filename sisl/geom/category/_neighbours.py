from collections import namedtuple

from numpy import ndarray

from sisl._internal import set_module
from .base import AtomCategory, NullCategory, _sanitize_loop


__all__ = ["AtomNeighbours"]


@set_module("sisl.geom")
class AtomNeighbours(AtomCategory):
    r""" Classify atoms based on number of neighbours

    Parameters
    ----------
    min : int, optional
       minimum number of neighbours
    max : int
       maximum number of neighbours
    neigh_cat : Category, optional
       a category the neighbour must be in to be counted
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
    >>> AtomNeighbours(4) # 4 neighbours within (0.01, Geometry.maxR())
    >>> AtomNeighbours(4, R=1.44) # 4 neighbours within (0.01, 1.44)
    >>> AtomNeighbours(4, R=(1, 1.44)) # 4 neighbours within (1, Geometry.maxR())
    >>> AtomNeighbours(4, R=lambda atom: (0.01, PeriodicTable().radius(atom.Z))) # 4 neighbours within (0.01, <>)
    """
    __slots__ = ("_min", "_max", "_in", "_R")

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            if isinstance(args[-1], AtomCategory):
                *args, kwargs["neigh_cat"] = args

        self._min = 0
        self._max = 2 ** 31

        if len(args) == 1:
            self._min = args[0]
            self._max = args[0]

        elif len(args) == 2:
            self._min = args[0]
            self._max = args[1]

        if "min" in kwargs:
            self._min = kwargs.pop("min")
        if "max" in kwargs:
            self._max = kwargs.pop("max")

        if self._min == self._max:
            name = f"={self._max}"
        elif self._max == 2 ** 31:
            name = f" ∈ [{self._min};∞["
        else:
            name = f" ∈ [{self._min};{self._max}]"

        self._in = kwargs.get("neigh_cat", None)
        self._R = kwargs.get("R", None)

        # Determine name. If there are requirements for the neighbours
        # then the name changes
        if self._in is None:
            self.set_name(f"neighbours{name}")
        else:
            self.set_name(f"neighbours({self._in}){name}")

    def R(self, atom):
        if self._R is None:
            return (0.01, atom.maxR())
        elif callable(self._R):
            return self._R(atom)
        elif isinstance(self._R, (tuple, list, ndarray)):
            return self._R
        return (0.01, self._R)

    @_sanitize_loop
    def categorize(self, geometry, atoms=None):
        """ Check if geometry and atoms matches the neighbour criteria """
        idx, rij = geometry.close(atoms, R=self.R(geometry.atoms[atoms]), ret_rij=True)
        idx, rij = idx[1], rij[1]
        if len(idx) < self._min:
            return NullCategory()

        # Check if we have a condition
        if not self._in is None:
            # Get category of neighbours
            cat = self._in.categorize(geometry, geometry.asc2uc(idx))
            idx1, rij1 = [], []
            for i in range(len(idx)):
                if not isinstance(cat[i], NullCategory):
                    idx1.append(idx[i])
                    rij1.append(rij[i])
            idx, rij = idx1, rij1
        n = len(idx)
        if self._min <= n and n <= self._max:
            return self
        return NullCategory()

    def __eq__(self, other):
        eq = self.__class__ is other.__class__
        if eq:
            return (self._min == other._min and
                    self._max == other._max and
                    self._in == other._in)
        return False
