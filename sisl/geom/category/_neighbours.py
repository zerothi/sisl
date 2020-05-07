from collections import namedtuple

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
    neigh_cat : Category
       a category the neighbour must be in to be counted
    """
    __slots__ = ("_min", "_max", "_in")

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            if isinstance(args[-1], AtomCategory):
                kwargs["neigh_cat"] = args.pop()

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

        # Determine name. If there are requirements for the neighbours
        # then the name changes
        if self._in is None:
            self.set_name(f"neighbours{name}")
        else:
            self.set_name(f"neighbours({self._in}){name}")

    @_sanitize_loop
    def categorize(self, geometry, atoms=None):
        """ Check that number of neighbours are matching """
        idx, rij = geometry.close(atoms, R=(0.01, geometry.atoms[atoms].maxR()), ret_rij=True)
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
