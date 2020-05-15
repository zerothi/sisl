import numpy as np

from sisl._internal import set_module, singledispatchmethod
from sisl._help import isiterable
from .base import AtomCategory, NullCategory, _sanitize_loop


__all__ = ["AtomZ", "AtomOdd", "AtomEven"]


@set_module("sisl.geom")
class AtomZ(AtomCategory):
    r""" Classify atoms based on atomic number

    Parameters
    ----------
    Z : int or array_like
       atomic number match for several values this is equivalent to AND
    """
    __slots__ = ("_Z",)

    def __init__(self, Z):
        if isiterable(Z):
            self._Z = set(Z)
        else:
            self._Z = set([Z])
        # using a sorted list ensures that we can compare
        super().__init__(f"Z={self._Z}")

    @_sanitize_loop
    def categorize(self, geometry, atoms=None):
        # _sanitize_loop will ensure that atoms will always be an integer
        if geometry.atoms.Z[atoms] in self._Z:
            return self
        return NullCategory()

    def __eq__(self, other):
        if isinstance(other, (list, tuple, np.ndarray)):
            # this *should* use the dispatch method for different
            # classes
            return super().__eq__(other)

        eq = self.__class__ is other.__class__
        if eq:
            return self._Z == other._Z
        return False


@set_module("sisl.geom")
class AtomOdd(AtomCategory):
    r""" Classify atoms based on indices (odd in this case)"""
    __slots__ = []

    def __init__(self):
        super().__init__("odd")

    @_sanitize_loop
    def categorize(self, geometry, atoms=None):
        # _sanitize_loop will ensure that atoms will always be an integer
        if atoms % 2 == 1:
            return self
        return NullClass()

    def __eq__(self, other):
        return self.__class__ is other.__class__


@set_module("sisl.geom")
class AtomEven(AtomCategory):
    r""" Classify atoms based on indices (even in this case)"""
    __slots__ = []

    def __init__(self):
        super().__init__("even")

    @_sanitize_loop
    def categorize(self, geometry, atoms):
        # _sanitize_loop will ensure that atoms will always be an integer
        if atoms % 2 == 0:
            return self
        return NullClass()

    def __eq__(self, other):
        return self.__class__ is other.__class__
