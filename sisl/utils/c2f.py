"""
Utilities for converting C stuff to Fortran stuff.
"""

from numbers import Integral

from numpy import ndarray


def index_c2f(lst):
    """ Returns the same object with all elements subtracted by 1 """

    # Initial check which is very easy
    if isinstance(lst, (Integral, ndarray)):
        return lst - 1

    # We try and ensure that we return the same
    # type as it already is...
    # This recursive scheme should ensure pretty much
    # all different types of constructs return
    # all elements correctly.
    cls = lst.__class__
    return cls([index_c2f(el) for el in lst])
