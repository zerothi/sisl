"""
Helper functions for returning special geometries often encountered
"""
from __future__ import print_function, division

import numpy as np

from sisl import Atom, Geometry, SuperCell

__all__ = ['honeycomb', 'graphene']


def honeycomb(alat, A, square=False):
    """
    Returns a honeycomb geometry with the graphene unit-cell (2 atoms)
    """
    sq3h = 3.**.5 * 0.5
    if square:
        sc = SuperCell(np.array([[3., 0., 0.],
                                 [0., 2 * sq3h, 0.],
                                 [0., 0., 10.]], np.float64) * alat, nsc=[3, 3, 1])
        g = Geometry(np.array([[0., 0., 0.],
                               [0.5, sq3h, 0.],
                               [1.5, sq3h, 0.],
                               [2., 0., 0.]], np.float64) * alat,
                     atoms=A, sc=sc)
    else:
        sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                 [1.5, -sq3h, 0.],
                                 [0., 0., 10.]], np.float64) * alat, nsc=[3, 3, 1])
        g = Geometry(np.array([[0., 0., 0.],
                               [1., 0., 0.]], np.float64) * alat,
                     atoms=A, sc=sc)
    return g


def graphene(alat=1.42, A=None, square=False):
    """
    Returns a geometry with the graphene unit-cell (2 atoms)
    """
    if A is None:
        return honeycomb(alat, Atom(Z=6, R=alat * 1.01), square)
    return honeycomb(alat, A, square)


if __name__ == "__main__":
    g = graphene()
    g = graphene(square=True)
