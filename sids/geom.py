"""
Helper functions for returning special geometries often encountered
"""
from __future__ import print_function, division

import numpy as np

from .atom import Atom
from .geometry import Geometry
from .supercell import SuperCell

__all__ = ['graphene','graphene_square','diamond']


def graphene(alat=1.42,C=None):
    """
    Returns a geometry with the graphene unit-cell (2-atoms)
    """
    sq3h  = 3.**.5 * 0.5
    if C is None:
        C = Atom(Z=6,R=alat * 1.01)
    sc = SuperCell(np.array([[1.5, sq3h,  0.],
                             [1.5,-sq3h,  0.],
                             [ 0.,   0., 10.]],np.float64) * alat, nsc=[3,3,1])
    gr = Geometry(np.array([[ 0., 0., 0.],
                            [ 1., 0., 0.]],np.float64) * alat,
                  atoms=C, sc=sc)
    return gr


def graphene_square(alat=1.42,C=None):
    """
    Returns a geometry with the square graphene cell (4-atoms)
    """
    sq3h  = 3.**.5 * 0.5
    if C is None:
        C = Atom(Z=6,R=alat * 1.01)
    sc = SuperCell(np.array([[3.,     0.,  0.],
                             [0., 2*sq3h,  0.],
                             [0.,     0., 10.]],np.float64) * alat, nsc=[3,3,1])
    gr = Geometry(np.array([[ 0.,   0., 0.],
                            [ 2.,   0., 0.],
                            [0.5, sq3h, 0.],
                            [1.5, sq3h, 0.] ],np.float64) * alat,
                  atoms=C, sc=sc)
    return gr


def diamond(alat=3.57,C=None):
    """
    Returns a geometry with the diamond unit-cell (2-atoms)
    """
    dist = alat * 3. ** .5 / 4
    if C is None:
        C = Atom(Z=6,R=dist * 1.01)
    sc = SuperCell(np.array([[0,1,1],
                             [1,0,1],
                             [1,1,0]],np.float64) * alat/2,
                   nsc=[3,3,3])
    dia = Geometry(np.array([[0,0,0],[1,1,1]],np.float64) * alat/4,
                   atoms=C , sc=sc)
    return dia

