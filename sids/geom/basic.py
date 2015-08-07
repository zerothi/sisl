"""
Helper functions for returning special geometries often encountered
"""
from __future__ import print_function, division

import numpy as np

from sids import Atom, Geometry, SuperCell

__all__ = ['bcc','fcc']


def bcc(alat,A):
    """
    Returns a BCC lattice (1 atom)
    """
    sc = SuperCell(np.array([[1, 1, 1],
                             [1,-1, 1],
                             [1, 1,-1]],np.float64) * alat/2,
                   nsc=[3,3,3])
    bcc = Geometry([0,0,0], atoms=A , sc=sc)
    return bcc


def fcc(alat,A):
    """
    Returns a geometry with the FCC crystal structure (1 atom)
    """
    sc = SuperCell(np.array([[0,1,1],
                             [1,0,1],
                             [1,1,0]],np.float64) * alat/2,
                   nsc=[3,3,3])
    fcc = Geometry([0,0,0], atoms=A, sc=sc)
    return fcc


if __name__ == "__main__":
    pass
    
