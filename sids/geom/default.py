"""
Helper functions for returning special geometries often encountered
"""
from __future__ import print_function, division

# The atom model
from .atom import Atom
from .geometry import Geometry

import numpy as np

def graphene(alat=1.42,orbs=1):
    """
    Returns a geometry with the graphene unit-cell (2-atoms)
    """
    sq3h  = 3.**.5 * 0.5
    C = Atom(Z=6,R=alat * 1.01,orbs=orbs)
    gr = Geometry(xa=np.array([[ 0., 0., 0.],
                               [ 1., 0., 0.]],np.float64) * alat,
                  cell=np.array([[1.5, sq3h,  0.],
                                 [1.5,-sq3h,  0.],
                                 [ 0.,   0., 10.]],np.float64) * alat,
                  atoms = C, nsc = [3,3,1])
    return gr

def diamond(alat=3.57,orbs=1):
    """
    Returns a geometry with the diamond unit-cell (2-atoms)
    """
    dist = alat * 3. **.5 / 4
    C = Atom(Z=6,R=dist * 1.01,orbs=2)
    dia = Geometry(cell=np.array([[0,1,1],
                                  [1,0,1],
                                  [1,1,0]],np.float64) * alat/2,
                   xyz = np.array([[0,0,0],[1,1,1]],np.float64)*alat/4,
                   atoms = C , nsc = [3,3,3])
    return dia

