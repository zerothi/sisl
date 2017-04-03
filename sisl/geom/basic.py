"""
Helper functions for returning special geometries often encountered
"""
from __future__ import print_function, division

import numpy as np

from sisl import Atom, Geometry, SuperCell

__all__ = ['sc', 'bcc', 'fcc', 'hcp']

# A few needed variables
_s30 = 1 / 2
_s60 = 3 ** .5 / 2
_s45 = 1 / 2 ** .5
_c30 = _s60
_c60 = _s30
_c45 = _s45
_t30 = 1 / 3 ** .5
_t45 = 1.
_t60 = 3 ** .5


def sc(alat, atom):
    """
    Returns a Simple cubic lattice (1 atom)
    """
    sc = SuperCell(np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]], np.float64) * alat,
                   nsc=[3, 3, 3])
    g = Geometry([0, 0, 0], atom, sc=sc)
    return g


def bcc(alat, atom, orthogonal=False):
    """
    Returns a BCC lattice (1 atom)
    """
    if orthogonal:
        sc = SuperCell(np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]], np.float64) * alat,
                       nsc=[3, 3, 3])
        ah = alat / 2
        g = Geometry([[0, 0, 0], [ah, ah, ah]], atom, sc=sc)
    else:
        sc = SuperCell(np.array([[1, 1, 1],
                                 [1, -1, 1],
                                 [1, 1, -1]], np.float64) * alat / 2,
                       nsc=[3, 3, 3])
        g = Geometry([0, 0, 0], atom, sc=sc)
    return g


def fcc(alat, atom, orthogonal=False):
    """
    Returns a geometry with the FCC crystal structure (1 atom)
    """
    if orthogonal:
        sc = SuperCell(np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]], np.float64) * alat,
                       nsc=[3, 3, 3])
        ah = alat / 2
        g = Geometry([[0, 0, 0], [ah, ah, 0],
                      [ah, 0, ah], [0, ah, ah]], atom, sc=sc)
    else:
        sc = SuperCell(np.array([[0, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 0]], np.float64) * alat / 2,
                       nsc=[3, 3, 3])
        g = Geometry([0, 0, 0], atom, sc=sc)
    return g


def hcp(a, atom, coa=1.63333, orthogonal=False):
    """
    Returns a geometry with the HCP crystal structure (1 atom)
    """
    # height of hcp structure
    c = a * coa
    a2sq = a / 2 ** .5
    if orthogonal:
        sc = SuperCell([[a + a * _c60 * 2, 0, 0],
                        [0, a * _c30 * 2, 0],
                        [0, 0, c / 2]])
        gt = Geometry([[0, 0, 0],
                       [a, 0, 0],
                       [a * _s30, a * _c30, 0],
                       [a * (1 + _s30), a * _c30, 0]], atom, sc=sc)
        # Create the rotated one on top
        gr = gt.copy()
        # mirror structure
        gr.xyz[0, 1] += sc.cell[1, 1]
        gr.xyz[1, 1] += sc.cell[1, 1]
        gr = gr.translate(-np.amin(gr.xyz, axis=0))
        # Now displace to get the correct offset
        gr = gr.translate([0, a * _s30 / 2, 0])
        g = gt.append(gr, 2)
    else:
        sc = SuperCell([a, a, c, 90, 90, 60],
                       nsc=[3, 3, 3])
        g = Geometry(
            [[0, 0, 0], [a2sq * _c30, a2sq * _s30, c / 2]], atom, sc=sc)
    return g
