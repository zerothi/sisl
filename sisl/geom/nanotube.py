"""
Helper functions for returning a nanotube
"""
from __future__ import print_function, division

import numpy as np

from sisl import Atom, Geometry, SuperCell

__all__ = ['nanotube']


def nanotube(bond, atom=None, chirality=(1, 1)):
    """
    Create a nano-tube geometry

    This routine is implemented as in ``ASE`` with few changes.
    
    Parameters
    ----------
    bond: float
       length between atoms in nano-tube
    atom: Atom(6)
       nanotube atoms
    chirality: (int, int)
       chirality of nanotube (n, m)
    """
    if atom is None:
        atom = Atom[6]

    # Correct the input...
    n, m = chirality
    if n < m:
        m, n = n, m
        sign = -1
    else:
        sign = 1

    sq3 = 3.0 ** .5
    a = sq3 * bond
    l2 = n * n + m * m + n * m
    l = l2 ** .5

    def gcd(a, b):
        while a != 0:
            a, b = b % a, a
        return b

    nd = gcd(n, m)
    if (n - m) % (3 * nd) == 0:
        ndr = 3 * nd
    else:
        ndr = nd

    nr = (2 * m + n) // ndr
    ns = -(2 * n + m) // ndr
    nn = 2 * l2 // ndr

    ichk = 0
    if nr == 0:
        n60 = 1
    else:
        n60 = nr * 4

    absn = abs(n60)
    nnp = []
    nnq = []
    for i in range(-absn, absn + 1):
        for j in range(-absn, absn + 1):
            j2 = nr * j - ns * i
            if j2 == 1:
                j1 = m * i - n * j
                if j1 > 0 and j1 < nn:
                    ichk += 1
                    nnp.append(i)
                    nnq.append(j)

    if ichk == 0:
        raise RuntimeError('not found p, q strange!!')
    if ichk >= 2:
        raise RuntimeError('more than 1 pair p, q strange!!')

    nnnp = nnp[0]
    nnnq = nnq[0]

    lp = nnnp * nnnp + nnnq * nnnq + nnnp * nnnq
    r = a * lp ** .5
    c = a * l
    t = sq3 * c / ndr

    rs = c / (2.0 * np.pi)

    q1 = np.arctan((sq3 * m) / (2 * n + m))
    q2 = np.arctan((sq3 * nnnq) / (2 * nnnp + nnnq))
    q3 = q1 - q2

    q4 = 2.0 * np.pi / nn
    q5 = bond * np.cos((np.pi / 6.0) - q1) / c * 2.0 * np.pi

    h1 = abs(t) / abs(np.sin(q3))
    h2 = bond * np.sin((np.pi / 6.0) - q1)

    xyz = np.empty([nn*2, 3], np.float64)
    for i in range(nn):
        ix = i * 2
        
        k = np.floor(i * abs(r) / h1)
        xyz[ix,0] = rs * np.cos(i * q4)
        xyz[ix,1] = rs * np.sin(i * q4)
        z = (i * abs(r) - k * h1) * np.sin(q3)
        kk2 = abs(np.floor((z + 0.0001) / t))
        if z >= t - 0.0001:
            z -= t * kk2
        elif z < 0:
            z += t * kk2
        xyz[ix,2] = z * sign

        # Next
        ix += 1
        xyz[ix, 0] = rs * np.cos(i * q4 + q5)
        xyz[ix, 1] = rs * np.sin(i * q4 + q5)
        z = (i * abs(r) - k * h1) * np.sin(q3) - h2
        if z >= 0 and z < t:
            pass 
        else:
            z -= h1 * np.sin(q3)
            kk = abs(np.floor(z / t))
            if z >= t - 0.0001:
                z -= t * kk
            elif z < 0:
                z += t * kk
        xyz[ix, 2] = z * sign

    # Sort the atomic coordinates according to z
    idx = np.argsort(xyz[:,2])
    xyz = xyz[idx,:]

    sc = SuperCell([rs * 4, rs * 4, t], nsc=[1,1,3])

    geom = Geometry(xyz, atom, sc=sc)
    # Return a geometry with the first atom at (0,0,0)
    return geom.translate(-np.amin(geom.xyz, axis=0))
