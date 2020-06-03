import numpy as np

from sisl._internal import set_module
from sisl import Geometry, SuperCell

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


@set_module("sisl.geom")
def sc(alat, atom):
    """ Simple cubic lattice with 1 atom

    Parameters
    ----------
    alat : float
        lattice parameter
    atom : Atom
        the atom in the SC lattice
    """
    sc = SuperCell(np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]], np.float64) * alat)
    g = Geometry([0, 0, 0], atom, sc=sc)
    if np.all(g.maxR(True) > 0.):
        g.optimize_nsc()
    return g


@set_module("sisl.geom")
def bcc(alat, atoms, orthogonal=False):
    """ Body centered cubic lattice with 1 (non-orthogonal) or 2 atoms (orthogonal)

    Parameters
    ----------
    alat : float
        lattice parameter
    atoms : Atom
        the atom(s) in the BCC lattice
    orthogonal : bool, optional
        whether the lattice is orthogonal (2 atoms)
    """
    if orthogonal:
        sc = SuperCell(np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]], np.float64) * alat)
        ah = alat / 2
        g = Geometry([[0, 0, 0], [ah, ah, ah]], atoms, sc=sc)
    else:
        sc = SuperCell(np.array([[-1, 1, 1],
                                 [1, -1, 1],
                                 [1, 1, -1]], np.float64) * alat / 2)
        g = Geometry([0, 0, 0], atoms, sc=sc)
    if np.all(g.maxR(True) > 0.):
        g.optimize_nsc()
    return g


@set_module("sisl.geom")
def fcc(alat, atoms, orthogonal=False):
    """ Face centered cubic lattice with 1 (non-orthogonal) or 4 atoms (orthogonal)

    Parameters
    ----------
    alat : float
        lattice parameter
    atoms : Atom
        the atom(s) in the FCC lattice
    orthogonal : bool, optional
        whether the lattice is orthogonal (4 atoms)
    """
    if orthogonal:
        sc = SuperCell(np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]], np.float64) * alat)
        ah = alat / 2
        g = Geometry([[0, 0, 0], [ah, ah, 0],
                      [ah, 0, ah], [0, ah, ah]], atoms, sc=sc)
    else:
        sc = SuperCell(np.array([[0, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 0]], np.float64) * alat / 2)
        g = Geometry([0, 0, 0], atoms, sc=sc)
    if np.all(g.maxR(True) > 0.):
        g.optimize_nsc()
    return g


@set_module("sisl.geom")
def hcp(a, atoms, coa=1.63333, orthogonal=False):
    """ Hexagonal closed packed lattice with 2 (non-orthogonal) or 4 atoms (orthogonal)

    Parameters
    ----------
    a : float
        lattice parameter for 1st and 2nd lattice vectors
    atoms : Atom
        the atom(s) in the HCP lattice
    coa : float, optional
        c over a parameter where c is the 3rd lattice vector length
    orthogonal : bool, optional
        whether the lattice is orthogonal (4 atoms)
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
                       [a * (1 + _s30), a * _c30, 0]], atoms, sc=sc)
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
        sc = SuperCell([a, a, c, 90, 90, 60])
        g = Geometry([[0, 0, 0], [a2sq * _c30, a2sq * _s30, c / 2]],
                     atoms, sc=sc)
    if np.all(g.maxR(True) > 0.):
        g.optimize_nsc()
    return g
