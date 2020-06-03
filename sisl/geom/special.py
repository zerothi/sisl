import numpy as np

from sisl._internal import set_module
from sisl import Atom, Geometry, SuperCell

__all__ = ['diamond']


@set_module("sisl.geom")
def diamond(alat=3.57, atoms=None):
    """ Diamond lattice with 2 atoms in the unitcell

    Parameters
    ----------
    alat : float
        the lattice constant for the diamond
    atoms : Atom, optional
        atom in the lattice, may be one or two atoms. Default is Carbon
    """
    dist = alat * 3. ** .5 / 4
    if atoms is None:
        atoms = Atom(Z=6, R=dist * 1.01)
    sc = SuperCell(np.array([[0, 1, 1],
                             [1, 0, 1],
                             [1, 1, 0]], np.float64) * alat / 2,
                   nsc=[3, 3, 3])
    dia = Geometry(np.array([[0, 0, 0], [1, 1, 1]], np.float64) * alat / 4,
                   atoms, sc=sc)
    return dia
