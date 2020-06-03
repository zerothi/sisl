import numpy as np

from sisl._internal import set_module
from sisl import Atom, Geometry, SuperCell

__all__ = ['honeycomb', 'graphene']


@set_module("sisl.geom")
def honeycomb(bond, atoms, orthogonal=False):
    """ Honeycomb lattice with 2 or 4 atoms per unit-cell, latter orthogonal cell

    This enables creating BN lattices with ease, or graphene lattices.

    Parameters
    ----------
    bond : float
        bond length between atoms (*not* lattice constant)
    atoms : Atom
        the atom (or atoms) that the honeycomb lattice consists of
    orthogonal : bool, optional
        if True returns an orthogonal lattice

    See Also
    --------
    graphene: the equivalent of this, but with default of Carbon atoms
    bilayer: create bilayer honeycomb lattices
    """
    sq3h = 3.**.5 * 0.5
    if orthogonal:
        sc = SuperCell(np.array([[3., 0., 0.],
                                 [0., 2 * sq3h, 0.],
                                 [0., 0., 10.]], np.float64) * bond, nsc=[3, 3, 1])
        g = Geometry(np.array([[0., 0., 0.],
                               [0.5, sq3h, 0.],
                               [1.5, sq3h, 0.],
                               [2., 0., 0.]], np.float64) * bond,
                     atoms, sc=sc)
    else:
        sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                 [1.5, -sq3h, 0.],
                                 [0., 0., 10.]], np.float64) * bond, nsc=[3, 3, 1])
        g = Geometry(np.array([[0., 0., 0.],
                               [1., 0., 0.]], np.float64) * bond,
                     atoms, sc=sc)
    return g


def graphene(bond=1.42, atoms=None, orthogonal=False):
    """ Graphene lattice with 2 or 4 atoms per unit-cell, latter orthogonal cell

    Parameters
    ----------
    bond : float
        bond length between atoms (*not* lattice constant)
    atom : Atom, optional
        the atom (or atoms) that the honeycomb lattice consists of.
        Default to Carbon atom.
    orthogonal : bool, optional
        if True returns an orthogonal lattice

    See Also
    --------
    honeycomb: the equivalent of this, but with non-default atoms
    bilayer: create bilayer honeycomb lattices
    """
    if atoms is None:
        return honeycomb(bond, Atom(Z=6, R=bond * 1.01), orthogonal)
    return honeycomb(bond, atoms, orthogonal)
