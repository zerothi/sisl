# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np

from sisl._internal import set_module
from sisl import Atom, Geometry, SuperCell

__all__ = ['honeycomb', 'graphene', 'fcc100', 'fcc110', 'fcc111']


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
    atoms : Atom, optional
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


def _finish_slab(g, size, vacuum):
    g = g.repeat(size[1], 1).repeat(size[0], 0)
    if vacuum is not None:
        g.cell[2, 2] += vacuum
        g.set_nsc([3, 3, 1])
    else:
        g.set_nsc([3, 3, 3])
    if np.all(g.maxR(True) > 0.):
        g.optimize_nsc()
    return g


def fcc100(alat, atoms, size=(1, 1, 2), vacuum=None):
    """ Face-centered cubic (100) surface

    Parameters
    ----------
    alat : float
        lattice constant of the fcc crystal
    atoms : Atom
        the atom that the crystal consists of
    size : 3-array, optional
        slab size along the lattice vectors
    vacuum : float, optional
        distance added to the third lattice vector to separate
        the slab from its periodic images

    See Also
    --------
    fcc110 : construction of fcc(110) surface
    fcc111 : construction of fcc(111) surface
    """
    sc = SuperCell(np.array([np.sqrt(0.5), np.sqrt(0.5), 0.5]) * alat)
    # A-layer
    g = Geometry([0, 0, 0], atoms=atoms, sc=sc).tile(size[2], 2)
    # B-layer
    g.xyz[1::2] += (sc.cell[0] + sc.cell[1]) / 2
    g = _finish_slab(g, size, vacuum)
    return g


def fcc110(alat, atoms, size=(1, 1, 2), vacuum=None):
    """ Face-centered cubic (110) surface

    Parameters
    ----------
    alat : float
        lattice constant of the fcc crystal
    atoms : Atom
        the atom that the crystal consists of
    size : 3-array, optional
        slab size along the lattice vectors
    vacuum : float, optional
        distance added to the third lattice vector to separate
        the slab from its periodic images

    See Also
    --------
    fcc100 : construction of fcc(100) surface
    fcc111 : construction of fcc(111) surface
    """
    sc = SuperCell(np.array([1., np.sqrt(0.5), np.sqrt(0.125)]) * alat)
    # A-layer
    g = Geometry([0, 0, 0], atoms=atoms, sc=sc).tile(size[2], 2)
    # B-layer
    g.xyz[1::2] += (sc.cell[0] + sc.cell[1]) / 2
    g = _finish_slab(g, size, vacuum)
    return g


def fcc111(alat, atoms, size=(1, 1, 3), vacuum=None, orthogonal=False):
    """ Face-centered cubic (111) surface

    Parameters
    ----------
    alat : float
        lattice constant of the fcc crystal
    atoms : Atom
        the atom that the crystal consists of
    size : 3-array, optional
        slab size along the lattice vectors
    vacuum : float, optional
        distance added to the third lattice vector to separate
        the slab from its periodic images
    orthogonal : bool, optional
        if True returns an orthogonal lattice

    See Also
    --------
    fcc100 : construction of fcc(100) surface
    fcc110 : construction of fcc(110) surface
    """
    if orthogonal:
        # 2-atom basis per layer
        sc = SuperCell(np.array([np.sqrt(0.5), np.sqrt(0.375) * 2, 1 / np.sqrt(3)]) * alat)
        # A-layer
        g = Geometry(np.array([[0, 0, 0],
                               [np.sqrt(0.125), np.sqrt(0.375), 0]]) * alat,
                     atoms=atoms, sc=sc).tile(size[2], 2)
        vec = 1.5 * sc.cell[0] + sc.cell[1] / 2
        # B-layer
        g.xyz[2::6] += vec / 3
        g.xyz[3::6] += vec / 3 - sc.cell[0]
        # C-layer
        g.xyz[4::6] += 2 * vec / 3 - sc.cell[0]
        g.xyz[5::6] += 2 * vec / 3 - sc.cell[0]
    else:
        # 1-atom basis per layer
        sc = SuperCell(np.array([[np.sqrt(0.5), 0, 0],
                                 [np.sqrt(0.125), np.sqrt(0.375), 0],
                                 [0, 0, 1 / np.sqrt(3)]]) * alat)
        # A-layer
        g = Geometry([0, 0, 0], atoms=atoms, sc=sc).tile(size[2], 2)
        # B-layer
        g.xyz[1::3] += sc.cell[0] / 3 + sc.cell[1] / 3
        # C-layer
        g.xyz[2::3] += -sc.cell[0] / 3 + 2 * sc.cell[1] / 3
    g = _finish_slab(g, size, vacuum)
    return g
