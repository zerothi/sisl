import numpy as np

from math import acos, pi
from sisl import geom, Atom, Cuboid

__all__ = ['nanoribbon']


def nanoribbon(width, kind='armchair', bond=1.42, atom=None, sat_bond=1.09, sat_atom=None):
    r""" Construction of a nanoribbon unit cell of type armchair or zigzag.
    The geometry is oriented along the x-axis.

    Parameters
    ----------
    width : int
       number of atoms in the transverse direction
    kind : {'armchair', 'zigzag'}
       type of ribbon
    bond : float, optional
       bond length between atoms in the honeycomb lattice
    atom : Atom, optional
       atom (or atoms) in the honeycomb lattice. Defaults to ``Atom(6)``
    sat_bond : float, optional
       bond length to the edge saturation atoms
    sat_atom : Atom, optional
       atom (or atoms) for the edge saturation. If ``None`` no edge saturation is
       applied. Defaults to ``Atom(1)``

    See Also
    --------
    honeycomb: honeycomb lattices
    graphene: graphene geometry
    """
    if atom is None:
        atom = Atom(Z=6, R=bond * 1.01)
    if sat_atom is None:
        sat_atom = Atom(Z=1, R=sat_bond * 1.01)

    # Width characterization
    if not isinstance(width, int):
        raise ValueError("nanoribbon: the width needs to be a postive integer!")
    width = max(width, 1)
    n, m = width // 2, width % 2

    ribbon = geom.honeycomb(bond, atom, orthogonal=True)

    kind = kind.lower()
    if kind.startswith('a'):
        # Construct armchair GNR
        if m == 1:
            ribbon = ribbon.repeat(n + 1, 1)
            ribbon = ribbon.remove(3 * (n + 1)).remove(0)
        else:
            ribbon = ribbon.repeat(n, 1)
    elif kind.startswith('z'):
        # Construct zigzag GNR
        ribbon = ribbon.rotate(90, [0, 0, -1])
        if m == 1:
            ribbon = ribbon.tile(n + 1, 0)
            ribbon = ribbon.remove(-1).remove(-1)
        else:
            ribbon = ribbon.tile(n, 0)
        # Invert y-coordinates
        ribbon.xyz[:, 1] *= -1
        # Set lattice vectors strictly orthogonal
        ribbon.cell[:] = np.diag([ribbon.cell[1, 0], -ribbon.cell[0, 1], ribbon.cell[2, 2]])
        # Sort along x-axis
        ribbon = ribbon.sort()
    else:
        raise ValueError("ribbon: kind must be armchair or zigzag")

    # Separate ribbons along y-axis
    ribbon.cell[1, 1] += 10.

    # Place ribbon inside unit cell
    ribbon = ribbon.move([-np.min(ribbon.xyz[:, 0]), -np.min(ribbon.xyz[:, 1]), 0])

    return ribbon
