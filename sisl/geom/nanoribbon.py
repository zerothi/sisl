import numpy as np

from sisl import geom, Atom, Geometry

__all__ = ['nanoribbon', 'graphene_nanoribbon', 'agnr', 'zgnr']


def nanoribbon(bond, atom, width, kind='armchair'):
    r""" Construction of a nanoribbon unit cell of type armchair or zigzag.
    The geometry is oriented along the x-axis.

    Parameters
    ----------
    bond : float
       bond length between atoms in the honeycomb lattice
    atom : Atom
       atom (or atoms) in the honeycomb lattice
    width : int
       number of atoms in the transverse direction
    kind : {'armchair', 'zigzag'}
       type of ribbon

    See Also
    --------
    honeycomb : honeycomb lattices
    graphene : graphene geometry
    graphene_nanoribbon : graphene nanoribbon
    agnr : armchair graphene nanoribbon
    zgnr : zigzag graphene nanoribbon
    """
    if not isinstance(bond, float):
        raise ValueError("nanoribbon: bond needs to be a float!")
    if not isinstance(atom, (Atom, list, tuple)):
        raise ValueError("nanoribbon: atom needs to be an instance of Atom (or list of Atoms)!")
    if not isinstance(width, int):
        raise ValueError("nanoribbon: the width needs to be a postive integer!")
    # Width characterization
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
        raise ValueError("nanoribbon: kind must be armchair or zigzag")

    # Separate ribbons along y-axis
    ribbon.cell[1, 1] += 10.

    # Movie inside unit cell
    ribbon = ribbon.move([-np.min(ribbon.xyz[:, 0]), -np.min(ribbon.xyz[:, 1]), 0])

    return ribbon


def graphene_nanoribbon(width, bond=1.42, atom=None, kind='armchair'):
    r""" Construction of a graphene nanoribbon

    Parameters
    ----------
    width : int
       number of atoms in the transverse direction
    bond : float, optional
       CC bond length. Defaults to 1.42
    atom : Atom, optional
       atom (or atoms) in the honeycomb lattice. Defaults to `Atom(6)`
    kind : {'armchair', 'zigzag'}
       type of ribbon

    See Also
    --------
    honeycomb : honeycomb lattices
    graphene : graphene geometry
    nanoribbon : honeycomb nanoribbon
    agnr : armchair graphene nanoribbon
    zgnr : zigzag graphene nanoribbon
    """
    if atom is None:
        atom = Atom(Z=6, R=bond * 1.01)
    return nanoribbon(bond, atom, width, kind=kind)


def agnr(width, bond=1.42, atom=None):
    r""" Construction of an armchair graphene nanoribbon

    Parameters
    ----------
    width : int
       number of atoms in the transverse direction
    bond : float, optional
       CC bond length. Defaults to 1.42
    atom : Atom, optional
       atom (or atoms) in the honeycomb lattice. Defaults to `Atom(6)`

    See Also
    --------
    honeycomb : honeycomb lattices
    graphene : graphene geometry
    nanoribbon : honeycomb nanoribbon
    graphene_nanoribbon : graphene nanoribbon
    zgnr : zigzag graphene nanoribbon
    """
    return graphene_nanoribbon(width, bond, atom, kind='armchair')


def zgnr(width, bond=1.42, atom=None):
    r""" Construction of a zigzag graphene nanoribbon

    Parameters
    ----------
    width : int
       number of atoms in the transverse direction
    bond : float, optional
       CC bond length. Defaults to 1.42
    atom : Atom, optional
       atom (or atoms) in the honeycomb lattice. Defaults to `Atom(6)`

    See Also
    --------
    honeycomb : honeycomb lattices
    graphene : graphene geometry
    nanoribbon : honeycomb nanoribbon
    graphene_nanoribbon : graphene nanoribbon
    agnr : armchair graphene nanoribbon
    """
    return graphene_nanoribbon(width, bond, atom, kind='zigzag')
