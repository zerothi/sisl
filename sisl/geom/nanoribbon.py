from numbers import Integral
import numpy as np

from sisl._internal import set_module
from sisl import geom, Atom, Geometry

__all__ = ['nanoribbon', 'graphene_nanoribbon', 'agnr', 'zgnr']


@set_module("sisl.geom")
def nanoribbon(bond, atoms, width, kind='armchair'):
    r""" Construction of a nanoribbon unit cell of type armchair or zigzag.

    The geometry is oriented along the :math:`x` axis.

    Parameters
    ----------
    bond : float
       bond length between atoms in the honeycomb lattice
    atoms : Atom
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
    if not isinstance(width, Integral):
        raise ValueError(f"nanoribbon: width needs to be a postive integer ({width})!")

    # Width characterization
    width = max(width, 1)
    n, m = width // 2, width % 2

    ribbon = geom.honeycomb(bond, atoms, orthogonal=True)

    kind = kind.lower()
    if kind == "armchair":
        # Construct armchair GNR
        if m == 1:
            ribbon = ribbon.repeat(n + 1, 1)
            ribbon = ribbon.remove(3 * (n + 1)).remove(0)
        else:
            ribbon = ribbon.repeat(n, 1)

    elif kind == "zigzag":
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
        ribbon.cell[:, :] = np.diag([ribbon.cell[1, 0], -ribbon.cell[0, 1], ribbon.cell[2, 2]])
        # Sort along x, then y
        ribbon = ribbon.sort(axis=(0, 1))

    else:
        raise ValueError(f"nanoribbon: kind must be armchair or zigzag ({kind})")

    # Separate ribbons along y-axis
    ribbon.cell[1, 1] += 20.

    # Move inside unit cell
    xyz = ribbon.xyz.min(axis=0) * [1, 1, 0]

    return ribbon.move(-xyz + [0, 10, 0])


@set_module("sisl.geom")
def graphene_nanoribbon(width, bond=1.42, atoms=None, kind='armchair'):
    r""" Construction of a graphene nanoribbon

    Parameters
    ----------
    width : int
       number of atoms in the transverse direction
    bond : float, optional
       C-C bond length. Defaults to 1.42
    atoms : Atom, optional
       atom (or atoms) in the honeycomb lattice. Defaults to ``Atom(6)``
    kind : {'armchair', 'zigzag'}
       type of ribbon

    See Also
    --------
    honeycomb : honeycomb lattices
    graphene : graphene geometry
    nanoribbon : honeycomb nanoribbon (used for this method)
    agnr : armchair graphene nanoribbon
    zgnr : zigzag graphene nanoribbon
    """
    if atoms is None:
        atoms = Atom(Z=6, R=bond * 1.01)
    return nanoribbon(bond, atoms, width, kind=kind)


@set_module("sisl.geom")
def agnr(width, bond=1.42, atoms=None):
    r""" Construction of an armchair graphene nanoribbon

    Parameters
    ----------
    width : int
       number of atoms in the transverse direction
    bond : float, optional
       C-C bond length. Defaults to 1.42
    atoms : Atom, optional
       atom (or atoms) in the honeycomb lattice. Defaults to ``Atom(6)``

    See Also
    --------
    honeycomb : honeycomb lattices
    graphene : graphene geometry
    nanoribbon : honeycomb nanoribbon
    graphene_nanoribbon : graphene nanoribbon
    zgnr : zigzag graphene nanoribbon
    """
    return graphene_nanoribbon(width, bond, atoms, kind='armchair')


@set_module("sisl.geom")
def zgnr(width, bond=1.42, atoms=None):
    r""" Construction of a zigzag graphene nanoribbon

    Parameters
    ----------
    width : int
       number of atoms in the transverse direction
    bond : float, optional
       C-C bond length. Defaults to 1.42
    atoms : Atom, optional
       atom (or atoms) in the honeycomb lattice. Defaults to ``Atom(6)``

    See Also
    --------
    honeycomb : honeycomb lattices
    graphene : graphene geometry
    nanoribbon : honeycomb nanoribbon
    graphene_nanoribbon : graphene nanoribbon
    agnr : armchair graphene nanoribbon
    """
    return graphene_nanoribbon(width, bond, atoms, kind='zigzag')
