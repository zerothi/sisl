from numbers import Integral
import numpy as np
from math import acos, pi

from sisl._internal import set_module
from sisl import geom, Atom, Cuboid

__all__ = ['bilayer']


@set_module("sisl.geom")
def bilayer(bond=1.42, bottom_atoms=None, top_atoms=None, stacking='AB',
            twist=(0, 0), separation=3.35, ret_angle=False, layer='both'):
    r""" Commensurate unit cell of a hexagonal bilayer structure, possibly with a twist angle.

    This routine follows the prescription of twisted bilayer graphene found in [1]_.

    Notes
    -----
    This routine may change in the future to force some of the arguments.

    Parameters
    ----------
    bond : float, optional
       bond length between atoms in the honeycomb lattice
    bottom_atoms : Atom, optional
       atom (or atoms) in the bottom layer. Defaults to ``Atom(6)``
    top_atoms : Atom, optional
       atom (or atoms) in the top layer, defaults to `bottom_atom`
    stacking : {'AB', 'AA', 'BA'}
       stacking sequence of the bilayer, where XY means that site X in bottom layer coincides with site Y in top layer
    twist : tuple of int, optional
       integer coordinates (m, n) defining a commensurate twist angle
    separation : float, optional
       distance between the two layers
    ret_angle : bool, optional
       return the twist angle (in degrees) in addition to the geometry instance
    layer : {'both', 'bottom', 'top'}
       control which layer(s) to return

    See Also
    --------
    honeycomb: honeycomb lattices
    graphene: graphene geometry

    References
    ----------
    .. [1] G. Trambly de Laissardiere, D. Mayou, L. Magaud, "Localization of Dirac Electrons in Rotated Graphene Bilayers", Nano Letts. 10, 804-808 (2010)
    """
    if bottom_atoms is None:
        bottom_atoms = top_atoms
    if bottom_atoms is None:
        bottom_atoms = Atom(Z=6, R=bond * 1.01)
    if top_atoms is None:
        top_atoms = bottom_atoms

    # Construct two layers
    bottom = geom.honeycomb(bond, bottom_atoms)
    top = geom.honeycomb(bond, top_atoms)
    ref_cell = bottom.cell.copy()

    stacking = stacking.lower()
    if stacking == 'aa':
        top = top.move([0, 0, separation])
    elif stacking == 'ab':
        top = top.move([-bond, 0, separation])
    elif stacking == 'ba':
        top = top.move([bond, 0, separation])
    else:
        raise ValueError("bilayer: stacking must be one of {AA, AB, BA}")

    # Compute twist angle
    m, n = twist
    m, n = abs(m), abs(n)
    if m > n:
        # Set m as the smaller integer
        m, n = n, m

    if not (isinstance(n, Integral) and isinstance(m, Integral)):
        raise ValueError("bilayer: twist coordinates need to be integers!")

    if m == n:
        # No twisting
        theta = 0
        rep = 1
        natoms = 2
    else:
        # Twisting
        cos_theta = (n ** 2 + 4 * n * m + m ** 2) / (2 * (n ** 2 + n * m + m ** 2))
        theta = acos(cos_theta) * 180 / pi
        rep = 4 * (n + m)
        natoms = 2 * (n ** 2 + n * m + m ** 2)

    if rep > 1:
        # Set origo through an A atom near the middle of the geometry
        align_vec = - rep * (ref_cell[0] + ref_cell[1]) / 2

        bottom = (bottom
                  .tile(rep, axis=0)
                  .tile(rep, axis=1)
                  .move(align_vec))

        # Set new lattice vectors
        bottom.cell[0] = n * ref_cell[0] + m * ref_cell[1]
        bottom.cell[1] = -m * ref_cell[0] + (n + m) * ref_cell[1]

        # Remove atoms outside cell
        cell_box = Cuboid(bottom.cell, center=[- bond * 1e-4] * 3)

        # Reduce atoms in bottom
        inside_idx = cell_box.within_index(bottom.xyz)
        bottom = bottom.sub(inside_idx)

        # Rotate top layer around A atom in bottom layer
        top = (top
               .tile(rep, axis=0)
               .tile(rep, axis=1)
               .move(align_vec)
               .rotate(theta, [0, 0, 1]))

        inside_idx = cell_box.within_index(top.xyz)
        top = top.sub(inside_idx)

        # Ensure the cells are commensurate
        top.cell[:] = bottom.cell[:]

    # Which layers to be returned
    layer = layer.lower()
    if layer == 'bottom':
        bilayer = bottom
    elif layer == 'top':
        bilayer = top
    elif layer == 'both':
        bilayer = bottom.add(top)
        natoms *= 2
    else:
        raise ValueError("bilayer: layer must be one of {both, bottom, top}")

    if rep > 1:
        # Rotate and shift unit cell back
        fxyz_min = bilayer.fxyz.min(axis=0)
        fxyz_min[2] = 0.
        # This is a small hack since rotate is not numerically
        # precise.
        # TODO We could consider using mpfmath in Quaternion for increased
        # precision...
        fxyz_min[np.fabs(fxyz_min) > 1.e-7] *= 0.49
        offset = fxyz_min.dot(bilayer.cell)
        vec = bilayer.cell[0] + bilayer.cell[1]
        vec_costh = vec[0] / vec.dot(vec) ** 0.5
        vec_th = acos(vec_costh) * 180 / pi
        bilayer = bilayer.move(-offset).rotate(vec_th, [0, 0, 1])

    # Sanity check
    assert len(bilayer) == natoms

    if ret_angle:
        return bilayer, theta
    else:
        return bilayer
