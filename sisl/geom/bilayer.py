import numpy as np

from sisl import geom, Atom, Cuboid

__all__ = ['bilayer']


def bilayer(bond=1.42, bottom_atom=None, top_atom=None, stacking='AB',
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
    bottom_atom : Atom, optional
       atom (or atoms) in the bottom layer. Defaults to ``Atom(6)``
    top_atom : Atom, optional
       atom (or atoms) in the top layer, defaults to `bottom_atom`
    stacking : {'AB', 'AA', 'BA'}
       stacking sequence of the bilayer, where XY means that site X in bottom layer coincides with site Y in top layer
    twist : tuple of int, optional
       integer coordinates (m, n) defining a commensurate twist angle
    separation : float, optional
       distance between the two layers (in Angstrom)
    ret_angle : bool, optional
       return the twist angle (in degrees) in addition to the geometry instance
    layer : {'both', 'bottom', 'top'}
       control which layer(s) to return

    References
    ----------
    .. [1] G. Trambly de Laissardiere, D. Mayou, L. Magaud, "Localization of Dirac Electrons in Rotated Graphene Bilayers", Nano Letts. 10, 804-808 (2010)
    """
    if bottom_atom is None:
        bottom_atom = top_atom
    if bottom_atom is None:
        bottom_atom = Atom(Z=6, R=bond * 1.01)
    if top_atom is None:
        top_atom = bottom_atom

    # Construct two layers
    bottom = geom.honeycomb(bond, bottom_atom)
    top = geom.honeycomb(bond, top_atom)
    ref_cell = bottom.cell.copy()

    if stacking.lower() == 'aa':
        top = top.move([0, 0, separation])
    elif stacking.lower() == 'ab':
        top = top.move([-bond, 0, separation])
    elif stacking.lower() == 'ba':
        top = top.move([bond, 0, separation])

    # Compute twist angle
    m, n = twist
    m, n = abs(m), abs(n)
    if m > n:
        # Set m as the smaller integer
        m, n = n, m

    if not (isinstance(n, int) and isinstance(m, int)):
        raise ValueError("bilayer: twist coordinates need to be integers!")

    if m == n:
        # No twisting
        theta = 0
        rep = 1
        natoms = 2
    else:
        # Twisting
        cos_theta = (n ** 2 + 4 * n * m + m ** 2) / (2 * (n ** 2 + n * m + m ** 2))
        theta = np.arccos(cos_theta) * 180 / np.pi
        rep = 4 * (n + m)
        natoms = 2 * (n ** 2 + n * m + m ** 2)

    if rep > 1:
        # Set origo through an A atom near the middle of the geometry
        bottom = bottom.tile(rep, axis=0).tile(rep, axis=1)
        top = top.tile(rep, axis=0).tile(rep, axis=1)
        tvec = rep * (ref_cell[0] + ref_cell[1]) / 2
        bottom = bottom.move(-tvec)
        top = top.move(-tvec)

        # Set new lattice vectors
        bottom.cell[0] = n * ref_cell[0] + m * ref_cell[1]
        bottom.cell[1] = -m * ref_cell[0] + (n + m) * ref_cell[1]

        # Rotate top layer around A atom in bottom layer
        top = top.rotate(theta, [0, 0, 1])

        top.cell[:] = bottom.cell[:]

    # Which layers to be returned
    if layer.lower() == 'bottom':
        bilayer = bottom
    elif layer.lower() == 'top':
        bilayer = top
    else:
        bilayer = bottom.add(top)
        natoms *= 2

    if rep > 1:
        # Remove atoms outside cell
        cell_box = Cuboid(bilayer.cell)
        cell_box.set_center([-0.0001] * 3)
        inside_idx = cell_box.within_index(bilayer.xyz)
        bilayer = bilayer.sub(inside_idx)

        # Rotate whole cell
        vec = bilayer.cell[0] + bilayer.cell[1]
        vec_costh = vec[0] / vec.dot(vec) ** 0.5
        vec_th = np.arccos(vec_costh) * 180 / np.pi
        bilayer = bilayer.rotate(vec_th, [0, 0, 1])

    # Sanity check
    assert len(bilayer) == natoms

    if ret_angle:
        return bilayer, theta
    else:
        return bilayer
