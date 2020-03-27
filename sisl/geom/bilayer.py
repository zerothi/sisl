import numpy as np

from sisl import Atom, geom, Cuboid

__all__ = ['bilayer']


def bilayer(bond=1.42, bottom_atom=None, top_atom=None, stacking='AB', twist=(0, 0), separation=3.35, ret_angle=False, layer='both'):
    """ Commensurate unit cell of a hexagonal bilayer structure, possibly with a twist angle.

    This routine follows the prescription of twisted bilayer graphene found in
    Laissardiere et al., Nano Lett. 10, 804-808 (2010).


    Parameters
    ----------
    bond : 1.42
       bond length (in Angstrom) between atoms in the honeycomb lattice
    bottom_atom : Atom(6)
       atom (or atoms) in the bottom layer
    top_atom : Atom(6)
       atom (or atoms) in the top layer
    stacking : {'AB', 'AA'}
       stacking sequence of the bilayer
    twist : (0, 0)
       integer coordinates (m, n) defining a commensurate twist angle
    separation : float
       distance between the two layers (in Angstrom)
    ret_angle : bool
       return the twist angle (in degrees) in addition to the geometry instance
    layer : {'both', 'bottom', 'top'}
       control which layer(s) to include in geometry
    """
    if bottom_atom is None:
        bottom_atom = Atom(Z=6, R=bond * 1.01)
    if top_atom is None:
        top_atom = Atom(Z=6, R=bond * 1.01)

    # Construct two layers
    bottom = geom.honeycomb(bond=bond, atom=bottom_atom)
    top = geom.honeycomb(bond=bond, atom=top_atom)
    honeycomb = bottom.copy()

    if stacking.lower() == 'aa':
        top = top.move([0, 0, separation])
    elif stacking.lower() == 'ab':
        top = top.move([bond, 0, separation])

    # Compute twist angle
    m, n = twist
    m, n = abs(m), abs(n)
    if m > n:
        # Set m as the smaller integer
        m, n = n, m

    if not (isinstance(n, int) and isinstance(m, int)):
        raise RuntimeError('twist coordinates need to be integers!')

    if m == n:
        # No twisting
        theta = 0
        rep = 1
        natoms = 2
    else:
        # Twisting
        cos_theta = (n ** 2 + 4 * n * m + m ** 2) / (2 * (n ** 2 + n * m + m ** 2) )
        theta = np.arccos(cos_theta) * 360 / (2 * np.pi)
        rep = 4 * (n + m)
        natoms = 2 * (n ** 2 + n * m + m ** 2)

    if rep > 1:
        # Set origo through an A atom near the middle of the geometry
        bottom = bottom.tile(rep, axis=0).tile(rep, axis=1)
        top = top.tile(rep, axis=0).tile(rep, axis=1)
        tvec = rep * (honeycomb.cell[0] + honeycomb.cell[1]) / 2
        bottom = bottom.move(-tvec)
        top = top.move(-tvec)

        # Set new lattice vectors
        bottom.cell[0] = n * honeycomb.cell[0] + m * honeycomb.cell[1]
        bottom.cell[1] = -m * honeycomb.cell[0] + (n + m) * honeycomb.cell[1]

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
        center = 1e-3 * np.array([1, 1, 1])
        cell_box = Cuboid(1.0 * bilayer.cell[:], center=center)
        inside_idx = cell_box.within_index(bilayer.xyz)
        bilayer = bilayer.sub(inside_idx)

        # Rotate whole cell
        vec = bilayer.cell[0] + bilayer.cell[1]
        vec_costh = np.dot([1, 0, 0], vec) / np.dot(vec, vec) ** .5
        vec_th = np.arccos(vec_costh) * 360 / (2 * np.pi)
        bilayer = bilayer.rotate(vec_th, [0, 0, 1])

    # Sanity check
    assert len(bilayer) == natoms

    if ret_angle:
        return bilayer, theta
    else:
        return bilayer
