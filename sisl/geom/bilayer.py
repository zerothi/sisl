import numpy as np

from sisl import Atom, geom, Cuboid

__all__ = ['bilayer']


def bilayer(bond, atom=None, twist=(5, 6), separation=3.35, return_angle=False, layer='both'):
    """ Commensurate unit cell of a twisted (hexagonal) bilayer structure.

    This routine follows the prescription of twisted bilayer graphene found in
    Laissardiere et al., Nano Lett. 10, 804-808 (2010).


    Parameters
    ----------
    bond : float
       length between atoms in nano-tube
    atom : Atom(6)
       bilayer atoms
    twist : (int, int)
       coordinates (n, m) defining the twist angle (theta)
    separation : float
       distance between the two layers
    return_angle : bool
       computed twist angle is returned in addition to the geometry instance
    layer : {'both', 'bottom', 'top'}
       control which layer(s) to include in geometry
    """
    if atom is None:
        atom = Atom(Z=6, R=bond * 1.01)

    # Compute twist angle
    m, n = twist
    if m < n:
        m, n = n, m

    if not (isinstance(n, int) and isinstance(m, int)):
        raise RuntimeError('twist coordinates need to be integers!')

    cos_theta = (n ** 2 + 4 * n * m + m ** 2) / (2 * (n ** 2 + n * m + m ** 2) )
    theta = np.arccos(cos_theta) * 360 / (2 * np.pi)

    # Build structure
    graphene = geom.graphene(bond=bond, atom=atom)

    # Choose a repetition that should assure both layers extend over the unit cell
    rep = 4 * (n + m)

    # Construct bottom layer
    bottom = graphene.tile(rep, axis=0).tile(rep, axis=1)
    bottom = bottom.move(-bottom.center())
    bottom.cell[0] = n * graphene.cell[0] + m * graphene.cell[1]
    bottom.cell[1] = -m * graphene.cell[0] + (n + m) * graphene.cell[1]

    # Construct top layer
    top = bottom.move([0, 0, separation]).rotate(theta, [0, 0, 1])
    top.cell[:] = bottom.cell[:]

    # Compute number of atoms per layer
    natoms = 2 * (n ** 2 + n * m + m ** 2)

    # Which layers to include
    if layer.lower() == 'bottom':
        bilayer = bottom
    elif layer.lower() == 'top':
        bilayer = top
    else:
        bilayer = bottom.add(top)
        natoms *= 2

    # Remove atoms outside cell
    cell_box = Cuboid(bilayer.cell[:], center=bilayer.sc.center())
    outside_idx = []
    for i in bilayer:
        if not cell_box.within(bilayer[i]):
            outside_idx.append(i)
    bilayer = bilayer.remove(outside_idx)

    # Center geometry around first atom
    bilayer = bilayer.move(-bilayer.xyz[0])

    # Rotate whole cell
    vec = bilayer.cell[0] + bilayer.cell[1]
    vec_costh = np.dot([1, 0, 0], vec) / np.dot(vec, vec) ** .5
    vec_th = np.arccos(vec_costh) * 360 / (2 * np.pi)
    bilayer = bilayer.rotate(vec_th, [0, 0, 1])

    # Sanity check
    assert len(bilayer) == natoms

    if return_angle:
        return bilayer, theta
    else:
        return bilayer
