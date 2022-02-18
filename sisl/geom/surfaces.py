# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np

from sisl._internal import set_module
from sisl import Atom, Geometry, SuperCell

__all__ = ['fcc_slab']


@set_module("sisl.geom")
def fcc_slab(alat, atoms, miller, size=None, vacuum=None, orthogonal=False, offset=0):
    """ Construction of a surface slab from a face-centered cubic crystal

    The slab layers are stacked along the z-axis

    Parameters
    ----------
    alat : float
        lattice constant of the fcc crystal
    atoms : Atom
        the atom that the crystal consists of
    miller : str or array
        Miller indices of the surface facet
    size : 3-array, optional
        slab size along the lattice vectors
    vacuum : float, optional
        distance added to the third lattice vector to separate
        the slab from its periodic images
    orthogonal : bool, optional
        if True returns an orthogonal lattice
    offset : int, optional
        index to shuffle the layer sequence, eg ABC to BCA
    """
    if isinstance(miller, str):
        miller = (int(miller[0]), int(miller[1]), int(miller[2]))

    if miller == (1, 0, 0):

        if size is None:
            size = (1, 1, 2)

        sc = SuperCell(np.array([0.5 ** 0.5, 0.5 ** 0.5, 0.5]) * alat)
        g = Geometry([0, 0, 0], atoms=atoms, sc=sc)
        g = g.tile(size[2], 2)

        # slide AB layers relative to each other
        B = (offset + 1) % 2
        g.xyz[B::2] += (sc.cell[0] + sc.cell[1]) / 2

    elif miller == (1, 1, 0):

        if size is None:
            size = (1, 1, 2)

        sc = SuperCell(np.array([1., 0.5, 0.125]) ** 0.5 * alat)
        g = Geometry([0, 0, 0], atoms=atoms, sc=sc)
        g = g.tile(size[2], 2)

        # slide AB layers relative to each other
        B = (offset + 1) % 2
        g.xyz[B::2] += (sc.cell[0] + sc.cell[1]) / 2

    elif miller == (1, 1, 1):

        if size is None:
            size = (1, 1, 3)

        if not orthogonal:
            sc = SuperCell(np.array([[0.5 ** 0.5, 0, 0],
                                     [0.125 ** 0.5, 0.375 ** 0.5, 0],
                                     [0, 0, 1 / 3 ** 0.5]]) * alat)
            g = Geometry([0, 0, 0], atoms=atoms, sc=sc)
            g = g.tile(size[2], 2)

            # slide ABC layers relative to each other
            B = (offset + 1) % 3
            C = (offset + 2) % 3
            g.xyz[B::3] += sc.cell[0] / 3 + sc.cell[1] / 3
            g.xyz[C::3] += -sc.cell[0] / 3 + 2 * sc.cell[1] / 3

        elif orthogonal:
            sc = SuperCell(np.array([0.5, 4 * 0.375, 1 / 3]) ** 0.5 * alat)
            g = Geometry(np.array([[0, 0, 0],
                                   [0.125, 0.375, 0]]) ** 0.5 * alat,
                         atoms=atoms, sc=sc)
            g = g.tile(size[2], 2)

            # slide ABC layers relative to each other
            B = (offset + 1) % 6
            C = (offset + 2) % 6
            vec = 1.5 * sc.cell[0] + sc.cell[1] / 2
            for i in range(2):
                g.xyz[B+i::6] += vec / 3 - i % 2 * sc.cell[0]
                g.xyz[C+i::6] += 2 * vec / 3 - sc.cell[0]

    # finish slab
    g = g.repeat(size[1], 1).repeat(size[0], 0)

    if vacuum is not None:
        g.cell[2, 2] += vacuum
        g.set_nsc([3, 3, 1])
    else:
        g.set_nsc([3, 3, 3])

    if np.all(g.maxR(True) > 0.):
        g.optimize_nsc()

    return g
