"""
Sile object for reading/writing ref files from ScaleUp
"""

from __future__ import division, print_function

# Import sile objects
from .sile import SileScaleUp
from ..sile import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell
from sisl._help import ensure_array

import numpy as np

__all__ = ['rhamSileScaleUp']


class rhamSileScaleUp(SileScaleUp):
    """ rham file object for ScaleUp

    This file contains the real-space Hamiltonian for a ScaleUp simulation
    """

    @Sile_fh_open
    def read_hamiltonian(self, geometry=None):
        """ Reads a Hamiltonian from the Sile """
        from sisl import Hamiltonian

        # First line is comment
        self.readline()

        no = geometry.no
        no_s = geometry.no_s

        # Easily construct the sparse matrix in python
        from scipy.sparse import lil_matrix

        # List of Hamiltonians per spin
        Hs = []

        old_spin = 0
        while True:

            line = self.readline().split()
            # EOF:
            if len(line) == 0:
                break

            if int(line[0]) != old_spin:
                if old_spin > 0:
                    Hs.append(H)

                # We need to create a new Hamiltonian
                H = lil_matrix((no, no_s), dtype=np.float64)
                old_spin = int(line[0])

            isc = ensure_array(map(int, line[1:4]))
            o1, o2 = map(int, line[4:6])
            rH, iH = map(float, line[6:8])

            i = geometry.sc_index(isc)
            H[o1-1, o2-1 + i * no] = rH
            # Currently we skip the imaginary part as it should be zero.

        # Append the just read Hamiltonian
        Hs.append(H)

        H = Hamiltonian.fromsp(geometry, Hs)
        H.finalize()
        return H


add_sile('rham', rhamSileScaleUp, case=False, gzip=True)
