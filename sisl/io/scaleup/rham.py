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

        # Create a copy as we may change the
        # internal atoms (due to wrong orbital values)
        g = geometry.copy()

        # First line is comment
        self.readline()

        # First read the entire file
        lines = self.readlines()

        def pl(line):
            l = line.split()
            s = int(l[0])
            isc = list(map(int, l[1:4]))
            o1, o2 = map(int, l[4:6])
            rH, iH = map(float, l[6:8])
            return s, isc, o1-1, o2-1, rH, iH

        ns = 0
        no = 0
        m_sc = np.zeros([3], np.int32)

        # To save creating a new list
        for i, line in enumerate(lines):
            lines[i] = pl(line)
            ns = max(lines[i][0], ns)
            m_sc[0] = max(abs(lines[i][1][0]), m_sc[0])
            m_sc[1] = max(abs(lines[i][1][1]), m_sc[1])
            m_sc[2] = max(abs(lines[i][1][2]), m_sc[2])
            no = max(lines[i][2], no)

        # Correct for C-index
        if no + 1 != g.no:
            # First try and read the orbocc file
            try:
                species = get_sile(self.file.rsplit('rham', 1)[0] + 'orbocc').read_atom()
                for i, atom in enumerate(species.atom):
                    g.atom._atom[i] = atom
            except:
                pass
        # Check again, to be sure...
        if no + 1 != g.no:
            raise ValueError(('The Geometry has a different number of '
                              'orbitals, please correct by adding the orbocc file.'))

        # Now, we know the size etc. of the Hamiltonian
        m_sc = m_sc * 2 + 1
        if np.any(g.nsc != m_sc):
            # Correct the number of supercells
            g.set_nsc(m_sc)

        # Easily construct the sparse matrix in python
        from scipy.sparse import lil_matrix

        # List of Hamiltonians per spin
        Hs = [None] * ns
        # Get system size
        no = g.no
        no_s = g.no_s

        old_s = 0
        for s, isc, o1, o2, rH, iH in lines:

            if s != old_s:
                # We need to create a new Hamiltonian
                H = lil_matrix((no, no_s), dtype=np.float64)
                old_s = s
                Hs[s-1] = H

            i = g.sc_index(isc)
            H[o1, o2 + i * no] = rH
            # Currently we skip the imaginary part as it should be zero.

        H = Hamiltonian.fromsp(g, Hs)
        H.finalize()
        return H


add_sile('rham', rhamSileScaleUp, case=False, gzip=True)
