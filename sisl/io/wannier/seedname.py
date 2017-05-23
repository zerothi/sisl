"""
Sile object for reading/writing Wannier90 in/output
"""
from __future__ import print_function

import numpy as np

# Import sile objects
from .sile import SileW90
from ..sile import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell
from sisl.physics import Hamiltonian

from sisl.units import unit_convert

__all__ = ['winSileW90']


class winSileW90(SileW90):
    """ Wannier seedname input file object

    This `Sile` enables easy interaction with the Wannier90 code.

    A seedname is the basis of reading all Wannier90 output because
    every file in Wannier90 is based of the name of the seed.

    Hence, if the correct flags are present in the seedname.win file,
    and the corresponding files are created, then the corresponding
    quantity may be read.

    For instance to read the Wannier-centres you *must* have this in your
    seedname.win:

        write_xyz = true

    while if you want to read the Wannier Hamiltonian you should have this:

        write_xyz = true
        plot_hr = true


    Examples
    --------

    >>> H = win90.read_hamiltonian()

    >>> H = win90.read_hamiltonian(dtype=numpy.float64) # only read real-part

    >>> H = win90.read_hamiltonian(cutoff=0.00001) # explicitly set the cutoff for the elements

    """

    def _setup(self):
        """ Setup `winSileW90` after initialization """
        self._seed = self.file.replace('.win', '')

    def _set_file(self, suffix):
        """ Update readed file """
        self._file = self._seed + suffix

    @Sile_fh_open
    def _read_supercell(self):
        """ Defered routine """

        f, l = self.step_to('unit_cell_cart', case=False)
        if not f:
            raise ValueError("The unit-cell vectors could not be found in the seed-file.")

        l = self.readline()
        lines = []
        while not l.startswith('end'):
            lines.append(l)
            l = self.readline()

        # Check whether the first element is a specification of the units
        pos_unit = lines[0].split()
        if len(pos_unit) > 2:
            unit = 1.
        else:
            unit = unit_convert(pos_unit[0], 'Ang')
            # Remove the line with the unit...
            lines.pop(0)

        # Create the cell
        cell = np.empty([3, 3], np.float64)
        for i in [0, 1, 2]:
            cell[i, :] = [float(x) for x in lines[i].split()]

        return SuperCell(cell * unit)

    def read_supercell(self):
        """ Reads a `SuperCell` and creates the Wannier90 cell """
        self._set_file('.win')

        return self._read_supercell()

    @Sile_fh_open
    def _read_geometry(self):
        """ Defered routine """

        nc = int(self.readline())

        # Comment
        self.readline()

        na = 0
        sp = [None] * nc
        xyz = np.empty([nc, 3], np.float64)
        for ia in range(nc):
            l = self.readline().split()
            sp[ia] = l.pop(0)
            if sp[ia] == 'X':
                na = ia + 1
            xyz[ia, :] = [float(k) for k in l[:3]]

        return Geometry(xyz[:na, :], atom='H')

    def read_geometry(self, *args, **kwargs):
        """ Reads a `Geometry` and creates the Wannier90 cell """

        # Read in the super-cell
        sc = self.read_supercell()

        self._set_file('_centres.xyz')

        geom = self._read_geometry()
        geom.set_sc(sc)

        return geom

    @Sile_fh_open
    def _read_hamiltonian(self, geom, dtype=np.complex128, **kwargs):
        """ Reads a Hamiltonian

        Reads the Hamiltonian model
        """

        cutoff = kwargs.get('cutoff', 0.00001)

        # Rewind to ensure we can read the entire matrix structure
        self.fh.seek(0)

        # Time of creation
        self.readline()

        # Retrieve # of wannier functions (or size of Hamiltonian)
        no = int(self.readline())
        # Number of Wigner-Seitz degeneracy points
        nrpts = int(self.readline())

        # First read across the Wigner-Seitz degeneracy
        # This is formatted with 15 per-line.
        if nrpts % 15 == 0:
            nlines = nrpts
        else:
            nlines = nrpts + 15 - nrpts % 15

        ws = []
        for i in range(nlines // 15):
            ws.extend(map(int, self.readline().split()))

        # Convert to numpy array and invert (for weights)
        ws = 1. / np.array(ws, np.float64).flatten()

        # Figure out the number of supercells
        # and maintain the Hamiltonian in the ham list
        nsc = [0, 0, 0]

        # List for holding the Hamiltonian
        ham = []
        iws = -1

        while True:
            l = self.readline()
            if l == '':
                break

            # Split here...
            l = l.split()

            # Get super-cell, row and column
            iA, iB, iC, r, c = map(int, l[:5])

            nsc[0] = max(nsc[0], abs(iA))
            nsc[1] = max(nsc[1], abs(iB))
            nsc[2] = max(nsc[2], abs(iC))

            # Update index for degeneracy, if required
            if r + c == 2:
                iws += 1

            # Get degeneracy of this element
            f = ws[iws]

            # Store in the Hamiltonian array:
            #   isc
            #   row
            #   column
            #   Hr
            #   Hi
            ham.append(([iA, iB, iC], r-1, c-1, float(l[5]) * f, float(l[6]) * f))

        # Update number of super-cells
        geom.set_nsc([i * 2 + 1 for i in nsc])

        # With the geometry in place we can read in the entire matrix
        # Create a new sparse matrix
        from scipy.sparse import lil_matrix
        Hr = lil_matrix((geom.no, geom.no_s), dtype=dtype)
        Hi = lil_matrix((geom.no, geom.no_s), dtype=dtype)

        # populate the Hamiltonian by examining the cutoff value
        for isc, r, c, hr, hi in ham:

            # Calculate the column corresponding to the
            # correct super-cell
            c = c + geom.sc_index(isc) * geom.no

            if abs(hr) > cutoff:
                Hr[r, c] = hr
            if abs(hi) > cutoff:
                Hi[r, c] = hi
        del ham

        if np.dtype(dtype).kind == 'c':
            Hr = Hr.tocsr()
            Hi = Hi.tocsr()
            Hr = Hr + 1j*Hi

        return Hamiltonian.sp2HS(geom, Hr)

    def read_hamiltonian(self, *args, **kwargs):
        """ Read the electronic structure of the Wannier90 output 

        Parameters
        ----------
        cutoff: (float, 0.00001)
           the cutoff value for the zero Hamiltonian elements
        """

        # Retrieve the geometry...
        geom = self.read_geometry()

        # Set file
        self._set_file('_hr.dat')

        return self._read_hamiltonian(geom, *args, **kwargs)

    def ArgumentParser(self, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geometry().ArgumentParser(*args, **newkw)


add_sile('win', winSileW90, gzip=True)
