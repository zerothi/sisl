"""
Sile object for reading/writing SIESTA binary files
"""
from __future__ import print_function

import numpy as np

try:
    import _siesta
    found_module = True
except:
    found_module = False

# Import sile objects
from .sile import SileBinSIESTA
from ..sile import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell, Grid
from sisl.units.siesta import unit_convert
from sisl.physics import Hamiltonian


__all__ = ['TSHSSileSiesta']


class TSHSSileSiesta(SileBinSIESTA):
    """ TranSIESTA file object """

    def read_supercell(self):
        """ Returns a SuperCell object from a siesta.TSHS file
        """

        n_s = _siesta.read_tshs_sizes(self.file)[3]
        arr = _siesta.read_tshs_cell(self.file, n_s)
        nsc = np.array(arr[0].T, np.int32)
        cell = np.array(arr[1].T, np.float64)
        cell.shape = (3, 3)
        isc = np.array(arr[2].T, np.int32)
        isc.shape = (-1, 3)

        SC = SuperCell(cell, nsc=nsc)
        SC.sc_off = isc
        return SC

    def read_geometry(self):
        """ Returns Geometry object from a siesta.TSHS file """

        # Read supercell
        sc = self.read_supercell()

        na = _siesta.read_tshs_sizes(self.file)[1]
        arr = _siesta.read_tshs_geom(self.file, na)
        xyz = np.array(arr[0].T, np.float64)
        xyz.shape = (-1, 3)
        lasto = np.array(arr[1], np.int32)

        # Create all different atoms...
        # The TSHS file does not contain the
        # atomic numbers, so we will just
        # create them individually
        orbs = np.diff(lasto)

        # Get unique orbitals
        uorb = np.unique(orbs)
        # Create atoms
        atoms = []
        for Z, orb in enumerate(uorb):
            atoms.append(Atom(Z+1, orbs=orb))

        def get_atom(atoms, orbs):
            for atom in atoms:
                if atom.orbs == orbs:
                    return atom

        atom = []
        for i, orb in enumerate(orbs):
            atom.append(get_atom(atoms, orb))

        # Create and return geometry object
        geom = Geometry(xyz, atom, sc=sc)

        return geom

    def read_hamiltonian(self, **kwargs):
        """ Returns the electronic structure from the siesta.TSHS file """

        # First read the geometry
        geom = self.read_geometry()

        # Now read the sizes used...
        sizes = _siesta.read_tshs_sizes(self.file)
        spin = sizes[0]
        no = sizes[2]
        nnz = sizes[4]
        ncol, col, dH, dS = _siesta.read_tshs_es(self.file, spin, no, nnz)

        # Create the Hamiltonian container
        H = Hamiltonian(geom, nnzpr=1, orthogonal=False, spin=spin)

        # Create the new sparse matrix
        H._data.ncol = np.array(ncol, np.int32)
        ptr = np.cumsum(ncol)
        ptr = np.insert(ptr, 0, 0)
        H._data.ptr = np.array(ptr, np.int32)
        # Correct fortran indices
        H._data.col = np.array(col, np.int32) - 1
        H._data._nnz = len(col)

        H._data._D = np.empty([nnz, spin+1], np.float64)
        for i in range(spin):
            # this is because of the F-ordering
            H._data._D[:, i] = dH[:, i]
        H._data._D[:, spin] = dS[:]

        return H


if found_module:
    add_sile('TSHS', TSHSSileSiesta)
