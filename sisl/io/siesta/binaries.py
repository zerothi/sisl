"""
Sile object for reading/writing SIESTA binary files
"""
from __future__ import print_function

import numpy as np

try:
    from . import _siesta
    found_module = True
except Exception as e:
    found_module = False

# Import sile objects
from .sile import SileBinSiesta
from ..sile import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell, Grid
from sisl.units.siesta import unit_convert
from sisl.physics import Hamiltonian


__all__ = ['TSHSSileSiesta']
__all__ += ['rhoSileSiesta', 'vSileSiesta']


class TSHSSileSiesta(SileBinSiesta):
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
        for _, orb in enumerate(orbs):
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
        H = Hamiltonian(geom, spin, nnzpr=1, orthogonal=False)

        # Create the new sparse matrix
        H._csr.ncol = ncol.astype(np.int32, copy=False)
        H._csr.ptr = np.insert(np.cumsum(ncol, dtype=np.int32), 0, 0)
        # Correct fortran indices
        H._csr.col = col.astype(np.int32, copy=False) - 1
        H._csr._nnz = len(col)

        H._csr._D = np.empty([nnz, spin+1], np.float64)
        H._csr._D[:, :spin] = dH[:, :]
        H._csr._D[:, spin] = dS[:]

        return H


class GridSileSiesta(SileBinSiesta):
    """ Grid file object from a binary Siesta output file """

    grid_unit = 1.

    def read_supercell(self, *args, **kwargs):

        cell = _siesta.read_grid_cell(self.file)
        cell = np.array(cell.T, np.float64)
        cell.shape = (3, 3)

        SC = SuperCell(cell)
        return SC

    def read_grid(self, *args, **kwargs):
        """ Read grid contained in the Grid file """
        # Read the sizes
        nspin, mesh = _siesta.read_grid_sizes(self.file)
        # Read the cell and grid
        cell, grid = _siesta.read_grid(self.file, nspin, mesh[0], mesh[1], mesh[2])

        cell = np.array(cell.T, np.float64)
        cell.shape = (3, 3)

        g = Grid(mesh, sc=SuperCell(cell), dtype=np.float32)
        g.grid = np.array(grid.swapaxes(0, 2), np.float32) * self.grid_unit
        return g


class rhoSileSiesta(GridSileSiesta):
    """ .*RHO* file object from a binary Siesta output file """
    grid_unit = 1.


class vSileSiesta(GridSileSiesta):
    """ .V* file object from a binary Siesta output file """
    grid_unit = unit_convert('Ry', 'eV')


if found_module:
    add_sile('TSHS', TSHSSileSiesta)
    add_sile('RHO', rhoSileSiesta)
    add_sile('RHOINIT', rhoSileSiesta)
    add_sile('DRHO', rhoSileSiesta)
    add_sile('IOCH', rhoSileSiesta)
    add_sile('TOCH', rhoSileSiesta)
    add_sile('VH', vSileSiesta)
    add_sile('VNA', vSileSiesta)
    add_sile('VT', vSileSiesta)
