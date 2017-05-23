"""
Sile object for reading/writing SIESTA binary files
"""
from __future__ import print_function

# Import sile objects
from .sile import SileCDFSIESTA
from ..sile import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell, Grid
from sisl.units.siesta import unit_convert
from sisl.physics import Hamiltonian

import numpy as np

__all__ = ['ncSileSiesta']

Bohr2Ang = unit_convert('Bohr', 'Ang')
Ry2eV = unit_convert('Ry', 'eV')


class ncSileSiesta(SileCDFSIESTA):
    """ SIESTA file object """

    def read_supercell(self):
        """ Returns a SuperCell object from a SIESTA.nc file
        """
        cell = np.array(self._value('cell'), np.float64)
        # Yes, this is ugly, I really should implement my unit-conversion tool
        cell *= Bohr2Ang
        cell.shape = (3, 3)

        nsc = np.array(self._value('nsc'), np.int32)

        return SuperCell(cell, nsc=nsc)

    def read_geometry(self):
        """ Returns Geometry object from a SIESTA.nc file

        NOTE: Interaction range of the Atoms are currently not read.
        """

        # Read supercell
        sc = self.read_supercell()

        xyz = np.array(self._value('xa'), np.float64)
        xyz.shape = (-1, 3)

        if 'BASIS' in self.groups:
            bg = self.groups['BASIS']
            # We can actually read the exact basis-information
            b_idx = np.array(bg.variables['basis'][:], np.int32)

            # Get number of different species
            n_b = len(bg.groups)

            spc = [None] * n_b
            zb = np.zeros([n_b], np.int32)
            for basis in bg.groups:
                # Retrieve index
                ID = bg.groups[basis].ID
                atm = dict()
                atm['Z'] = int(bg.groups[basis].Atomic_number)
                # We could possibly read in dR, however, that is not so easy?
                atm['mass'] = float(bg.groups[basis].Mass)
                atm['tag'] = basis
                atm['orbs'] = int(bg.groups[basis].Number_of_orbitals)
                spc[ID - 1] = Atom[atm]
            atom = [None] * len(xyz)
            for ia in range(len(xyz)):
                atom[ia] = spc[b_idx[ia] - 1]
        else:
            atom = Atom[1]

        xyz *= Bohr2Ang

        # Create and return geometry object
        geom = Geometry(xyz, atom, sc=sc)
        return geom

    def read_hamiltonian(self, **kwargs):
        """ Returns a tight-binding model from the underlying NetCDF file """

        # Get the default spin channel
        ispin = kwargs.get('ispin', -1)
        spin = 1
        if ispin == -1:
            spin = len(self._dimension('spin'))

        # First read the geometry
        geom = self.read_geometry()

        # Populate the things
        sp = self._crt_grp(self, 'SPARSE')
        v = sp.variables['isc_off']
        # pre-allocate the super-cells
        geom.sc.set_nsc(np.amax(v[:, :], axis=0) * 2 + 1)
        geom.sc.sc_off[:, :] = v[:, :]

        # Now create the tight-binding stuff (we re-create the
        # array, hence just allocate the smallest amount possible)
        ham = Hamiltonian(geom, nnzpr=1, orthogonal=False, spin=spin)

        # Use Ef to move H to Ef = 0
        Ef = float(self._value('Ef')[0]) * Ry2eV ** ham._E_order
        S = np.array(sp.variables['S'][:], np.float64)

        ncol = np.array(sp.variables['n_col'][:], np.int32)
        # Update maximum number of connections (in case future stuff happens)
        ptr = np.append(np.array(0, np.int32), np.cumsum(ncol)).flatten()
        col = np.array(sp.variables['list_col'][:], np.int32) - 1

        # Copy information over
        ham._data.ncol = ncol
        ham._data.ptr = ptr
        ham._data.col = col
        ham._nnz = len(col)

        # Create new container
        H = np.array(sp.variables['H'][ispin, :],
                     np.float64) * Ry2eV ** ham._E_order
        # Correct for the Fermi-level, Ef == 0
        H -= Ef * S[:]

        ham._data._D = np.empty([ham._data.ptr[-1], spin+1], np.float64)
        if ispin == -1:
            for i in range(spin):
                # Create new container
                H = np.array(sp.variables['H'][i, :],
                             np.float64) * Ry2eV ** ham._E_order
                # Correct for the Fermi-level, Ef == 0
                H -= Ef * S[:]
                ham._data._D[:, i] = H[:]
        else:
            # Create new container
            H = np.array(sp.variables['H'][ispin, :],
                         np.float64) * Ry2eV ** ham._E_order
            # Correct for the Fermi-level, Ef == 0
            H -= Ef * S[:]
            ham._data._D[:, 0] = H[:]
        ham._data._D[:, ham.S_idx] = S[:]

        return ham

    def grids(self):
        """ Return a list of available grids in this file. """

        grids = []
        for g in self.groups['GRID'].variables:
            grids.expand(g)

        return grids

    def read_grid(self, name, idx=0):
        """ Reads a grid in the current SIESTA.nc file

        Enables the reading and processing of the grids created by SIESTA
        """
        # Swap as we swap back in the end
        geom = self.read_geometry().swapaxes(0, 2)

        # Shorthand
        g = self.groups['GRID']

        # Create the grid
        nx = len(g.dimensions['nx'])
        ny = len(g.dimensions['ny'])
        nz = len(g.dimensions['nz'])

        # Shorthand variable name
        v = g.variables[name]

        # Create the grid, SIESTA uses periodic, always
        grid = Grid([nz, ny, nx], bc=Grid.Periodic, dtype=v.dtype)

        if len(v[:].shape) == 3:
            grid.grid = v[:, :, :]
        else:
            grid.grid = v[idx, :, :, :]

        try:
            u = v.unit
            if u == 'Ry':
                # Convert to ev
                grid *= Ry2eV
        except:
            # Simply, we have no units
            pass

        # Read the grid, we want the z-axis to be the fastest
        # looping direction, hence x,y,z == 0,1,2
        grid = grid.swapaxes(0, 2)
        grid.set_geom(geom)

        return grid

    def write_geometry(self, geom):
        """
        Creates the NetCDF file and writes the geometry information
        """
        sile_raise_write(self)

        # Create initial dimensions
        self._crt_dim(self, 'one', 1)
        self._crt_dim(self, 'n_s', np.prod(geom.nsc))
        self._crt_dim(self, 'xyz', 3)
        self._crt_dim(self, 'no_s', np.prod(geom.nsc) * geom.no)
        self._crt_dim(self, 'no_u', geom.no)
        self._crt_dim(self, 'na_u', geom.na)

        # Create initial geometry
        v = self._crt_var(self, 'nsc', 'i4', ('xyz',))
        v.info = 'Number of supercells in each unit-cell direction'
        v = self._crt_var(self, 'lasto', 'i4', ('na_u',))
        v.info = 'Last orbital of equivalent atom'
        v = self._crt_var(self, 'xa', 'f8', ('na_u', 'xyz'))
        v.info = 'Atomic coordinates'
        v.unit = 'Bohr'
        v = self._crt_var(self, 'cell', 'f8', ('xyz', 'xyz'))
        v.info = 'Unit cell'
        v.unit = 'Bohr'

        # Create designation of the creation
        self.method = 'sisl'

        # Save stuff
        self.variables['nsc'][:] = geom.nsc
        self.variables['xa'][:] = geom.xyz / Bohr2Ang
        self.variables['cell'][:] = geom.cell / Bohr2Ang

        # Create basis group
        bs = self._crt_grp(self, 'BASIS')

        # Create variable of basis-indices
        b = self._crt_var(bs, 'basis', 'i4', ('na_u',))
        b.info = "Basis of each atom by ID"

        orbs = np.empty([geom.na], np.int32)

        for ia, a, isp in geom.iter_species():
            b[ia] = isp + 1
            orbs[ia] = a.orbs
            if a.tag in bs.groups:
                # Assert the file sizes
                if bs.groups[a.tag].Number_of_orbitals != a.orbs:
                    raise ValueError(
                        'File ' +
                        self.file +
                        ' has erroneous data in regards of ' +
                        'of the already stored dimensions.')
            else:
                ba = bs.createGroup(a.tag)
                ba.ID = np.int32(isp + 1)
                ba.Atomic_number = np.int32(a.Z)
                ba.Mass = a.mass
                ba.Label = a.tag
                ba.Element = a.symbol
                ba.Number_of_orbitals = np.int32(a.orbs)

        # Store the lasto variable as the remaining thing to do
        self.variables['lasto'][:] = np.cumsum(orbs)

    def write_hamiltonian(self, ham, **kwargs):
        """ Writes Hamiltonian model to file

        Parameters
        ----------
        ham : `Hamiltonian` model
           the model to be saved in the NC file
        Ef : double=0
           the Fermi level of the electronic structure (in eV)
        """
        # Ensure finalizations
        ham.finalize()

        # Ensure that the geometry is written
        self.write_geometry(ham.geom)

        self._crt_dim(self, 'spin', ham._spin)

        v = self._crt_var(self, 'Ef', 'f8', ('one',))
        v.info = 'Fermi level'
        v.unit = 'Ry'
        v[:] = 0.
        if 'Ef' in kwargs:
            v[:] = kwargs['Ef'] / Ry2eV ** ham._E_order
        v = self._crt_var(self, 'Qtot', 'f8', ('one',))
        v.info = 'Total charge'
        v[:] = 0.
        if 'Qtot' in kwargs:
            v[:] = kwargs['Qtot']
        if 'Q' in kwargs:
            v[:] = kwargs['Q']

        # Append the sparsity pattern
        # Create basis group
        sp = self._crt_grp(self, 'SPARSE')

        self._crt_dim(sp, 'nnzs', ham._data.col.shape[0])
        v = self._crt_var(sp, 'n_col', 'i4', ('no_u',))
        v.info = "Number of non-zero elements per row"
        v[:] = ham._data.ncol[:]
        v = self._crt_var(sp, 'list_col', 'i4', ('nnzs',),
                          chunksizes=(len(ham._data.col),), **self._cmp_args)
        v.info = "Supercell column indices in the sparse format"
        v[:] = ham._data.col[:] + 1  # correct for fortran indices
        v = self._crt_var(sp, 'isc_off', 'i4', ('n_s', 'xyz'))
        v.info = "Index of supercell coordinates"
        v[:] = ham.geom.sc.sc_off[:, :]

        # Save tight-binding parameters
        v = self._crt_var(sp, 'S', 'f8', ('nnzs',),
                          chunksizes=(len(ham._data.col),), **self._cmp_args)
        v.info = "Overlap matrix"
        if ham.orthogonal:
            # We need to create the orthogonal pattern
            tmp = ham._data.copy(dims=[0])
            tmp.empty(keep=True)
            for i in range(tmp.shape[0]):
                tmp[i, i] = 1.
            v[:] = tmp._D[:, 0]
            del tmp
        else:
            v[:] = ham._data._D[:, ham.S_idx]
        v = self._crt_var(sp, 'H', 'f8', ('spin', 'nnzs'),
                          chunksizes=(1, len(ham._data.col)), **self._cmp_args)
        v.info = "Hamiltonian"
        v.unit = "Ry"
        for i in range(ham._spin):
            v[i, :] = ham._data._D[:, i] / Ry2eV ** ham._E_order

        # Create the settings
        st = self._crt_grp(self, 'SETTINGS')
        v = self._crt_var(st, 'ElectronicTemperature', 'f8', ('one',))
        v.info = "Electronic temperature used for smearing DOS"
        v.unit = "Ry"
        v[:] = 0.025 / Ry2eV
        v = self._crt_var(st, 'BZ', 'i4', ('xyz', 'xyz'))
        v.info = "Grid used for the Brillouin zone integration"
        v[:] = np.identity(3) * 2
        v = self._crt_var(st, 'BZ_displ', 'i4', ('xyz',))
        v.info = "Monkhorst-Pack k-grid displacements"
        v.unit = "b**-1"
        v[:] = np.zeros([3], np.float64)

    def ArgumentParser(self, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geometry().ArgumentParser(*args, **newkw)


add_sile('nc', ncSileSiesta)
