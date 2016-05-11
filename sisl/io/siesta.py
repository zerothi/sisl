"""
Sile object for reading/writing SIESTA binary files
"""
from __future__ import print_function

# Import sile objects
from sisl.io.sile import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell, Grid
from sisl import Bohr, Ry
from sisl.tb import TightBinding, PhononTightBinding

import numpy as np

__all__ = ['SIESTASile']


class SIESTASile(NCSile):
    """ SIESTA file object """

    def read_sc(self):
        """ Returns a SuperCell object from a SIESTA.nc file
        """
        if not hasattr(self, 'fh'):
            with self:
                return self.read_sc()

        cell = np.array(self.variables['cell'][:], np.float64)
        # Yes, this is ugly, I really should implement my unit-conversion tool
        cell = cell / Bohr
        cell.shape = (3, 3)

        nsc = np.array(self.variables['nsc'][:], np.int32)

        return SuperCell(cell, nsc=nsc)

    def read_geom(self):
        """ Returns Geometry object from a SIESTA.nc file

        NOTE: Interaction range of the Atoms are currently not read.
        """
        if not hasattr(self, 'fh'):
            with self:
                return self.read_geom()

        # Read supercell
        sc = self.read_sc()

        xyz = np.array(self.variables['xa'][:], np.float64)
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
            atoms = [None] * len(xyz)
            for ia in range(len(xyz)):
                atoms[ia] = spc[b_idx[ia] - 1]
        else:
            atoms = Atom[1]

        xyz /= Bohr

        # Create and return geometry object
        geom = Geometry(xyz, atoms=atoms, sc=sc)
        return geom

    def read_tb(self, **kwargs):
        """ Returns a tight-binding model from the underlying NetCDF file """
        if not hasattr(self, 'fh'):
            with self:
                return self.read_tb(**kwargs)

        ispin = 0
        if 'ispin' in kwargs:
            ispin = kwargs['ispin']

        # First read the geometry
        geom = self.read_geom()

        # Populate the things
        sp = self._crt_grp(self, 'SPARSE')
        v = sp.variables['isc_off']
        # pre-allocate the super-cells
        geom.sc.set_nsc(np.amax(v[:, :], axis=0) * 2 + 1)
        geom.sc.sc_off[:, :] = v[:, :]

        # Now create the tight-binding stuff (we re-create the
        # array, hence just allocate the smallest amount possible)
        tb = TightBinding(geom, nc=1)

        # Use Ef to move H to Ef = 0
        Ef = float(self.variables['Ef'][0]) / Ry ** tb._E_order

        S = np.array(sp.variables['S'][:], np.float64)
        H = np.array(sp.variables['H'][ispin, :],
                     np.float64) / Ry ** tb._E_order

        # Correct for the Fermi-level, Ef == 0
        H -= Ef * S[:]
        ncol = np.array(sp.variables['n_col'][:], np.int32)
        # Update maximum number of connections (in case future stuff happens)
        ptr = np.append(np.array(0, np.int32), np.cumsum(ncol)).flatten()

        col = np.array(sp.variables['list_col'][:], np.int32) - 1

        # Copy information over
        tb.ncol = ncol
        tb.ptr = ptr
        tb.col = col
        tb._nnzs = len(col)
        # Create new container
        tb._TB = np.empty([len(tb), 2], np.float64)
        tb._TB[:, 0] = H[:]
        tb._TB[:, 1] = S[:]

        return tb

    def grids(self):
        """ Return a list of available grids in this file. """
        if not hasattr(self, 'fh'):
            with self:
                return self.grids()

        grids = []
        for g in self.groups['GRID'].variables:
            grids.expand(g)

        return grids

    def read_grid(self, name, idx=0):
        """ Reads a grid in the current SIESTA.nc file

        Enables the reading and processing of the grids created by SIESTA
        """
        if not hasattr(self, 'fh'):
            with self:
                return self.read_grid(name)

        # Swap as we swap back in the end
        geom = self.read_geom().swapaxes(0, 2)

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
                grid /= Ry
        except:
            # Simply, we have no units
            pass

        # Read the grid, we want the z-axis to be the fastest
        # looping direction, hence x,y,z == 0,1,2
        grid = grid.swapaxes(0, 2)
        grid.set_geom(geom)

        return grid

    def write_geom(self, geom):
        """
        Creates the NetCDF file and writes the geometry information
        """
        sile_raise_write(self)

        if not hasattr(self, 'fh'):
            with self:
                return self.write_geom(geom)

        # Create initial dimensions
        self._crt_dim(self, 'one', 1)
        self._crt_dim(self, 'n_s', np.prod(geom.nsc))
        self._crt_dim(self, 'xyz', 3)
        self._crt_dim(self, 'no_s', np.prod(geom.nsc) * geom.no)
        self._crt_dim(self, 'no_u', geom.no)
        self._crt_dim(self, 'spin', 1)
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
        self.method = 'python'

        # Save stuff
        self.variables['nsc'][:] = geom.nsc
        self.variables['xa'][:] = geom.xyz * Bohr
        self.variables['cell'][:] = geom.cell * Bohr

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

    def write_tb(self, tb, **kwargs):
        """ Writes tight-binding model to file

        Parameters
        ----------
        tb : `TightBinding` model
           the model to be saved in the NC file
        Ef : double=0
           the Fermi level of the electronic structure (in eV)
        """
        # Ensure finalizations
        tb.finalize()

        if not hasattr(self, 'fh'):
            with self:
                return self.write_tb(tb, **kwargs)

        # Ensure that the geometry is written
        self.write_geom(tb.geom)

        v = self._crt_var(self, 'Ef', 'f8', ('one',))
        v.info = 'Fermi level'
        v.unit = 'Ry'
        v[:] = 0.
        if 'Ef' in kwargs:
            v[:] = kwargs['Ef'] * Ry ** tb._E_order
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

        self._crt_dim(sp, 'nnzs', tb.col.shape[0])
        v = self._crt_var(sp, 'n_col', 'i4', ('no_u',))
        v.info = "Number of non-zero elements per row"
        v[:] = tb.ncol[:]
        v = self._crt_var(sp, 'list_col', 'i4', ('nnzs',),
                          chunksizes=(len(tb.col),), **self._cmp_args)
        v.info = "Supercell column indices in the sparse format"
        v[:] = tb.col[:] + 1  # correct for fortran indices
        v = self._crt_var(sp, 'isc_off', 'i4', ('n_s', 'xyz'))
        v.info = "Index of supercell coordinates"
        v[:] = tb.geom.sc.sc_off[:, :]

        # Save tight-binding parameters
        v = self._crt_var(sp, 'S', 'f8', ('nnzs',),
                          chunksizes=(len(tb.col),), **self._cmp_args)
        v.info = "Overlap matrix"
        v[:] = tb._TB[:, 1]
        v = self._crt_var(sp, 'H', 'f8', ('spin', 'nnzs'),
                          chunksizes=(1, len(tb.col)), **self._cmp_args)
        v.info = "Hamiltonian"
        v.unit = "Ry"
        v[:] = tb._TB[:, 0] * Ry ** tb._E_order

        # Create the settings
        st = self._crt_grp(self, 'SETTINGS')
        v = self._crt_var(st, 'ElectronicTemperature', 'f8', ('one',))
        v.info = "Electronic temperature used for smearing DOS"
        v.unit = "Ry"
        v[:] = 0.025 * Ry
        v = self._crt_var(st, 'BZ', 'i4', ('xyz', 'xyz'))
        v.info = "Grid used for the Brillouin zone integration"
        v[:] = np.identity(3) * 2
        v = self._crt_var(st, 'BZ_displ', 'i4', ('xyz',))
        v.info = "Monkhorst-Pack k-grid displacements"
        v.unit = "b**-1"
        v[:] = np.zeros([3], np.float64)


if __name__ == "__main__":
    # Create geometry
    alat = 3.57
    dist = alat * 3. ** .5 / 4
    C = Atom(Z=6, R=dist * 1.01, orbs=2)
    sc = SuperCell(np.array([[0, 1, 1],
                             [1, 0, 1],
                             [1, 1, 0]], np.float64) * alat / 2)
    geom = Geometry(np.array([[0, 0, 0], [1, 1, 1]], np.float64) * alat / 4,
                    atoms=C, sc=sc)
    # Write stuff
    geom.write(SIESTASile('diamond.nc', 'w'))
    geomr = SIESTASile('diamond.nc', 'r').read_geom()
    print(geomr)
    print(geomr.cell)
    print(geomr.xyz)

    # Create the tight-binding model
    geom.set_supercell([3, 3, 3])
    tb = TightBinding(geom, nc=1)
    for ia in tb.geom:
        idx = tb.close(ia, dR=(0.1, dist * 1.01))
        tb[tb.a2o(ia), tb.a2o(idx[0])] = (0., 1.)
        tb[tb.a2o(ia), tb.a2o(idx[1])] = (-.2, 0.)
    print('Before', tb.tocsr()[0])
    SIESTASile('diamond.nc', 'w').write_tb(tb)
    tbr = SIESTASile('diamond.nc', 'r').read_tb()
    print('After', tbr.tocsr()[0])
