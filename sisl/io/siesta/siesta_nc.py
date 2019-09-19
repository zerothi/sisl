from __future__ import print_function

from numbers import Integral
import numpy as np

from .sile import SileCDFSiesta
from ..sile import *

from sisl._array import aranged
from sisl.unit.siesta import unit_convert
from sisl import Geometry, Atom, Atoms, SuperCell, Grid, SphericalOrbital
from sisl.physics import SparseOrbitalBZ
from sisl.physics import DensityMatrix, EnergyDensityMatrix
from sisl.physics import DynamicalMatrix
from sisl.physics import Hamiltonian
try:
    from . import _siesta
except:
    pass
from ._help import *


__all__ = ['ncSileSiesta']

Bohr2Ang = unit_convert('Bohr', 'Ang')
Ry2eV = unit_convert('Ry', 'eV')


class ncSileSiesta(SileCDFSiesta):
    """ Generic NetCDF output file containing a large variety of information """

    def read_supercell_nsc(self):
        """ Returns number of supercell connections """
        return np.array(self._value('nsc'), np.int32)

    def read_supercell(self):
        """ Returns a SuperCell object from a Siesta.nc file """
        cell = np.array(self._value('cell'), np.float64)
        # Yes, this is ugly, I really should implement my unit-conversion tool
        cell *= Bohr2Ang
        cell.shape = (3, 3)

        nsc = self.read_supercell_nsc()

        return SuperCell(cell, nsc=nsc)

    def read_basis(self):
        """ Returns a set of atoms corresponding to the basis-sets in the nc file """
        if 'BASIS' not in self.groups:
            return None

        basis = self.groups['BASIS']
        atom = [None] * len(basis.groups)

        for a_str in basis.groups:
            a = basis.groups[a_str]

            if 'orbnl_l' not in a.variables:

                # Do the easy thing.

                # Get number of orbitals
                label = a.Label.strip()
                Z = int(a.Atomic_number)
                mass = float(a.Mass)

                i = int(a.ID) - 1
                atom[i] = Atom(Z, [-1] * a.Number_of_orbitals, mass=mass, tag=label)
                continue

            # Retrieve values
            orb_l = a.variables['orbnl_l'][:] # angular quantum number
            orb_n = a.variables['orbnl_n'][:] # principal quantum number
            orb_z = a.variables['orbnl_z'][:] # zeta
            orb_P = a.variables['orbnl_ispol'][:] > 0 # polarization shell, or not
            orb_q0 = a.variables['orbnl_pop'][:] # q0 for the orbitals
            orb_delta = a.variables['delta'][:] # delta for the functions
            orb_psi = a.variables['orb'][:, :]

            # Now loop over all orbitals
            orbital = []

            # Number of basis-orbitals (before m-expansion)
            no = len(a.dimensions['norbs'])

            # All orbital data
            for io in range(no):

                n = orb_n[io]
                l = orb_l[io]
                z = orb_z[io]
                P = orb_P[io]

                # Grid spacing in Bohr (conversion is done later
                # because the normalization is easier)
                delta = orb_delta[io]

                # Since the readed data has fewer significant digits we
                # might as well re-create the table of the radial component.
                r = aranged(orb_psi.shape[1]) * delta

                # To get it per Ang**3
                # TODO, check that this is correct.
                # The fact that we have to have it normalized means that we need
                # to convert psi /sqrt(Bohr**3) -> /sqrt(Ang**3)
                # \int psi^\dagger psi == 1
                psi = orb_psi[io, :] * r ** l / Bohr2Ang ** (3./2.)

                # Create the sphericalorbital and then the atomicorbital
                sorb = SphericalOrbital(l, (r * Bohr2Ang, psi), orb_q0[io])

                # This will be -l:l (this is the way siesta does it)
                orbital.extend(sorb.toAtomicOrbital(n=n, Z=z, P=P))

            # Get number of orbitals
            label = a.Label.strip()
            Z = int(a.Atomic_number)
            mass = float(a.Mass)

            i = int(a.ID) - 1
            atom[i] = Atom(Z, orbital, mass=mass, tag=label)
        return atom

    def read_geometry(self):
        """ Returns Geometry object from a Siesta.nc file """

        # Read supercell
        sc = self.read_supercell()

        xyz = np.array(self._value('xa'), np.float64)
        xyz.shape = (-1, 3)

        if 'BASIS' in self.groups:
            basis = self.read_basis()
            species = self.groups['BASIS'].variables['basis'][:] - 1
            atom = Atoms([basis[i] for i in species])
        else:
            atom = Atom(1)

        xyz *= Bohr2Ang

        # Create and return geometry object
        geom = Geometry(xyz, atom, sc=sc)
        return geom

    def read_force(self):
        """ Returns a vector with final forces contained. """
        return _a.arrayd(self._value('fa')) * Ry2eV / Bohr2Ang

    def read_fermi_level(self):
        """ Returns the fermi-level """
        return self._value('Ef')[:] * Ry2eV

    def _read_class(self, cls, dim=1, **kwargs):
        # Get the default spin channel
        # First read the geometry
        geom = self.read_geometry()

        # Populate the things
        sp = self._crt_grp(self, 'SPARSE')

        # Now create the tight-binding stuff (we re-create the
        # array, hence just allocate the smallest amount possible)
        C = cls(geom, dim, nnzpr=1)

        C._csr.ncol = np.array(sp.variables['n_col'][:], np.int32)
        # Update maximum number of connections (in case future stuff happens)
        C._csr.ptr = np.insert(np.cumsum(C._csr.ncol, dtype=np.int32), 0, 0)
        C._csr.col = np.array(sp.variables['list_col'][:], np.int32) - 1

        # Copy information over
        C._csr._nnz = len(C._csr.col)
        C._csr._D = np.empty([C._csr.ptr[-1], dim], np.float64)

        # Convert from isc to sisl isc
        _csr_from_sc_off(C.geometry, sp.variables['isc_off'][:, :], C._csr)

        return C

    def _read_class_spin(self, cls, **kwargs):
        # Get the default spin channel
        spin = len(self._dimension('spin'))

        # First read the geometry
        geom = self.read_geometry()

        # Populate the things
        sp = self._crt_grp(self, 'SPARSE')

        # Since we may read in an orthogonal basis (stored in a Siesta compliant file)
        # we can check whether it is orthogonal by checking the sum of the absolute S
        # I.e. whether only diagonal elements are present.
        S = np.array(sp.variables['S'][:], np.float64)
        orthogonal = np.abs(S).sum() == geom.no

        # Now create the tight-binding stuff (we re-create the
        # array, hence just allocate the smallest amount possible)
        C = cls(geom, spin, nnzpr=1, orthogonal=orthogonal)

        C._csr.ncol = np.array(sp.variables['n_col'][:], np.int32)
        # Update maximum number of connections (in case future stuff happens)
        C._csr.ptr = np.insert(np.cumsum(C._csr.ncol, dtype=np.int32), 0, 0)
        C._csr.col = np.array(sp.variables['list_col'][:], np.int32) - 1

        # Copy information over
        C._csr._nnz = len(C._csr.col)
        if orthogonal:
            C._csr._D = np.empty([C._csr.ptr[-1], spin], np.float64)
        else:
            C._csr._D = np.empty([C._csr.ptr[-1], spin + 1], np.float64)
            C._csr._D[:, C.S_idx] = S

        # Convert from isc to sisl isc
        _csr_from_sc_off(C.geometry, sp.variables['isc_off'][:, :], C._csr)

        return C

    def read_overlap(self, **kwargs):
        """ Returns a overlap matrix from the underlying NetCDF file """
        S = self._read_class(SparseOrbitalBZ, **kwargs)

        sp = self._crt_grp(self, 'SPARSE')
        S._csr._D[:, 0] = sp.variables['S'][:]

        return S

    def read_hamiltonian(self, **kwargs):
        """ Returns a Hamiltonian from the underlying NetCDF file """
        H = self._read_class_spin(Hamiltonian, **kwargs)

        sp = self._crt_grp(self, 'SPARSE')
        if sp.variables['H'].unit != 'Ry':
            raise SileError(self.__class__.__name__ + '.read_hamiltonian requires the stored matrix to be in Ry!')

        for i in range(len(H.spin)):
            H._csr._D[:, i] = sp.variables['H'][i, :] * Ry2eV

        # fix siesta specific notation
        _mat_spin_convert(H)

        # Shift to the Fermi-level
        Ef = - self._value('Ef')[:] * Ry2eV
        H.shift(Ef)

        return H

    def read_dynamical_matrix(self, **kwargs):
        """ Returns a dynamical matrix from the underlying NetCDF file

        This assumes that the dynamical matrix is stored in the field "H" as would the
        Hamiltonian. This is counter-intuitive but is required when using PHtrans.
        """
        D = self._read_class_spin(DynamicalMatrix, **kwargs)

        sp = self._crt_grp(self, 'SPARSE')
        if sp.variables['H'].unit != 'Ry**2':
            raise SileError(self.__class__.__name__ + '.read_dynamical_matrix requires the stored matrix to be in Ry**2!')
        D._csr._D[:, 0] = sp.variables['H'][0, :] * Ry2eV ** 2

        return D

    def read_density_matrix(self, **kwargs):
        """ Returns a density matrix from the underlying NetCDF file """
        # This also adds the spin matrix
        DM = self._read_class_spin(DensityMatrix, **kwargs)

        sp = self._crt_grp(self, 'SPARSE')
        for i in range(len(DM.spin)):
            DM._csr._D[:, i] = sp.variables['DM'][i, :]

        # fix siesta specific notation
        _mat_spin_convert(DM)

        return DM

    def read_energy_density_matrix(self, **kwargs):
        """ Returns energy density matrix from the underlying NetCDF file """
        EDM = self._read_class_spin(EnergyDensityMatrix, **kwargs)

        # Shift to the Fermi-level
        Ef = self._value('Ef')[:] * Ry2eV
        if Ef.size == 1:
            Ef = np.tile(Ef, 2)

        sp = self._crt_grp(self, 'SPARSE')
        for i in range(len(EDM.spin)):
            EDM._csr._D[:, i] = sp.variables['EDM'][i, :] * Ry2eV
            if i < 2 and 'DM' in sp.variables:
                EDM._csr._D[:, i] -= sp.variables['DM'][i, :] * Ef[i]

        # fix siesta specific notation
        _mat_spin_convert(EDM)

        return EDM

    def read_force_constant(self):
        """ Reads the force-constant stored in the nc file

        Returns
        -------
        force constants : numpy.ndarray with 5 dimensions containing all the forces. The 2nd dimensions contains
                 contains the directions, and 3rd dimensions contains -/+ displacements.
        """
        if not 'FC' in self.groups:
            raise SislError(str(self) + '.read_force_constant cannot find the FC group.')
        fc = self.groups['FC']

        disp = fc.variables['disp'][0] * Bohr2Ang
        f0 = fc.variables['fa0'][:, :]
        fc = (fc.variables['fa'][:, :, :, :, :] - f0.reshape(1, 1, 1, -1, 3)) / disp
        fc[:, :, 1, :, :] *= -1
        return fc * Ry2eV / Bohr2Ang

    def grids(self):
        """ Return a list of available grids in this file. """

        grids = []
        for g in self.groups['GRID'].variables:
            grids.expand(g)

        return grids

    def read_grid(self, name, spin=0):
        """ Reads a grid in the current Siesta.nc file

        Enables the reading and processing of the grids created by Siesta

        Parameters
        ----------
        name : str
           name of the grid variable to read
        spin : int or array_like, optional
           the spin-index for retrieving one of the components. If a vector
           is passed it refers to the fraction per indexed component. I.e.
           ``[0.5, 0.5]`` will return sum of half the first two components.
           Default to the first component.
        """
        geom = self.read_geometry()

        # Shorthand
        g = self.groups['GRID']

        # Create the grid
        nx = len(g.dimensions['nx'])
        ny = len(g.dimensions['ny'])
        nz = len(g.dimensions['nz'])

        # Shorthand variable name
        v = g.variables[name]

        # Create the grid, Siesta uses periodic, always
        grid = Grid([nz, ny, nx], bc=Grid.PERIODIC, geometry=geom, dtype=v.dtype)

        # Unit-conversion
        BohrC2AngC = Bohr2Ang ** 3

        unit = {'Rho': 1. / BohrC2AngC,
                'RhoInit': 1. / BohrC2AngC,
                'RhoTot': 1. / BohrC2AngC,
                'RhoDelta': 1. / BohrC2AngC,
                'RhoXC': 1. / BohrC2AngC,
                'RhoBader': 1. / BohrC2AngC,
                'Chlocal': 1. / BohrC2AngC,
        }.get(name, 1.)

        if len(v[:].shape) == 3:
            grid.grid = v[:, :, :] * unit
        elif isinstance(spin, Integral):
            grid.grid = v[spin, :, :, :] * unit
        else:
            if len(spin) > v.shape[0]:
                raise SileError(self.__class__.__name__ + '.read_grid requires spin to be an integer or '
                                'an array of length equal to the number of spin components.')
            grid.grid[:, :, :] = v[0, :, :, :] * (spin[0] * unit)
            for i, scale in enumerate(spin[1:]):
                grid.grid[:, :, :] += v[1+i, :, :, :] * (scale * unit)

        try:
            if v.unit == 'Ry':
                # Convert to ev
                grid *= Ry2eV
        except:
            # Allowed pass due to pythonic reading
            pass

        # Read the grid, we want the z-axis to be the fastest
        # looping direction, hence x,y,z == 0,1,2
        grid.grid = np.copy(np.swapaxes(grid.grid, 0, 2), order='C')

        return grid

    def write_basis(self, atom):
        """ Write the current atoms orbitals as the basis

        Parameters
        ----------
        atom : Atoms
           atom specifications to write.
        """
        sile_raise_write(self)
        bs = self._crt_grp(self, 'BASIS')

        # Create variable of basis-indices
        b = self._crt_var(bs, 'basis', 'i4', ('na_u',))
        b.info = "Basis of each atom by ID"

        for isp, (a, ia) in enumerate(atom.iter(True)):
            b[ia] = isp + 1
            if a.tag in bs.groups:
                # Assert the file sizes
                if bs.groups[a.tag].Number_of_orbitals != a.no:
                    raise ValueError('File {} has erroneous data '
                                     'in regards of the already stored dimensions.'.format(self.file))
            else:
                ba = bs.createGroup(a.tag)
                ba.ID = np.int32(isp + 1)
                ba.Atomic_number = np.int32(a.Z)
                ba.Mass = a.mass
                ba.Label = a.tag
                ba.Element = a.symbol
                ba.Number_of_orbitals = np.int32(a.no)

    def write_geometry(self, geom):
        """ Creates the NetCDF file and writes the geometry information """
        sile_raise_write(self)

        # Create initial dimensions
        self._crt_dim(self, 'one', 1)
        self._crt_dim(self, 'n_s', np.prod(geom.nsc, dtype=np.int32))
        self._crt_dim(self, 'xyz', 3)
        self._crt_dim(self, 'no_s', np.prod(geom.nsc, dtype=np.int32) * geom.no)
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
        self.write_basis(geom.atom)

        # Store the lasto variable as the remaining thing to do
        self.variables['lasto'][:] = geom.lasto + 1

    def write_overlap(self, **kwargs):
        """ Write the overlap matrix to the NetCDF file """
        raise NotImplementedError('Currently not implemented')

    def write_hamiltonian(self, H, **kwargs):
        """ Writes Hamiltonian model to file

        Parameters
        ----------
        H : Hamiltonian
           the model to be saved in the NC file
        Ef : float, optional
           the Fermi level of the electronic structure (in eV), default to 0.
        """
        csr = H._csr.copy()
        if csr.nnz == 0:
            raise SileError(str(self) + '.write_hamiltonian cannot write a zero element sparse matrix!')

        # Convert to siesta CSR
        _csr_to_siesta(H.geometry, csr)
        csr.finalize()

        # Ensure that the geometry is written
        self.write_geometry(H.geometry)

        self._crt_dim(self, 'spin', len(H.spin))

        if H.dkind != 'f':
            raise NotImplementedError('Currently we only allow writing a floating point Hamiltonian to the Siesta format')

        v = self._crt_var(self, 'Ef', 'f8', ('one',))
        v.info = 'Fermi level'
        v.unit = 'Ry'
        v[:] = kwargs.get('Ef', 0.) / Ry2eV
        v = self._crt_var(self, 'Qtot', 'f8', ('one',))
        v.info = 'Total charge'
        v[0] = kwargs.get('Q', kwargs.get('Qtot', H.geometry.q0))

        # Append the sparsity pattern
        # Create basis group
        sp = self._crt_grp(self, 'SPARSE')

        self._crt_dim(sp, 'nnzs', csr.col.shape[0])
        v = self._crt_var(sp, 'n_col', 'i4', ('no_u',))
        v.info = "Number of non-zero elements per row"
        v[:] = csr.ncol[:]
        v = self._crt_var(sp, 'list_col', 'i4', ('nnzs',),
                          chunksizes=(len(csr.col),), **self._cmp_args)
        v.info = "Supercell column indices in the sparse format"
        v[:] = csr.col[:] + 1  # correct for fortran indices
        v = self._crt_var(sp, 'isc_off', 'i4', ('n_s', 'xyz'))
        v.info = "Index of supercell coordinates"
        v[:, :] = _siesta.siesta_sc_off(*H.geometry.nsc).T

        # Save tight-binding parameters
        v = self._crt_var(sp, 'S', 'f8', ('nnzs',),
                          chunksizes=(len(csr.col),), **self._cmp_args)
        v.info = "Overlap matrix"
        if H.orthogonal:
            # We need to create the orthogonal pattern
            tmp = csr.copy(dims=[0])
            tmp.empty(keep_nnz=True)
            for i in range(tmp.shape[0]):
                tmp[i, i] = 1.

            if tmp.nnz != H.nnz:
                # We have added more stuff, something that we currently do not allow.
                raise ValueError(self.__class__.__name__ + '.write_hamiltonian '
                                 'is trying to write a Hamiltonian in Siesta format with '
                                 'not all on-site terms defined. Please correct. '
                                 'I.e. add explicitly *all* on-site terms.')

            v[:] = tmp._D[:, 0]
            del tmp
        else:
            v[:] = csr._D[:, H.S_idx]
        v = self._crt_var(sp, 'H', 'f8', ('spin', 'nnzs'),
                          chunksizes=(1, len(csr.col)), **self._cmp_args)
        v.info = "Hamiltonian"
        v.unit = "Ry"
        _mat_spin_convert(csr, H.spin)
        for i in range(len(H.spin)):
            v[i, :] = csr._D[:, i] / Ry2eV

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

    def write_density_matrix(self, DM, **kwargs):
        """ Writes density matrix model to file

        Parameters
        ----------
        DM : DensityMatrix
           the model to be saved in the NC file
        """
        DM.finalize()
        csr = DM._csr.copy()
        if csr.nnz == 0:
            raise SileError(str(self) + '.write_density_matrix cannot write a zero element sparse matrix!')

        # Convert to siesta CSR
        _csr_to_siesta(DM.geometry, csr)
        csr.finalize()

        # Ensure that the geometry is written
        self.write_geometry(DM.geom)

        self._crt_dim(self, 'spin', len(DM.spin))

        if DM.dkind != 'f':
            raise NotImplementedError('Currently we only allow writing a floating point density matrix to the Siesta format')

        v = self._crt_var(self, 'Qtot', 'f8', ('one',))
        v.info = 'Total charge'
        v[:] = np.sum(DM.geom.atom.q0)
        if 'Qtot' in kwargs:
            v[:] = kwargs['Qtot']
        if 'Q' in kwargs:
            v[:] = kwargs['Q']

        # Append the sparsity pattern
        # Create basis group
        sp = self._crt_grp(self, 'SPARSE')

        self._crt_dim(sp, 'nnzs', csr.col.shape[0])
        v = self._crt_var(sp, 'n_col', 'i4', ('no_u',))
        v.info = "Number of non-zero elements per row"
        v[:] = csr.ncol[:]
        v = self._crt_var(sp, 'list_col', 'i4', ('nnzs',),
                          chunksizes=(len(csr.col),), **self._cmp_args)
        v.info = "Supercell column indices in the sparse format"
        v[:] = csr.col[:] + 1  # correct for fortran indices
        v = self._crt_var(sp, 'isc_off', 'i4', ('n_s', 'xyz'))
        v.info = "Index of supercell coordinates"
        v[:, :] = _siesta.siesta_sc_off(*DM.geometry.nsc).T

        # Save sparse matrices
        v = self._crt_var(sp, 'S', 'f8', ('nnzs',),
                          chunksizes=(len(csr.col),), **self._cmp_args)
        v.info = "Overlap matrix"
        if DM.orthogonal:
            # We need to create the orthogonal pattern
            tmp = csr.copy(dims=[0])
            tmp.empty(keep_nnz=True)
            for i in range(tmp.shape[0]):
                tmp[i, i] = 1.

            if tmp.nnz != DM.nnz:
                # We have added more stuff, something that we currently do not allow.
                raise ValueError(self.__class__.__name__ + '.write_density_matrix '
                                 'is trying to write a density matrix in Siesta format with '
                                 'not all on-site terms defined. Please correct. '
                                 'I.e. add explicitly *all* on-site terms.')

            v[:] = tmp._D[:, 0]
            del tmp
        else:
            v[:] = csr._D[:, DM.S_idx]
        v = self._crt_var(sp, 'DM', 'f8', ('spin', 'nnzs'),
                          chunksizes=(1, len(csr.col)), **self._cmp_args)
        v.info = "Density matrix"
        _mat_spin_convert(csr, DM.spin)
        for i in range(len(DM.spin)):
            v[i, :] = csr._D[:, i]

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

    def write_energy_density_matrix(self, EDM, **kwargs):
        """ Writes energy density matrix model to file

        Parameters
        ----------
        EDM : EnergyDensityMatrix
           the model to be saved in the NC file
        """
        EDM.finalize()
        csr = EDM._csr.copy()
        if csr.nnz == 0:
            raise SileError(str(self) + '.write_energy_density_matrix cannot write a zero element sparse matrix!')

        # Convert to siesta CSR
        _csr_to_siesta(EDM.geometry, csr)
        csr.finalize()

        # Ensure that the geometry is written
        self.write_geometry(EDM.geometry)

        self._crt_dim(self, 'spin', len(EDM.spin))

        if EDM.dkind != 'f':
            raise NotImplementedError('Currently we only allow writing a floating point density matrix to the Siesta format')

        v = self._crt_var(self, 'Ef', 'f8', ('one',))
        v.info = 'Fermi level'
        v.unit = 'Ry'
        v[:] = kwargs.get('Ef', 0.) / Ry2eV
        v = self._crt_var(self, 'Qtot', 'f8', ('one',))
        v.info = 'Total charge'
        v[:] = np.sum(EDM.geometry.atom.q0)
        if 'Qtot' in kwargs:
            v[:] = kwargs['Qtot']
        if 'Q' in kwargs:
            v[:] = kwargs['Q']

        # Append the sparsity pattern
        # Create basis group
        sp = self._crt_grp(self, 'SPARSE')

        self._crt_dim(sp, 'nnzs', csr.col.shape[0])
        v = self._crt_var(sp, 'n_col', 'i4', ('no_u',))
        v.info = "Number of non-zero elements per row"
        v[:] = csr.ncol[:]
        v = self._crt_var(sp, 'list_col', 'i4', ('nnzs',),
                          chunksizes=(len(csr.col),), **self._cmp_args)
        v.info = "Supercell column indices in the sparse format"
        v[:] = csr.col[:] + 1  # correct for fortran indices
        v = self._crt_var(sp, 'isc_off', 'i4', ('n_s', 'xyz'))
        v.info = "Index of supercell coordinates"
        v[:, :] = _siesta.siesta_sc_off(*EDM.geometry.nsc).T

        # Save tight-binding parameters
        v = self._crt_var(sp, 'S', 'f8', ('nnzs',),
                          chunksizes=(len(csr.col),), **self._cmp_args)
        v.info = "Overlap matrix"
        if EDM.orthogonal:
            # We need to create the orthogonal pattern
            tmp = csr.copy(dims=[0])
            tmp.empty(keep_nnz=True)
            for i in range(tmp.shape[0]):
                tmp[i, i] = 1.

            if tmp.nnz != EDM.nnz:
                # We have added more stuff, something that we currently do not allow.
                raise ValueError(self.__class__.__name__ + '.write_energy_density_matrix '
                                 'is trying to write a density matrix in Siesta format with '
                                 'not all on-site terms defined. Please correct. '
                                 'I.e. add explicitly *all* on-site terms.')

            v[:] = tmp._D[:, 0]
            del tmp
        else:
            v[:] = csr._D[:, EDM.S_idx]
        v = self._crt_var(sp, 'EDM', 'f8', ('spin', 'nnzs'),
                          chunksizes=(1, len(csr.col)), **self._cmp_args)
        v.info = "Energy density matrix"
        v.unit = "Ry"
        _mat_spin_convert(csr, EDM.spin)
        for i in range(len(EDM.spin)):
            v[i, :] = csr._D[:, i] / Ry2eV

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

    def write_dynamical_matrix(self, D, **kwargs):
        """ Writes dynamical matrix model to file

        Parameters
        ----------
        D : DynamicalMatrix
           the model to be saved in the NC file
        """
        D.finalize()
        csr = D._csr.copy()
        if csr.nnz == 0:
            raise SileError(str(self) + '.write_dynamical_matrix cannot write a zero element sparse matrix!')

        # Convert to siesta CSR
        _csr_to_siesta(D.geometry, csr)
        csr.finalize()

        # Ensure that the geometry is written
        self.write_geometry(D.geometry)

        self._crt_dim(self, 'spin', 1)

        if D.dkind != 'f':
            raise NotImplementedError('Currently we only allow writing a floating point dynamical matrix to the Siesta format')

        v = self._crt_var(self, 'Ef', 'f8', ('one',))
        v.info = 'Fermi level'
        v.unit = 'Ry'
        v[:] = 0.
        v = self._crt_var(self, 'Qtot', 'f8', ('one',))
        v.info = 'Total charge'
        v.unit = 'e'
        v[:] = 0.

        # Append the sparsity pattern
        # Create basis group
        sp = self._crt_grp(self, 'SPARSE')

        self._crt_dim(sp, 'nnzs', csr.col.shape[0])
        v = self._crt_var(sp, 'n_col', 'i4', ('no_u',))
        v.info = "Number of non-zero elements per row"
        v[:] = csr.ncol[:]
        v = self._crt_var(sp, 'list_col', 'i4', ('nnzs',),
                          chunksizes=(len(csr.col),), **self._cmp_args)
        v.info = "Supercell column indices in the sparse format"
        v[:] = csr.col[:] + 1  # correct for fortran indices
        v = self._crt_var(sp, 'isc_off', 'i4', ('n_s', 'xyz'))
        v.info = "Index of supercell coordinates"
        v[:, :] = _siesta.siesta_sc_off(*D.geometry.nsc).T

        # Save tight-binding parameters
        v = self._crt_var(sp, 'S', 'f8', ('nnzs',),
                          chunksizes=(len(csr.col),), **self._cmp_args)
        v.info = "Overlap matrix"
        if D.orthogonal:
            # We need to create the orthogonal pattern
            tmp = csr.copy(dims=[0])
            tmp.empty(keep_nnz=True)
            for i in range(tmp.shape[0]):
                tmp[i, i] = 1.

            if tmp.nnz != D.nnz:
                # We have added more stuff, something that we currently do not allow.
                raise ValueError(self.__class__.__name__ + '.write_dynamical_matrix '
                                 'is trying to write a Hamiltonian in Siesta format with '
                                 'not all on-site terms defined. Please correct. '
                                 'I.e. add explicitly *all* on-site terms.')

            v[:] = tmp._D[:, 0]
            del tmp
        else:
            v[:] = csr._D[:, D.S_idx]
        v = self._crt_var(sp, 'H', 'f8', ('spin', 'nnzs'),
                          chunksizes=(1, len(csr.col)), **self._cmp_args)
        v.info = "Dynamical matrix"
        v.unit = "Ry**2"
        v[0, :] = csr._D[:, 0] / Ry2eV ** 2

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

    def ArgumentParser(self, p=None, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geometry().ArgumentParser(p, *args, **newkw)


add_sile('nc', ncSileSiesta)
