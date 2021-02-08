from numbers import Integral
from itertools import product, groupby
from collections import deque
import numpy as np
from numpy import pi

try:
    from . import _siesta
    found_module = True
except Exception as e:
    print(e)
    found_module = False

from ..sile import add_sile, SileError
from .sile import SileBinSiesta
from sisl._internal import set_module
from sisl.messages import warn, SislError

from ._help import *
import sisl._array as _a
from sisl import Geometry, Atom, Atoms, SuperCell, Grid, SparseCSR
from sisl import AtomicOrbital
from sisl.sparse import _ncol_to_indptr
from sisl.unit.siesta import unit_convert
from sisl.physics.sparse import SparseOrbitalBZ
from sisl.physics import Hamiltonian, DensityMatrix, EnergyDensityMatrix
from sisl.physics.overlap import Overlap
from sisl.physics.electron import EigenstateElectron


__all__ = ['tshsSileSiesta', 'onlysSileSiesta', 'tsdeSileSiesta']
__all__ += ['hsxSileSiesta', 'dmSileSiesta']
__all__ += ['wfsxSileSiesta']
__all__ += ['gridSileSiesta']
__all__ += ['tsgfSileSiesta']


_Bohr2Ang = unit_convert('Bohr', 'Ang')
_Ry2eV = unit_convert('Ry', 'eV')
_eV2Ry = unit_convert('eV', 'Ry')


def _bin_check(obj, method, message):
    ierr = _siesta.io_m.iostat_query()
    if ierr != 0:
        raise SileError(f'{str(obj)}.{method} {message} (ierr={ierr})')


def _toF(array, dtype, scale=None):
    if scale is None:
        return array.astype(dtype, order='F', copy=False)
    elif array.dtype == dtype and array.flags.f_contiguous:
        # no need to copy since the order is correct
        return array * scale

    # We have to handle cases
    out = np.empty_like(array, dtype, order='F')
    np.multiply(array, scale, out=out)
    return out


def _geometry_align(geom_b, geom_u, cls, method):
    """ Routine used to align two geometries

    There are a few twists in this since the fdf-reads will automatically
    try and pass a geometry from the output files.
    In cases where the *.ion* files are non-existing this will
    result in a twist.

    This routine will select and return a merged Geometry which
    fulfills the correct number of atoms and orbitals.

    However, if the input geometries have mis-matching number
    of atoms a SislError will be raised.

    Parameters
    ----------
    geom_b : Geometry from binary file
    geom_u : Geometry supplied by user

    Raises
    ------
    SislError : if the geometries have non-equal atom count
    """
    if geom_b is None:
        return geom_u
    elif geom_u is None:
        return geom_b

    # Default to use the users geometry
    geom = geom_u

    is_copy = False
    def get_copy(geom, is_copy):
        if is_copy:
            return geom, True
        return geom.copy(), True

    if geom_b.na != geom.na:
        # we have no way of solving this issue...
        raise SileError(f"{cls.__name__}.{method} could not use the passed geometry as the "
                        f"of atoms is not consistent, user-atoms={geom_u.na}, file-atoms={geom_b.na}.")

    # Try and figure out what to do
    if not np.allclose(geom_b.xyz, geom.xyz):
        warn(f"{cls.__name__}.{method} has mismatched atomic coordinates, will copy geometry and use file XYZ.")
        geom, is_copy = get_copy(geom, is_copy)
        geom.xyz[:, :] = geom_b.xyz[:, :]
    if not np.allclose(geom_b.sc.cell, geom.sc.cell):
        warn(f"{cls.__name__}.{method} has non-equal lattice vectors, will copy geometry and use file lattice.")
        geom, is_copy = get_copy(geom, is_copy)
        geom.sc.cell[:, :] = geom_b.sc.cell[:, :]
    if not np.array_equal(geom_b.nsc, geom.nsc):
        warn(f"{cls.__name__}.{method} has non-equal number of supercells, will copy geometry and use file supercell count.")
        geom, is_copy = get_copy(geom, is_copy)
        geom.set_nsc(geom_b.nsc)

    # Now for the difficult part.
    # If there is a mismatch in the number of orbitals we will
    # prefer to use the user-supplied atomic species, but fill with
    # *random* orbitals
    if not np.array_equal(geom_b.atoms.orbitals, geom.atoms.orbitals):
        warn(f"{cls.__name__}.{method} has non-equal number of orbitals per atom, will correct with *empty* orbitals.")
        geom, is_copy = get_copy(geom, is_copy)

        # Now create a new atom specie with the correct number of orbitals
        norbs = geom_b.atoms.orbitals[:]
        atoms = Atoms([geom.atoms[i].copy(orbitals=[-1.] * norbs[i]) for i in range(geom.na)])
        geom._atoms = atoms

    return geom


@set_module("sisl.io.siesta")
class onlysSileSiesta(SileBinSiesta):
    """ Geometry and overlap matrix """

    def read_supercell(self):
        """ Returns a SuperCell object from a TranSiesta file """
        n_s = _siesta.read_tshs_sizes(self.file)[3]
        _bin_check(self, 'read_supercell', 'could not read sizes.')
        arr = _siesta.read_tshs_cell(self.file, n_s)
        _bin_check(self, 'read_supercell', 'could not read cell.')
        nsc = np.array(arr[0], np.int32)
        # We have to transpose since the data is read *as-is*
        # The cell in fortran files are (:, A1)
        # after reading this is still obeyed (regardless of order)
        # So we transpose to get it C-like
        # Note that care must be taken for the different data-structures
        # In particular not all data needs to be transposed (sparse H and S)
        cell = arr[1].T * _Bohr2Ang
        return SuperCell(cell, nsc=nsc)

    def read_geometry(self, geometry=None):
        """ Returns Geometry object from a TranSiesta file """

        # Read supercell
        sc = self.read_supercell()

        na = _siesta.read_tshs_sizes(self.file)[1]
        _bin_check(self, 'read_geometry', 'could not read sizes.')
        arr = _siesta.read_tshs_geom(self.file, na)
        _bin_check(self, 'read_geometry', 'could not read geometry.')
        # see onlysSileSiesta.read_supercell for .T
        xyz = arr[0].T * _Bohr2Ang
        lasto = arr[1]

        # Since the TSHS file does not contain species information
        # and/or other stuff we *can* reuse an existing
        # geometry which contains the correct atomic numbers etc.
        orbs = np.diff(lasto)
        if geometry is None:
            # Create all different atoms...
            # The TSHS file does not contain the
            # atomic numbers, so we will just
            # create them individually

            # Get unique orbitals
            uorb = np.unique(orbs)
            # Create atoms
            atoms = []
            for Z, orb in enumerate(uorb):
                atoms.append(Atom(Z+1, [-1] * orb))

            def get_atom(atoms, orbs):
                for atom in atoms:
                    if atom.no == orbs:
                        return atom

            atom = []
            for orb in orbs:
                atom.append(get_atom(atoms, orb))

        else:
            # Create a new geometry with the correct atomic numbers
            atom = []
            for ia, no in zip(geometry, orbs):
                a = geometry.atoms[ia]
                if a.no == no:
                    atom.append(a)
                else:
                    # correct atom
                    atom.append(a.__class__(a.Z, [-1. for io in range(no)], mass=a.mass, tag=a.tag))

        # Create and return geometry object
        return Geometry(xyz, atom, sc=sc)

    def read_overlap(self, **kwargs):
        """ Returns the overlap matrix from the TranSiesta file """
        tshs_g = self.read_geometry()
        geom = _geometry_align(tshs_g, kwargs.get('geometry', tshs_g), self.__class__, 'read_overlap')

        # read the sizes used...
        sizes = _siesta.read_tshs_sizes(self.file)
        _bin_check(self, 'read_overlap', 'could not read sizes.')
        # see onlysSileSiesta.read_supercell for .T
        isc = _siesta.read_tshs_cell(self.file, sizes[3])[2].T
        _bin_check(self, 'read_overlap', 'could not read cell.')
        no = sizes[2]
        nnz = sizes[4]
        ncol, col, dS = _siesta.read_tshs_s(self.file, no, nnz)
        _bin_check(self, 'read_overlap', 'could not read overlap matrix.')

        # Create the Hamiltonian container
        S = Overlap(geom, nnzpr=1)

        # Create the new sparse matrix
        S._csr.ncol = ncol.astype(np.int32, copy=False)
        S._csr.ptr = _ncol_to_indptr(ncol)
        # Correct fortran indices
        S._csr.col = col.astype(np.int32, copy=False) - 1
        S._csr._nnz = len(col)

        S._csr._D = _a.emptyd([nnz, 1])
        S._csr._D[:, 0] = dS[:]

        # Convert to sisl supercell
        # equivalent as _csr_from_siesta with explicit isc from file
        _csr_from_sc_off(S.geometry, isc, S._csr)

        # In siesta the matrix layout is written in CSC format
        # due to fortran indexing, this means that we need to transpose
        # to get it to correct layout.
        return S.transpose(sort=kwargs.get("sort", True))

    def read_fermi_level(self):
        r""" Query the Fermi-level contained in the file

        Returns
        -------
        Ef : fermi-level of the system
        """
        Ef = _siesta.read_tshs_ef(self.file) * _Ry2eV
        _bin_check(self, 'read_fermi_level', 'could not read fermi-level.')
        return Ef


@set_module("sisl.io.siesta")
class tshsSileSiesta(onlysSileSiesta):
    """ Geometry, Hamiltonian and overlap matrix file """

    def read_hamiltonian(self, **kwargs):
        """ Returns the electronic structure from the siesta.TSHS file """
        tshs_g = self.read_geometry()
        geom = _geometry_align(tshs_g, kwargs.get('geometry', tshs_g), self.__class__, 'read_hamiltonian')

        # read the sizes used...
        sizes = _siesta.read_tshs_sizes(self.file)
        _bin_check(self, 'read_hamiltonian', 'could not read sizes.')
        # see onlysSileSiesta.read_supercell for .T
        isc = _siesta.read_tshs_cell(self.file, sizes[3])[2].T
        _bin_check(self, 'read_hamiltonian', 'could not read cell.')
        spin = sizes[0]
        no = sizes[2]
        nnz = sizes[4]
        ncol, col, dH, dS = _siesta.read_tshs_hs(self.file, spin, no, nnz)
        _bin_check(self, 'read_hamiltonian', 'could not read Hamiltonian and overlap matrix.')

        # Check whether it is an orthogonal basis set
        orthogonal = np.abs(dS).sum() == geom.no

        # Create the Hamiltonian container
        H = Hamiltonian(geom, spin, nnzpr=1, orthogonal=orthogonal)

        # Create the new sparse matrix
        H._csr.ncol = ncol.astype(np.int32, copy=False)
        H._csr.ptr = _ncol_to_indptr(ncol)
        # Correct fortran indices
        H._csr.col = col.astype(np.int32, copy=False) - 1
        H._csr._nnz = len(col)

        if orthogonal:
            H._csr._D = _a.emptyd([nnz, spin])
            H._csr._D[:, :] = dH[:, :] * _Ry2eV
        else:
            H._csr._D = _a.emptyd([nnz, spin+1])
            H._csr._D[:, :spin] = dH[:, :] * _Ry2eV
            H._csr._D[:, spin] = dS[:]

        _mat_spin_convert(H)

        # Convert to sisl supercell
        # equivalent as _csr_from_siesta with explicit isc from file
        _csr_from_sc_off(H.geometry, isc, H._csr)

        # Find all indices where dS == 1 (remember col is in fortran indices)
        idx = col[np.isclose(dS, 1.).nonzero()[0]]
        if np.any(idx > no):
            print(f'Number of orbitals: {no}')
            print(idx)
            raise SileError(str(self) + '.read_hamiltonian could not assert '
                            'the supercell connections in the primary unit-cell.')

        # see onlysSileSiesta.read_overlap for .transpose()
        # For H, DM and EDM we also need to Hermitian conjugate it.
        return H.transpose(spin=False, sort=kwargs.get("sort", True))

    def write_hamiltonian(self, H, **kwargs):
        """ Writes the Hamiltonian to a siesta.TSHS file """
        # we sort below, so no need to do it here
        # see onlysSileSiesta.read_overlap for .transpose()
        csr = H.transpose(spin=False, sort=False)._csr
        if csr.nnz == 0:
            raise SileError(str(self) + '.write_hamiltonian cannot write '
                            'a zero element sparse matrix!')

        # Convert to siesta CSR
        _csr_to_siesta(H.geometry, csr)
        csr.finalize(sort=kwargs.get("sort", True))
        _mat_spin_convert(csr, H.spin)

        # Extract the data to pass to the fortran routine
        cell = H.geometry.cell
        xyz = H.geometry.xyz

        # Get H and S
        if H.orthogonal:
            h = csr._D
            s = csr.diags(1., dim=1)
            # Ensure all data is correctly formatted (i.e. have the same sparsity pattern)
            s.align(csr)
            s.finalize(sort=kwargs.get("sort", True))
            if s.nnz != len(h):
                raise SislError('The diagonal elements of your orthogonal Hamiltonian '
                                'have not been defined, this is a requirement.')
            s = s._D[:, 0]
        else:
            h = csr._D[:, :H.S_idx]
            s = csr._D[:, H.S_idx]

        # Get shorter variants
        nsc = H.geometry.nsc[:].astype(np.int32)
        isc = _siesta.siesta_sc_off(*nsc)

        # see onlysSileSiesta.read_supercell for .T
        _siesta.write_tshs_hs(self.file, nsc[0], nsc[1], nsc[2],
                              cell.T / _Bohr2Ang, xyz.T / _Bohr2Ang, H.geometry.firsto,
                              csr.ncol, csr.col + 1,
                              _toF(h, np.float64, _eV2Ry), _toF(s, np.float64),
                              isc)
        _bin_check(self, 'write_hamiltonian', 'could not write Hamiltonian and overlap matrix.')


@set_module("sisl.io.siesta")
class dmSileSiesta(SileBinSiesta):
    """ Density matrix file """

    def read_density_matrix(self, **kwargs):
        """ Returns the density matrix from the siesta.DM file """

        # Now read the sizes used...
        spin, no, nsc, nnz = _siesta.read_dm_sizes(self.file)
        _bin_check(self, 'read_density_matrix', 'could not read density matrix sizes.')

        ncol, col, dDM = _siesta.read_dm(self.file, spin, no, nsc, nnz)
        _bin_check(self, 'read_density_matrix', 'could not read density matrix.')

        # Try and immediately attach a geometry
        geom = kwargs.get('geometry', kwargs.get('geom', None))
        if geom is None:
            # We truly, have no clue,
            # Just generate a boxed system
            xyz = [[x, 0, 0] for x in range(no)]
            sc = SuperCell([no, 1, 1], nsc=nsc)
            geom = Geometry(xyz, Atom(1), sc=sc)

        if nsc[0] != 0 and np.any(geom.nsc != nsc):
            # We have to update the number of supercells!
            geom.set_nsc(nsc)

        if geom.no != no:
            raise SileError(str(self) + '.read_density_matrix could not use the '
                            'passed geometry as the number of atoms or orbitals is '
                            'inconsistent with DM file.')

        # Create the density matrix container
        DM = DensityMatrix(geom, spin, nnzpr=1, dtype=np.float64, orthogonal=False)

        # Create the new sparse matrix
        DM._csr.ncol = ncol.astype(np.int32, copy=False)
        DM._csr.ptr = _ncol_to_indptr(ncol)
        # Correct fortran indices
        DM._csr.col = col.astype(np.int32, copy=False) - 1
        DM._csr._nnz = len(col)

        DM._csr._D = _a.emptyd([nnz, spin+1])
        DM._csr._D[:, :spin] = dDM[:, :]
        # DM file does not contain overlap matrix... so neglect it for now.
        DM._csr._D[:, spin] = 0.

        _mat_spin_convert(DM)

        # Convert the supercells to sisl supercells
        if nsc[0] != 0 or geom.no_s >= col.max():
            _csr_from_siesta(geom, DM._csr)
        else:
            warn(str(self) + '.read_density_matrix may result in a wrong sparse pattern!')

        return DM.transpose(spin=False, sort=kwargs.get("sort", True))

    def write_density_matrix(self, DM, **kwargs):
        """ Writes the density matrix to a siesta.DM file """
        csr = DM.transpose(spin=False, sort=False)._csr
        # This ensures that we don't have any *empty* elements
        if csr.nnz == 0:
            raise SileError(str(self) + '.write_density_matrix cannot write '
                            'a zero element sparse matrix!')

        _csr_to_siesta(DM.geometry, csr)
        # We do not really need to sort this one, but we do for consistency
        # of the interface.
        csr.finalize(sort=kwargs.get("sort", True))
        _mat_spin_convert(csr, DM.spin)

        # Get DM
        if DM.orthogonal:
            dm = csr._D
        else:
            dm = csr._D[:, :DM.S_idx]

        # Ensure shapes (say if only 1 spin)
        dm.shape = (-1, len(DM.spin))

        nsc = DM.geometry.sc.nsc.astype(np.int32)

        _siesta.write_dm(self.file, nsc, csr.ncol, csr.col + 1, _toF(dm, np.float64))
        _bin_check(self, 'write_density_matrix', 'could not write density matrix.')


@set_module("sisl.io.siesta")
class tsdeSileSiesta(dmSileSiesta):
    """ Non-equilibrium density matrix and energy density matrix file """

    def read_energy_density_matrix(self, **kwargs):
        """ Returns the energy density matrix from the siesta.DM file """

        # Now read the sizes used...
        spin, no, nsc, nnz = _siesta.read_tsde_sizes(self.file)
        _bin_check(self, 'read_energy_density_matrix', 'could not read energy density matrix sizes.')
        ncol, col, dEDM = _siesta.read_tsde_edm(self.file, spin, no, nsc, nnz)
        _bin_check(self, 'read_energy_density_matrix', 'could not read energy density matrix.')

        # Try and immediately attach a geometry
        geom = kwargs.get('geometry', kwargs.get('geom', None))
        if geom is None:
            # We truly, have no clue,
            # Just generate a boxed system
            xyz = [[x, 0, 0] for x in range(no)]
            sc = SuperCell([no, 1, 1], nsc=nsc)
            geom = Geometry(xyz, Atom(1), sc=sc)

        if nsc[0] != 0 and np.any(geom.nsc != nsc):
            # We have to update the number of supercells!
            geom.set_nsc(nsc)

        if geom.no != no:
            raise SileError(str(self) + '.read_energy_density_matrix could '
                            'not use the passed geometry as the number of atoms or orbitals '
                            'is inconsistent with DM file.')

        # Create the energy density matrix container
        EDM = EnergyDensityMatrix(geom, spin, nnzpr=1, dtype=np.float64, orthogonal=False)

        # Create the new sparse matrix
        EDM._csr.ncol = ncol.astype(np.int32, copy=False)
        EDM._csr.ptr = _ncol_to_indptr(ncol)
        # Correct fortran indices
        EDM._csr.col = col.astype(np.int32, copy=False) - 1
        EDM._csr._nnz = len(col)

        EDM._csr._D = _a.emptyd([nnz, spin+1])
        EDM._csr._D[:, :spin] = dEDM[:, :] * _Ry2eV
        # EDM file does not contain overlap matrix... so neglect it for now.
        EDM._csr._D[:, spin] = 0.

        _mat_spin_convert(EDM)

        # Convert the supercells to sisl supercells
        if nsc[0] != 0 or geom.no_s >= col.max():
            _csr_from_siesta(geom, EDM._csr)
        else:
            warn(str(self) + '.read_energy_density_matrix may result in a wrong sparse pattern!')

        return EDM.transpose(spin=False, sort=kwargs.get("sort", True))

    def read_fermi_level(self):
        r""" Query the Fermi-level contained in the file

        Returns
        -------
        Ef : fermi-level of the system
        """
        Ef = _siesta.read_tsde_ef(self.file) * _Ry2eV
        _bin_check(self, 'read_fermi_level', 'could not read fermi-level.')
        return Ef

    def write_density_matrices(self, DM, EDM, Ef=0., **kwargs):
        r""" Writes the density matrix to a siesta.DM file

        Parameters
        ----------
        DM : DensityMatrix
           density matrix to write to the file
        EDM : EnergyDensityMatrix
           energy density matrix to write to the file
        Ef : float, optional
           fermi-level to be contained
        """
        DMcsr = DM.transpose(spin=False, sort=False)._csr
        EDMcsr = EDM.transpose(spin=False, sort=False)._csr
        DMcsr.align(EDMcsr)
        EDMcsr.align(DMcsr)

        if DMcsr.nnz == 0:
            raise SileError(str(self) + '.write_density_matrices cannot write '
                            'a zero element sparse matrix!')

        _csr_to_siesta(DM.geometry, DMcsr)
        _csr_to_siesta(DM.geometry, EDMcsr)
        sort = kwargs.get("sort", True)
        DMcsr.finalize(sort=sort)
        EDMcsr.finalize(sort=sort)
        _mat_spin_convert(DMcsr, DM.spin)
        _mat_spin_convert(EDMcsr, EDM.spin)

        # Ensure everything is correct
        if not (np.allclose(DMcsr.ncol, EDMcsr.ncol) and
                np.allclose(DMcsr.col, EDMcsr.col)):
            raise ValueError(str(self) + '.write_density_matrices got non compatible '
                             'DM and EDM matrices.')

        if DM.orthogonal:
            dm = DMcsr._D
        else:
            dm = DMcsr._D[:, :DM.S_idx]
        if EDM.orthogonal:
            edm = EDMcsr._D
        else:
            edm = EDMcsr._D[:, :EDM.S_idx]

        nsc = DM.geometry.sc.nsc.astype(np.int32)

        _siesta.write_tsde_dm_edm(self.file, nsc, DMcsr.ncol, DMcsr.col + 1,
                                  _toF(dm, np.float64),
                                  _toF(edm, np.float64, _eV2Ry), Ef * _eV2Ry)
        _bin_check(self, 'write_density_matrices', 'could not write DM + EDM matrices.')


@set_module("sisl.io.siesta")
class hsxSileSiesta(SileBinSiesta):
    """ Hamiltonian and overlap matrix file

    This file does not contain all information regarding the system.

    To ensure no errors are being raised one should pass a `Geometry` with
    correct number of atoms and correct number of supercells.
    The number of orbitals will be updated in the returned matrices geometry.

    >>> hsx = hsxSileSiesta("siesta.HSX")
    >>> HS = hsx.read_hamiltonian() # may fail
    >>> HS = hsx.read_hamiltonian(geometry=<>) # should run correctly if above satisfied

    Users are adviced to use the `tshsSileSiesta` instead since that correctly contains
    all information.
    """

    def _xij2system(self, xij, geometry=None):
        """ Create a new geometry with *correct* nsc and somewhat correct xyz

        Parameters
        ----------
        xij : SparseCSR
            orbital distances
        geometry : Geometry, optional
            passed geometry
        """
        def get_geom_handle(xij):
            atoms = self._read_atoms()
            if not atoms is None:
                return Geometry(np.zeros([len(atoms), 3]), atoms)

            N = len(xij)
            # convert csr to dok format
            row = (xij.ncol > 0).nonzero()[0]
            # Now we have [0 0 0 0 1 1 1 1 2 2 ... no-1 no-1]
            row = np.repeat(row, xij.ncol[row])
            col = xij.col

            # Parse xij to correct geometry
            # first figure out all zeros (i.e. self-atom-orbitals)
            idx0 = np.isclose(np.fabs(xij._D).sum(axis=1), 0.).nonzero()[0]
            row0 = row[idx0]

            # convert row0 and col0 to a first attempt of "atomization"
            atoms = []
            for r in range(N):
                idx0r = (row0 == r).nonzero()[0]
                row0r = row0[idx0r]
                # although xij == 0, we just do % to ensure unit-cell orbs
                col0r = col[idx0[idx0r]] % N
                if np.all(col0r >= r):
                    # we have a new atom
                    atoms.append(set(col0r))
                else:
                    atoms[-1].update(set(col0r))

            # convert list of orbitals to lists
            def conv(a):
                a = list(a)
                a.sort()
                return a
            atoms = [list(a) for a in atoms]
            if sum(map(len, atoms)) != len(xij):
                raise ValueError(f"{self.__class__.__name__} could not determine correct "
                                 "number of orbitals.")

            atms = Atoms(Atom('H', [-1. for _ in atoms[0]]))
            for orbs in atoms[1:]:
                atms.append(Atom('H', [-1. for _ in orbs]))
            return Geometry(np.zeros([len(atoms), 3]), atms)

        geom_handle = get_geom_handle(xij)

        def convert_to_atom(geom, xij):
            # o2a does not check for correct super-cell index
            n_s = xij.shape[1] // xij.shape[0]
            atm_s = geom.o2a(np.arange(xij.shape[1]))

            # convert csr to dok format
            row = (xij.ncol > 0).nonzero()[0]
            row = np.repeat(row, xij.ncol[row])
            col = xij.col
            arow = atm_s[row]
            acol = atm_s[col]
            del atm_s, row, col
            idx = np.lexsort((acol, arow))
            arow = arow[idx]
            acol = acol[idx]
            xij = xij._D[idx]
            del idx

            # Now figure out if xij is consistent
            duplicates = np.logical_and(np.diff(acol) == 0,
                                        np.diff(arow) == 0).nonzero()[0]
            if duplicates.size > 0:
                if not np.allclose(xij[duplicates+1] - xij[duplicates], 0.):
                    raise ValueError(f"{self.__class__.__name__} converting xij(orb) -> xij(atom) went wrong. "
                                     "This may happen if your coordinates are not inside the unitcell, please pass "
                                     "a usable geometry.")

            # remove duplicates to create new matrix
            arow = np.delete(arow, duplicates)
            acol = np.delete(acol, duplicates)
            xij = np.delete(xij, duplicates, axis=0)

            # Create a new sparse matrix
            # Create the new index pointer
            indptr = np.insert(np.array([0, len(xij)], np.int32), 1,
                               (np.diff(arow) != 0).nonzero()[0] + 1)
            assert len(indptr) == geom.na + 1
            return SparseCSR((xij, acol, indptr), shape=(geom.na, geom.na * n_s))

        def coord_from_xij(xij):
            # first atom is at 0, 0, 0
            na = len(xij)
            xyz = _a.zerosd([na, 3])
            xyz[0] = [0, 0, 0]
            mark = _a.zerosi(na)
            mark[0] = 1
            run_atoms = deque([0])
            while len(run_atoms) > 0:
                atm = run_atoms.popleft()
                xyz_atm = xyz[atm].reshape(1, 3)
                neighbours = xij.edges(atm, exclude=atm)
                neighbours = neighbours[neighbours < na]

                # update those that haven't been calculated
                idx = mark[neighbours] == 0
                neigh_idx = neighbours[idx]
                if len(neigh_idx) == 0:
                    continue
                xyz[neigh_idx, :] = xij[atm, neigh_idx] - xyz_atm
                mark[neigh_idx] = 1
                # add more atoms to be processed, since we have *mark*
                # we will only run every atom once
                run_atoms.extend(neigh_idx.tolist())

                # check that everything is correct
                if (~idx).sum() > 0:
                    neg_neighbours = neighbours[~idx]
                    if not np.allclose(xyz[neg_neighbours, :],
                                       xij[atm, neg_neighbours] - xyz_atm):
                        raise ValueError(f"{self.__class__.__name__} xij(orb) -> xyz did not  "
                                         f"find same coordinates for different connections")

            if mark.sum() != na:
                raise ValueError(f"{self.__class__.__name__} xij(orb) -> Geometry does not  "
                                 f"have a fully connected geometry. It is impossible to create relative coordinates")
            return xyz

        def sc_from_xij(xij, xyz):
            na = xij.shape[0]
            n_s = xij.shape[1] // xij.shape[0]
            if n_s == 1:
                # easy!!
                return SuperCell(xyz.max(axis=0) - xyz.min(axis=0) + 10., nsc=[1] * 3)

            sc_off = _a.zerosd([n_s, 3])
            mark = _a.zerosi(n_s)
            mark[0] = 1
            for atm in range(na):
                neighbours = xij.edges(atm, exclude=atm)
                uneighbours = neighbours % na
                neighbour_isc = neighbours // na

                # get offset in terms of unit-cell
                off = xij[atm, neighbours] - (xyz[uneighbours] - xyz[atm].reshape(1, 3))

                idx = mark[neighbour_isc] == 0
                if not np.allclose(off[~idx], sc_off[neighbour_isc[~idx]]):
                    raise ValueError(f"{self.__class__.__name__} xij(orb) -> xyz did not  "
                                     f"find same supercell offsets for different connections")

                if idx.sum() == 0:
                    continue

                for idx in idx.nonzero()[0]:
                    nidx = neighbour_isc[idx]
                    if mark[nidx] == 0:
                        mark[nidx] = 1
                        sc_off[nidx] = off[idx]
                    elif not np.allclose(sc_off[nidx], off[idx]):
                        raise ValueError(f"{self.__class__.__name__} xij(orb) -> xyz did not  "
                                         f"find same supercell offsets for different connections")
            # We know that siesta returns isc
            # for iz in [0, 1, 2, 3, -3, -2, -1]:
            #  for iy in [0, 1, 2, -2, -1]:
            #   for ix in [0, 1, -1]:
            # every block we find a half monotonically increasing vector additions
            # Note the first is always [0, 0, 0]
            # So our best chance is to *guess* the first nsc
            # then reshape, then guess, then reshape, then guess :)
            sc_diff = np.diff(sc_off, axis=0)

            def get_nsc(sc_off):
                """ Determine nsc depending on the axis """
                # correct the offsets
                ndim = sc_off.ndim

                if sc_off.shape[0] == 1:
                    return 1

                # always select the 2nd one since that contains the offset
                # for the first isc [1, 0, 0] or [0, 1, 0] or [0, 0, 1]
                sc_dir = sc_off[(1, ) + np.index_exp[0] * (ndim - 2)].reshape(1, 3)
                norm2_sc_dir = (sc_dir ** 2).sum()
                # figure out the maximum integer part
                # we select 0 indices for all already determined lattice
                # vectors since we know the first one is [0, 0, 0]
                idx = np.index_exp[:] + np.index_exp[0] * (ndim - 2)
                projection = (sc_off[idx] * sc_dir).sum(-1) / norm2_sc_dir
                iprojection = np.rint(projection)
                # reduce, find 0
                idx_zero = np.isclose(iprojection, 0, atol=1e-5).nonzero()[0]
                if idx_zero.size <= 1:
                    return 1

                # only take those values that are continuous
                # we *must* have some supercell connections
                idx_max = idx_zero[1]

                # find where they are close
                # since there may be *many* zeros (non-coupling elements)
                # we first have to cut off anything that is not integer
                if np.allclose(projection[:idx_max], iprojection[:idx_max], atol=1e-5):
                    return idx_max
                raise ValueError(f"Could not determine nsc from coordinates")

            nsc = _a.onesi(3)
            nsc[0] = get_nsc(sc_off)
            sc_off = sc_off.reshape(-1, nsc[0], 3)
            nsc[1] = get_nsc(sc_off)
            sc_off = sc_off.reshape(-1, nsc[1], nsc[0], 3)
            nsc[2] = sc_off.shape[0]

            # now determine cell parameters
            if all(nsc > 1):
                cell = _a.arrayd([sc_off[0, 0, 1],
                                  sc_off[0, 1, 0],
                                  sc_off[1, 0, 0]])
            else:
                # we will never have all(nsc == 1) since that is
                # taken care of at the start

                # this gets a bit tricky, since we don't know one of the
                # lattice vectors
                cell = _a.zerosd([3, 3])
                i = 0
                for idx, isc in enumerate(nsc):
                    if isc > 1:
                        sl = [0, 0, 0]
                        sl[2 - idx] = 1
                        cell[i] = sc_off[tuple(sl)]
                        i += 1

                # figure out the last vectors
                # We'll just use Cartesian coordinates
                while i < 3:
                    # this means we don't have any supercell connections
                    # along at least 1 other lattice vector.
                    lcell = np.fabs(cell).sum(0)

                    # figure out which Cartesian direction we are *missing*
                    cart_dir = np.argmin(lcell)
                    cell[i, cart_dir] = xyz[:, cart_dir].max() - xyz[:, cart_dir].min() + 10.
                    i += 1

            return SuperCell(cell, nsc)

        # now we have all orbitals, ensure compatibility with passed geometry
        if geometry is None:
            atm_xij = convert_to_atom(geom_handle, xij)
            xyz = coord_from_xij(atm_xij)
            sc = sc_from_xij(atm_xij, xyz)
            geometry = Geometry(xyz, geom_handle.atoms, sc)

            # Move coordinates into unit-cell
            geometry.xyz[:, :] = (geometry.fxyz % 1.) @ geometry.cell

        else:
            if geometry.n_s != xij.shape[1] // xij.shape[0]:
                atm_xij = convert_to_atom(geom_handle, xij)
                sc = sc_from_xij(atm_xij, coord_from_xij(atm_xij))
                geometry.set_nsc(sc.nsc)

            def conv(orbs, atm):
                if len(orbs) == len(atm):
                    return atm
                return atm.copy(orbitals=[-1. for _ in orbs])
            atms = Atoms(list(map(conv, geom_handle.atoms, geometry.atoms)))
            if len(atms) != len(geometry):
                raise ValueError(f"{self.__class__.__name__} passed geometry for reading "
                                 "sparse matrix does not contain same number of atoms!")
            geometry = geometry.copy()
            # TODO check that geometry and xyz are the same!
            geometry._atoms = atms

        return geometry

    def _read_atoms(self, **kwargs):
        """ Reads basis set and geometry information from the HSX file """
        # Now read the sizes used...
        no, na, nspecies = _siesta.read_hsx_specie_sizes(self.file)
        _bin_check(self, 'read_geometry', 'could not read specie sizes.')
        # Read specie information
        labels, val_q, norbs, isa = _siesta.read_hsx_species(self.file, nspecies, no, na)
        # convert to proper string
        labels = labels.T.reshape(nspecies, -1)
        labels = labels.view(f"S{labels.shape[1]}")
        labels = list(map(lambda s: b''.join(s).decode('utf-8').strip(),
                          labels.tolist())
        )
        _bin_check(self, 'read_geometry', 'could not read species.')
        # to python index
        isa -= 1

        from sisl.atom import _ptbl

        # try and convert labels into symbols
        # We do this by:
        # 1. label -> symbol
        # 2. label[:2] -> symbol
        # 3. label[:1] -> symbol
        symbols = []
        lbls = []
        for label in labels:
            lbls.append(label)
            try:
                symbol = _ptbl.Z_label(label)
                symbols.append(symbol)
                continue
            except:
                pass
            try:
                symbol = _ptbl.Z_label(label[:2])
                symbols.append(symbol)
                continue
            except:
                pass
            try:
                symbol = _ptbl.Z_label(label[:1])
                symbols.append(symbol)
                continue
            except:
                # we have no clue, assign -1
                symbols.append(-1)

        # Read in orbital information
        atoms = []
        for ispecie in range(nspecies):
            n_l_zeta = _siesta.read_hsx_specie(self.file, ispecie+1, norbs[ispecie])
            _bin_check(self, 'read_geometry', f'could not read specie {ispecie}.')
            # create orbital
            # no shell will have l>5, so m=10 should be more than enough
            m = 10
            orbs = []
            for n, l, zeta in zip(*n_l_zeta):
                # manual loop on m quantum numbers
                if m > l:
                    m = -l
                orbs.append(AtomicOrbital(n=n, l=l, m=m, zeta=zeta, R=-1.))
                m += 1

            # now create atom
            atoms.append(Atom(symbols[ispecie], orbs, tag=lbls[ispecie]))

        # now read in xij to retrieve atomic positions
        Gamma, spin, no, no_s, nnz = _siesta.read_hsx_sizes(self.file)
        _bin_check(self, 'read_geometry', 'could not read matrix sizes.')
        ncol, col, _, dxij = _siesta.read_hsx_sx(self.file, Gamma, spin, no, no_s, nnz)
        dxij = dxij.T * _Bohr2Ang
        col -= 1
        _bin_check(self, 'read_geometry', 'could not read xij matrix.')

        # now create atoms object
        atoms = Atoms([atoms[ia] for ia in isa])

        return atoms

    def read_hamiltonian(self, **kwargs):
        """ Returns the electronic structure from the siesta.TSHS file """

        # Now read the sizes used...
        Gamma, spin, no, no_s, nnz = _siesta.read_hsx_sizes(self.file)
        _bin_check(self, 'read_hamiltonian', 'could not read Hamiltonian sizes.')
        ncol, col, dH, dS, dxij = _siesta.read_hsx_hsx(self.file, Gamma, spin, no, no_s, nnz)
        dxij = dxij.T * _Bohr2Ang
        col -= 1
        _bin_check(self, 'read_hamiltonian', 'could not read Hamiltonian.')

        ptr = _ncol_to_indptr(ncol)
        xij = SparseCSR((dxij, col, ptr), shape=(no, no_s))
        geom = self._xij2system(xij, kwargs.get('geometry', kwargs.get('geom', None)))

        if geom.no != no or geom.no_s != no_s:
            raise SileError(f"{str(self)}.read_hamiltonian could not use the "
                            "passed geometry as the number of atoms or orbitals is "
                            "inconsistent with HSX file.")

        # Create the Hamiltonian container
        H = Hamiltonian(geom, spin, nnzpr=1, dtype=np.float32, orthogonal=False)

        # Create the new sparse matrix
        H._csr.ncol = ncol.astype(np.int32, copy=False)
        H._csr.ptr = ptr
        # Correct fortran indices
        H._csr.col = col.astype(np.int32, copy=False)
        H._csr._nnz = len(col)

        H._csr._D = _a.emptyf([nnz, spin+1])
        H._csr._D[:, :spin] = dH[:, :] * _Ry2eV
        H._csr._D[:, spin] = dS[:]

        _mat_spin_convert(H)

        # Convert the supercells to sisl supercells
        if no_s // no == np.product(geom.nsc):
            _csr_from_siesta(geom, H._csr)

        return H.transpose(spin=False, sort=kwargs.get("sort", True))

    def read_overlap(self, **kwargs):
        """ Returns the overlap matrix from the siesta.HSX file """
        # Now read the sizes used...
        Gamma, spin, no, no_s, nnz = _siesta.read_hsx_sizes(self.file)
        _bin_check(self, 'read_overlap', 'could not read overlap matrix sizes.')
        ncol, col, dS, dxij = _siesta.read_hsx_sx(self.file, Gamma, spin, no, no_s, nnz)
        dxij = dxij.T * _Bohr2Ang
        col -= 1
        _bin_check(self, 'read_overlap', 'could not read overlap matrix.')

        ptr = _ncol_to_indptr(ncol)
        xij = SparseCSR((dxij, col, ptr), shape=(no, no_s))
        geom = self._xij2system(xij, kwargs.get('geometry', kwargs.get('geom', None)))

        if geom.no != no or geom.no_s != no_s:
            raise SileError(f"{str(self)}.read_overlap could not use the "
                            "passed geometry as the number of atoms or orbitals is "
                            "inconsistent with HSX file.")

        # Create the Hamiltonian container
        S = Overlap(geom, nnzpr=1)

        # Create the new sparse matrix
        S._csr.ncol = ncol.astype(np.int32, copy=False)
        S._csr.ptr = _ncol_to_indptr(ncol)
        # Correct fortran indices
        S._csr.col = col.astype(np.int32, copy=False)
        S._csr._nnz = len(col)

        S._csr._D = _a.emptyf([nnz, 1])
        S._csr._D[:, 0] = dS[:]

        # Convert the supercells to sisl supercells
        if no_s // no == np.product(geom.nsc):
            _csr_from_siesta(geom, S._csr)

        # not really necessary with Hermitian transposing, but for consistency
        return S.transpose(sort=kwargs.get("sort", True))


@set_module("sisl.io.siesta")
class wfsxSileSiesta(SileBinSiesta):
    r""" Binary WFSX file reader for Siesta """

    def yield_eigenstate(self, parent=None):
        r""" Reads eigenstates from the WFSX file

        Returns
        -------
        state: EigenstateElectron
        """
        # First query information
        nspin, nou, nk, Gamma = _siesta.read_wfsx_sizes(self.file)
        _bin_check(self, 'yield_eigenstate', 'could not read sizes.')

        if nspin in [4, 8]:
            nspin = 1 # only 1 spin
            func = _siesta.read_wfsx_index_4
        elif Gamma:
            func = _siesta.read_wfsx_index_1
        else:
            func = _siesta.read_wfsx_index_2

        if parent is None:
            def convert_k(k):
                if not np.allclose(k, 0.):
                    warn(f"{self.__class__.__name__}.yield_eigenstate returns a k-point in 1/Ang (not in reduced format), please pass 'parent' to ensure reduced k")
                return k
        else:
            # We can succesfully convert to proper reduced k-points
            if isinstance(parent, SuperCell):
                def convert_k(k):
                    return np.dot(k, parent.cell.T) / (2 * pi)
            else:
                def convert_k(k):
                    return np.dot(k, parent.sc.cell.T) / (2 * pi)

        for ispin, ik in product(range(1, nspin + 1), range(1, nk + 1)):
            k, _, nwf = _siesta.read_wfsx_index_info(self.file, ispin, ik)
            # Convert to 1/Ang
            k /= _Bohr2Ang
            _bin_check(self, 'yield_eigenstate', f"could not read index info [{ispin}, {ik}]")

            idx, eig, state = func(self.file, ispin, ik, nou, nwf)
            _bin_check(self, 'yield_eigenstate', f"could not read state information [{ispin}, {ik}, {nwf}]")

            # eig is already in eV
            # we probably need to add spin
            # see onlysSileSiesta.read_supercell for .T
            es = EigenstateElectron(state.T, eig, parent=parent,
                                    k=convert_k(k), gauge="r", index=idx - 1)
            yield es


@set_module("sisl.io.siesta")
class _gridSileSiesta(SileBinSiesta):
    r""" Binary real-space grid file

    The Siesta binary grid sile will automatically convert the units from Siesta
    units (Bohr, Ry) to sisl units (Ang, eV) provided the correct extension is present.
    """

    def read_supercell(self, *args, **kwargs):
        r""" Return the cell contained in the file """

        cell = _siesta.read_grid_cell(self.file).T * _Bohr2Ang
        _bin_check(self, 'read_supercell', 'could not read cell.')

        return SuperCell(cell)

    def read_grid_size(self):
        r""" Query grid size information such as the grid size and number of spin components

        Returns
        -------
        int : number of spin-components
        mesh : 3 values for the number of mesh-elements
        """

        # Read the sizes
        nspin, mesh = _siesta.read_grid_sizes(self.file)
        _bin_check(self, 'read_grid_size', 'could not read grid sizes.')
        return nspin, mesh

    def read_grid(self, index=0, dtype=np.float64, *args, **kwargs):
        """ Read grid contained in the Grid file

        Parameters
        ----------
        index : int or array_like, optional
           the spin-index for retrieving one of the components. If a vector
           is passed it refers to the fraction per indexed component. I.e.
           ``[0.5, 0.5]`` will return sum of half the first two components.
           Default to the first component.
        dtype : numpy.float64, optional
           default data-type precision
        """
        index = kwargs.get('spin', index)
        # Read the sizes and cell
        nspin, mesh = self.read_grid_size()
        sc = self.read_supercell()
        grid = _siesta.read_grid(self.file, nspin, mesh[0], mesh[1], mesh[2])
        _bin_check(self, 'read_grid', 'could not read grid.')

        if isinstance(index, Integral):
            grid = grid[:, :, :, index]
        else:
            if len(index) > grid.shape[0]:
                raise ValueError(self.__class__.__name__ + '.read_grid requires spin to be an integer or '
                                 'an array of length equal to the number of spin components.')
            # It is F-contiguous, hence the last index
            g = grid[:, :, :, 0] * index[0]
            for i, scale in enumerate(index[1:]):
                g += grid[:, :, :, 1+i] * scale
            grid = g

        # Simply create the grid (with no information)
        # We will overwrite the actual grid
        g = Grid([1, 1, 1], sc=sc)
        # NOTE: there is no need to swap-axes since the returned array is in F ordering
        #       and thus the first axis is the fast (x, y, z) is retained
        g.grid = grid * self.grid_unit
        return g


@set_module("sisl.io.siesta")
class _gfSileSiesta(SileBinSiesta):
    """ Surface Green function file containing, Hamiltonian, overlap matrix and self-energies

    Do not mix read and write statements when using this code. Complete one or the other
    before doing the other thing. Fortran does not allow the same file opened twice, if this
    is needed you are recommended to make a symlink to the file and thus open two different
    files.

    This small snippet reads/writes the GF file

    >>> with sisl.io._gfSileSiesta('hello.GF') as f:
    ...    nspin, no, k, E = f.read_header()
    ...    for ispin, new_k, k, E in f:
    ...        if new_k:
    ...            H, S = f.read_hamiltonian()
    ...        SeHSE = f.read_self_energy()

    To write a file do:

    >>> with sisl.io._gfSileSiesta('hello.GF') as f:
    ...    f.write_header(sisl.MonkhorstPack(...), E)
    ...    for ispin, new_k, k, E in f:
    ...        if new_k:
    ...            f.write_hamiltonian(H, S)
    ...        f.write_self_energy(SeHSE)
    """

    def _setup(self, *args, **kwargs):
        """ Simple setup that needs to be overwritten """
        self._iu = -1
        # The unit convention used for energy-points
        # This is necessary until Siesta uses CODATA values
        if kwargs.get("unit", "old").lower() in ("old", "4.1"):
            self._E_Ry2eV = 13.60580
        else:
            self._E_Ry2eV = _Ry2eV

    def _is_open(self):
        return self._iu != -1

    def _open_gf(self, mode, rewind=False):
        if self._is_open() and mode == self._mode:
            if rewind:
                _siesta.io_m.rewind_file(self._iu)
            else:
                # retain indices
                return
        else:
            if mode == 'r':
                self._iu = _siesta.io_m.open_file_read(self.file)
            elif mode == 'w':
                self._iu = _siesta.io_m.open_file_write(self.file)
        _bin_check(self, '_open_gf', 'could not open for {}.'.format({'r': 'reading', 'w': 'writing'}[mode]))

        # They will at any given time
        # correspond to the current Python indices that is to be read
        # The process for identification is done on this basis:
        #  iE is the current (Python) index for the energy-point to be read
        #  ik is the current (Python) index for the k-point to be read
        #  ispin is the current (Python) index for the spin-index to be read (only has meaning for a spin-polarized
        #         GF files)
        #  state is:
        #        -1 : the file-descriptor has just been opened (i.e. in front of header)
        #         0 : it means that the file-descriptor IS in front of H and S
        #         1 : it means that the file-descriptor is NOT in front of H and S but somewhere in front of a self-energy
        #  is_read is:
        #         0 : means that the current indices HAVE NOT been read
        #         1 : means that the current indices HAVE been read
        #
        # All routines in the gf_read/write sources requires input in Python indices
        self._state = -1
        self._is_read = 0
        self._ispin = 0
        self._ik = 0
        self._iE = 0

    def _close_gf(self):
        if not self._is_open():
            return
        # Close it
        _siesta.io_m.close_file(self._iu)
        self._iu = -1

        # Clean variables
        del self._state
        del self._iE
        del self._ik
        del self._ispin
        try:
            del self._no_u
        except:
            pass
        try:
            del self._nspin
        except:
            pass

    def _step_counter(self, method, **kwargs):
        """ Method for stepping values *must* be called before doing the actual read to check correct values """
        opt = {'method': method}
        if kwargs.get('header', False):
            # The header only exists once, so check whether it is the correct place to read/write
            if self._state != -1 or self._is_read == 1:
                raise SileError(self.__class__.__name__ + '.{method} failed because the header has already '
                                'been read.'.format(**opt))
            self._state = -1
            self._ispin = 0
            self._ik = 0
            self._iE = 0
            #print('HEADER: ', self._state, self._ispin, self._ik, self._iE)

        elif kwargs.get('HS', False):
            # Correct for the previous state and jump values
            if self._state == -1:
                # We have just read the header
                if self._is_read != 1:
                    raise SileError(self.__class__.__name__ + '.{method} failed because the file descriptor '
                                    'has not read the header.'.format(**opt))
                # Reset values as though the header has just been read
                self._state = 0
                self._ispin = 0
                self._ik = 0
                self._iE = 0

            elif self._state == 0:
                if self._is_read == 1:
                    raise SileError(self.__class__.__name__ + '.{method} failed because the file descriptor '
                                    'has already read the current HS for the given k-point.'.format(**opt))
            elif self._state == 1:
                # We have just read from the last energy-point
                if self._iE + 1 != self._nE or self._is_read != 1:
                    raise SileError(self.__class__.__name__ + '.{method} failed because the file descriptor '
                                    'has not read all energy-points for a given k-point.'.format(**opt))
                self._state = 0
                self._ik += 1
                if self._ik >= self._nk:
                    # We need to step spin
                    self._ispin += 1
                    self._ik = 0
                self._iE = 0

            #print('HS: ', self._state, self._ispin, self._ik, self._iE)

            if self._ispin >= self._nspin:
                opt['spin'] = self._ispin + 1
                opt['nspin'] = self._nspin
                raise SileError(self.__class__.__name__ + '.{method} failed because of missing information, '
                                'a non-existing entry has been requested! spin={spin} max_spin={nspin}.'.format(**opt))

        else:
            # We are reading an energy-point
            if self._state == -1:
                raise SileError(self.__class__.__name__ + '.{method} failed because the file descriptor '
                                'has an unknown state.'.format(**opt))

            elif self._state == 0:
                if self._is_read == 1:
                    # Fine, we have just read the HS, ispin and ik are correct
                    self._state = 1
                    self._iE = 0
                else:
                    raise SileError(self.__class__.__name__ + '.{method} failed because the file descriptor '
                                    'has an unknown state.'.format(**opt))

            elif self._state == 1:
                if self._is_read == 0 and self._iE < self._nE:
                    # we haven't read the current energy-point.and self._iE + 1 < self._nE:
                    pass
                elif self._is_read == 1 and self._iE + 1 < self._nE:
                    self._iE += 1
                else:
                    raise SileError(self.__class__.__name__ + '.{method} failed because the file descriptor '
                                    'has an unknown state.'.format(**opt))

                if self._iE >= self._nE:
                    # You are trying to read beyond the entry
                    opt['iE'] = self._iE + 1
                    opt['NE'] = self._nE
                    raise SileError(self.__class__.__name__ + '.{method} failed because of missing information, '
                                    'a non-existing energy-point has been requested! E_index={iE} max_E_index={NE}.'.format(**opt))
            #print('SE: ', self._state, self._ispin, self._ik, self._iE)

        # Always signal (when stepping) that we have not yet read the thing
        if kwargs.get('read', False):
            self._is_read = 1
        else:
            self._is_read = 0

    def Eindex(self, E):
        """ Return the closest energy index corresponding to the energy ``E``

        Parameters
        ----------
        E : float or int
           if ``int``, return it-self, else return the energy index which is
           closests to the energy.
        """
        if isinstance(E, Integral):
            return E
        idxE = np.abs(self._E - E).argmin()
        ret_E = self._E[idxE]
        if abs(ret_E - E) > 5e-3:
            warn(self.__class__.__name__ + " requesting energy " +
                 f"{E:.5f} eV, found {ret_E:.5f} eV as the closest energy!")
        elif abs(ret_E - E) > 1e-3:
            info(self.__class__.__name__ + " requesting energy " +
                 f"{E:.5f} eV, found {ret_E:.5f} eV as the closest energy!")
        return idxE

    def kindex(self, k):
        """ Return the index of the k-point that is closests to the queried k-point (in reduced coordinates)

        Parameters
        ----------
        k : array_like of float or int
           the queried k-point in reduced coordinates :math:`]-0.5;0.5]`. If ``int``
           return it-self.
        """
        if isinstance(k, Integral):
            return k
        ik = np.sum(np.abs(self._k - _a.asarrayd(k)[None, :]), axis=1).argmin()
        ret_k = self._k[ik, :]
        if not np.allclose(ret_k, k, atol=0.0001):
            warn(SileWarning(self.__class__.__name__ + " requesting k-point " +
                             "[{:.3f}, {:.3f}, {:.3f}]".format(*k) +
                             " found " +
                             "[{:.3f}, {:.3f}, {:.3f}]".format(*ret_k)))
        return ik

    def read_header(self):
        """ Read the header of the file and open it for reading subsequently

        NOTES: this method may change in the future

        Returns
        -------
        nspin : number of spin-components stored (1 or 2)
        no_u : size of the matrices returned
        k : k points in the GF file
        E : energy points in the GF file
        """
        # Ensure it is open (in read-mode)
        if self._is_open():
            _siesta.io_m.rewind_file(self._iu)
        else:
            self._open_gf('r')
        nspin, no_u, nkpt, NE = _siesta.read_gf_sizes(self._iu)
        _bin_check(self, 'read_header', 'could not read sizes.')
        self._nspin = nspin
        self._nk = nkpt
        self._nE = NE

        # We need to rewind (because of k and energy -points)
        _siesta.io_m.rewind_file(self._iu)
        self._step_counter('read_header', header=True, read=True)
        k, E = _siesta.read_gf_header(self._iu, nkpt, NE)
        _bin_check(self, 'read_header', 'could not read header information.')

        if self._nspin > 2: # non-colinear
            self._no_u = no_u * 2
        else:
            self._no_u = no_u
        self._E = E * self._E_Ry2eV
        self._k = k.T

        return nspin, no_u, self._k, self._E

    def disk_usage(self):
        """ Calculate the estimated size of the resulting file

        Returns
        -------
        estimated disk-space used in GB
        """
        is_open = self._is_open()
        if not is_open:
            self.read_header()

        # HS are only stored per k-point
        HS = 2 * self._nspin * self._nk
        SE = HS / 2 * self._nE

        # Now calculate the full size
        # no_u ** 2 = matrix size
        # 16 = bytes in double complex
        # 1024 ** 3 = B -> GB
        mem = (HS + SE) * self._no_u ** 2 * 16 / 1024 ** 3

        if not is_open:
            self._close_gf()

        return mem

    def read_hamiltonian(self):
        """ Return current Hamiltonian and overlap matrix from the GF file

        Returns
        -------
        complex128 : Hamiltonian matrix
        complex128 : Overlap matrix
        """
        self._step_counter('read_hamiltonian', HS=True, read=True)
        H, S = _siesta.read_gf_hs(self._iu, self._no_u)
        _bin_check(self, 'read_hamiltonian', 'could not read Hamiltonian and overlap matrices.')
        # we don't convert to C order!
        return H * _Ry2eV, S

    def read_self_energy(self):
        r""" Read the currently reached bulk self-energy

        The returned self-energy is:

        .. math::
            \boldsymbol \Sigma_{\mathrm{bulk}}(E) = \mathbf S E - \mathbf H - \boldsymbol \Sigma(E)

        Returns
        -------
        complex128 : Self-energy matrix
        """
        self._step_counter('read_self_energy', read=True)
        SE = _siesta.read_gf_se(self._iu, self._no_u, self._iE)
        _bin_check(self, 'read_self_energy', 'could not read self-energy.')
        # we don't convert to C order!
        return SE * _Ry2eV

    def HkSk(self, k=(0, 0, 0), spin=0):
        """ Retrieve H and S for the given k-point

        Parameters
        ----------
        k : int or array_like of float, optional
           k-point to read the corresponding Hamiltonian and overlap matrices
           for. If a specific k-point is passed `kindex` will be used to find
           the corresponding index.
        spin : int, optional
           spin-index for the Hamiltonian and overlap matrices
        """
        if not self._is_open():
            self.read_header()

        # find k-index that is requested
        ik = self.kindex(k)
        _siesta.read_gf_find(self._iu, self._nspin, self._nk, self._nE,
                             self._state, self._ispin, self._ik, self._iE, self._is_read,
                             0, spin, ik, 0)
        _bin_check(self, 'HkSk', 'could not find Hamiltonian and overlap matrix.')

        self._state = 0
        self._ispin = spin
        self._ik = ik
        self._iE = 0
        self._is_read = 0 # signal this is to be read
        return self.read_hamiltonian()

    def self_energy(self, E, k=0, spin=0):
        """ Retrieve self-energy for a given energy-point and k-point

        Parameters
        ----------
        E : int or float
           energy to retrieve self-energy at
        k : int or array_like of float, optional
           k-point to retrieve k-point at
        spin : int, optional
           spin-index to retrieve self-energy at
        """
        if not self._is_open():
            self.read_header()

        ik = self.kindex(k)
        iE = self.Eindex(E)
        _siesta.read_gf_find(self._iu, self._nspin, self._nk, self._nE,
                             self._state, self._ispin, self._ik, self._iE, self._is_read,
                             1, spin, ik, iE)
        _bin_check(self, 'self_energy', 'could not find requested self-energy.')

        self._state = 1
        self._ispin = spin
        self._ik = ik
        self._iE = iE
        self._is_read = 0 # signal this is to be read
        return self.read_self_energy()

    def write_header(self, bz, E, mu=0., obj=None):
        """ Write to the binary file the header of the file

        Parameters
        ----------
        bz : BrillouinZone
           contains the k-points, the weights and possibly the parent Hamiltonian (if `obj` is None)s
        E : array_like of cmplx or float
           the energy points. If `obj` is an instance of `SelfEnergy` where an
           associated ``eta`` is defined then `E` may be float, otherwise
           it *has* to be a complex array.
        mu : float, optional
           chemical potential in the file
        obj : ..., optional
           an object that contains the Hamiltonian definitions, defaults to ``bz.parent``
        """
        if obj is None:
            obj = bz.parent
        nspin = len(obj.spin)
        cell = obj.geometry.sc.cell
        na_u = obj.geometry.na
        no_u = obj.geometry.no
        xa = obj.geometry.xyz
        # The lasto in siesta requires lasto(0) == 0
        # and secondly, the Python index to fortran
        # index makes firsto behave like fortran lasto
        lasto = obj.geometry.firsto
        bloch = _a.onesi(3)
        mu = mu
        NE = len(E)
        if E.dtype not in [np.complex64, np.complex128]:
            E = E + 1j * obj.eta
        Nk = len(bz)
        k = bz.k
        w = bz.weight

        sizes = {
            'na_used': na_u,
            'nkpt': Nk,
            'ne': NE,
        }

        self._nspin = nspin
        self._E = E
        self._k = np.copy(k)
        self._nE = len(E)
        self._nk = len(k)
        if self._nspin > 2:
            self._no_u = no_u * 2
        else:
            self._no_u = no_u

        # Ensure it is open (in write mode)
        self._close_gf()
        self._open_gf('w')

        # Now write to it...
        self._step_counter('write_header', header=True, read=True)
        # see onlysSileSiesta.read_supercell for .T
        _siesta.write_gf_header(self._iu, nspin, _toF(cell.T, np.float64, 1. / _Bohr2Ang),
                                na_u, no_u, no_u, _toF(xa.T, np.float64, 1. / _Bohr2Ang),
                                lasto, bloch, 0, mu * _eV2Ry, _toF(k.T, np.float64),
                                w, self._E / self._E_Ry2eV,
                                **sizes)
        _bin_check(self, 'write_header', 'could not write header information.')

    def write_hamiltonian(self, H, S=None):
        """ Write the current energy, k-point and H and S to the file

        Parameters
        ----------
        H : matrix
           a square matrix corresponding to the Hamiltonian
        S : matrix, optional
           a square matrix corresponding to the overlap, for efficiency reasons
           it may be advantageous to specify this argument for orthogonal cells.
        """
        no = len(H)
        if S is None:
            S = np.eye(no, dtype=np.complex128, order='F')
        self._step_counter('write_hamiltonian', HS=True, read=True)
        _siesta.write_gf_hs(self._iu, self._ik, self._E[self._iE] / self._E_Ry2eV,
                            _toF(H, np.complex128, _eV2Ry),
                            _toF(S, np.complex128), no_u=no)
        _bin_check(self, 'write_hamiltonian', 'could not write Hamiltonian and overlap matrices.')

    def write_self_energy(self, SE):
        r""" Write the current self energy, k-point and H and S to the file

        The self-energy must correspond to the *bulk* self-energy

        .. math::
            \boldsymbol \Sigma_{\mathrm{bulk}}(E) = \mathbf S E - \mathbf H - \boldsymbol \Sigma(E)

        Parameters
        ----------
        SE : matrix
           a square matrix corresponding to the self-energy (Green function)
        """
        no = len(SE)
        self._step_counter('write_self_energy', read=True)
        _siesta.write_gf_se(self._iu, self._ik, self._iE, self._E[self._iE] / self._E_Ry2eV,
                            _toF(SE, np.complex128, _eV2Ry), no_u=no)
        _bin_check(self, 'write_self_energy', 'could not write self-energy.')

    def __len__(self):
        return self._nE * self._nk * self._nspin

    def __iter__(self):
        """ Iterate through the energies and k-points that this GF file is associated with

        Yields
        ------
        bool, list of float, float
        """
        # get everything
        e = self._E
        if self._nspin in [1, 2]:
            for ispin in range(self._nspin):
                for k in self._k:
                    yield ispin, True, k, e[0]
                    for E in e[1:]:
                        yield ispin, False, k, E

        else:
            for k in self._k:
                yield True, k, e[0]
                for E in e[1:]:
                    yield False, k, E

        # We will automatically close once we hit the end
        self._close_gf()


def _type(name, obj, dic=None):
    if dic is None:
        dic = {}
    # Always pass the docstring
    if not '__doc__' in dic:
        try:
            dic['__doc__'] = obj.__doc__.replace(obj.__name__, name)
        except:
            pass
    return type(name, (obj, ), dic)

# Faster than class ... \ pass
tsgfSileSiesta = _type("tsgfSileSiesta", _gfSileSiesta)
gridSileSiesta = _type("gridSileSiesta", _gridSileSiesta, {'grid_unit': 1.})

if found_module:
    add_sile('TSHS', tshsSileSiesta)
    add_sile('onlyS', onlysSileSiesta)
    add_sile('TSDE', tsdeSileSiesta)
    add_sile('DM', dmSileSiesta)
    add_sile('HSX', hsxSileSiesta)
    add_sile('TSGF', tsgfSileSiesta)
    add_sile('WFSX', wfsxSileSiesta)
    # These have unit-conversions
    add_sile('RHO', _type("rhoSileSiesta", _gridSileSiesta, {'grid_unit': 1./_Bohr2Ang ** 3}))
    add_sile('LDOS', _type("ldosSileSiesta", _gridSileSiesta, {'grid_unit': 1./_Bohr2Ang ** 3}))
    add_sile('RHOINIT', _type("rhoinitSileSiesta", _gridSileSiesta, {'grid_unit': 1./_Bohr2Ang ** 3}))
    add_sile('RHOXC', _type("rhoxcSileSiesta", _gridSileSiesta, {'grid_unit': 1./_Bohr2Ang ** 3}))
    add_sile('DRHO', _type("drhoSileSiesta", _gridSileSiesta, {'grid_unit': 1./_Bohr2Ang ** 3}))
    add_sile('BADER', _type("baderSileSiesta", _gridSileSiesta, {'grid_unit': 1./_Bohr2Ang ** 3}))
    add_sile('IOCH', _type("iorhoSileSiesta", _gridSileSiesta, {'grid_unit': 1./_Bohr2Ang ** 3}))
    add_sile('TOCH', _type("totalrhoSileSiesta", _gridSileSiesta, {'grid_unit': 1./_Bohr2Ang ** 3}))
    # The following two files *require* that
    #  STM.DensityUnits   Ele/bohr**3
    #  which I can't check!
    # They are however the default
    add_sile('STS', _type("stsSileSiesta", _gridSileSiesta, {'grid_unit': 1./_Bohr2Ang ** 3}))
    add_sile('STM.LDOS', _type("stmldosSileSiesta", _gridSileSiesta, {'grid_unit': 1./_Bohr2Ang ** 3}))
    add_sile('VH', _type("hartreeSileSiesta", _gridSileSiesta, {'grid_unit': _Ry2eV}))
    add_sile('VNA', _type("neutralatomhartreeSileSiesta", _gridSileSiesta, {'grid_unit': _Ry2eV}))
    add_sile('VT', _type("totalhartreeSileSiesta", _gridSileSiesta, {'grid_unit': _Ry2eV}))
