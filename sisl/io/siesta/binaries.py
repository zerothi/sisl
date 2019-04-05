from __future__ import print_function, division

from numbers import Integral
import numpy as np

try:
    from . import _siesta
    found_module = True
except Exception as e:
    found_module = False

from sisl.messages import warn, SislError
from ..sile import add_sile, SileError
from .sile import SileBinSiesta

import sisl._array as _a
from sisl import Geometry, Atom, Atoms, SuperCell, Grid
from sisl.unit.siesta import unit_convert
from sisl.physics.sparse import SparseOrbitalBZ
from sisl.physics import Hamiltonian, DensityMatrix, EnergyDensityMatrix
from ._help import *


Ang2Bohr = unit_convert('Ang', 'Bohr')
eV2Ry = unit_convert('eV', 'Ry')
Bohr2Ang = unit_convert('Bohr', 'Ang')
Ry2eV = unit_convert('Ry', 'eV')

__all__ = ['tshsSileSiesta', 'onlysSileSiesta', 'tsdeSileSiesta']
__all__ += ['hsxSileSiesta', 'dmSileSiesta']
__all__ += ['gridSileSiesta']
__all__ += ['tsgfSileSiesta']


def _bin_check(obj, method, message):
    if _siesta.io_m.iostat_query() != 0:
        raise SileError('{}.{} {}'.format(str(obj), method, message))


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
    # Default to use the users geometry
    geom = geom_u

    is_copy = False
    def get_copy(geom, is_copy):
        if is_copy:
            return geom, True
        return geom.copy(), True

    if geom_b.na != geom.na:
        # we have no way of solving this issue...
        raise SileError("{cls}.{method} could not use the passed geometry as the "
                        "of atoms is not consistent, user-atoms={u_na}, file-atoms={b_na}.".format(cls=cls.__name__, method=method,
                                                                                                   b_na=geom_b.na, u_na=geom_u.na))

    # Try and figure out what to do
    if not np.allclose(geom_b.xyz, geom.xyz):
        warn("{cls}.{method} has mismatched atomic coordinates, will copy geometry and use file XYZ.".format(cls=cls.__name__, method=method))
        geom, is_copy = get_copy(geom, is_copy)
        geom.xyz[:, :] = geom_b.xyz[:, :]
    if not np.allclose(geom_b.sc.cell, geom.sc.cell):
        warn("{cls}.{method} has non-equal lattice vectors, will copy geometry and use file lattice.".format(cls=cls.__name__, method=method))
        geom, is_copy = get_copy(geom, is_copy)
        geom.sc.cell[:, :] = geom_b.sc.cell[:, :]
    if not np.array_equal(geom_b.nsc, geom.nsc):
        warn("{cls}.{method} has non-equal number of supercells, will copy geometry and use file supercell count.".format(cls=cls.__name__, method=method))
        geom, is_copy = get_copy(geom, is_copy)
        geom.set_nsc(geom_b.nsc)

    # Now for the difficult part.
    # If there is a mismatch in the number of orbitals we will
    # prefer to use the user-supplied atomic species, but fill with
    # *random* orbitals
    if not np.array_equal(geom_b.atoms.orbitals, geom.atoms.orbitals):
        warn("{cls}.{method} has non-equal number of orbitals per atom, will correct with *empty* orbitals.".format(cls=cls.__name__, method=method))
        geom, is_copy = get_copy(geom, is_copy)

        # Now create a new atom specie with the correct number of orbitals
        norbs = geom_b.atoms.orbitals[:]
        atoms = Atoms([geom.atoms[i].copy(orbital=[-1] * norbs[i]) for i in range(geom.na)])
        geom._atoms = atoms

    return geom


class onlysSileSiesta(SileBinSiesta):
    """ Geometry and overlap matrix """

    def read_supercell(self):
        """ Returns a SuperCell object from a siesta.TSHS file """
        n_s = _siesta.read_tshs_sizes(self.file)[3]
        _bin_check(self, 'read_supercell', 'could not read sizes.')
        arr = _siesta.read_tshs_cell(self.file, n_s)
        _bin_check(self, 'read_supercell', 'could not read cell.')
        nsc = np.array(arr[0], np.int32)
        cell = np.array(arr[1].T, np.float64)
        cell.shape = (3, 3)
        return SuperCell(cell, nsc=nsc)

    def read_geometry(self):
        """ Returns Geometry object from a siesta.TSHS file """

        # Read supercell
        sc = self.read_supercell()

        na = _siesta.read_tshs_sizes(self.file)[1]
        _bin_check(self, 'read_geometry', 'could not read sizes.')
        arr = _siesta.read_tshs_geom(self.file, na)
        _bin_check(self, 'read_geometry', 'could not read geometry.')
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
            atoms.append(Atom(Z+1, [-1] * orb))

        def get_atom(atoms, orbs):
            for atom in atoms:
                if atom.no == orbs:
                    return atom

        atom = []
        for orb in orbs:
            atom.append(get_atom(atoms, orb))

        # Create and return geometry object
        geom = Geometry(xyz, atom, sc=sc)

        return geom

    def read_overlap(self, **kwargs):
        """ Returns the overlap matrix from the siesta.TSHS file """
        tshs_g = self.read_geometry()
        geom = _geometry_align(tshs_g, kwargs.get('geometry', tshs_g), self.__class__, 'read_overlap')

        # read the sizes used...
        sizes = _siesta.read_tshs_sizes(self.file)
        _bin_check(self, 'read_overlap', 'could not read sizes.')
        isc = _siesta.read_tshs_cell(self.file, sizes[3])[2].T
        _bin_check(self, 'read_overlap', 'could not read cell.')
        no = sizes[2]
        nnz = sizes[4]
        ncol, col, dS = _siesta.read_tshs_s(self.file, no, nnz)
        _bin_check(self, 'read_overlap', 'could not read overlap matrix.')

        # Create the Hamiltonian container
        S = SparseOrbitalBZ(geom, nnzpr=1)

        # Create the new sparse matrix
        S._csr.ncol = ncol.astype(np.int32, copy=False)
        S._csr.ptr = np.insert(np.cumsum(ncol, dtype=np.int32), 0, 0)
        # Correct fortran indices
        S._csr.col = col.astype(np.int32, copy=False) - 1
        S._csr._nnz = len(col)

        S._csr._D = np.empty([nnz, 1], np.float64)
        S._csr._D[:, 0] = dS[:]

        # Convert to sisl supercell
        _csr_from_sc_off(S.geometry, isc, S._csr)

        return S


class tshsSileSiesta(onlysSileSiesta):
    """ Geometry, Hamiltonian and overlap matrix file """

    def read_hamiltonian(self, **kwargs):
        """ Returns the electronic structure from the siesta.TSHS file """
        tshs_g = self.read_geometry()
        geom = _geometry_align(tshs_g, kwargs.get('geometry', tshs_g), self.__class__, 'read_hamiltonian')

        # read the sizes used...
        sizes = _siesta.read_tshs_sizes(self.file)
        _bin_check(self, 'read_hamiltonian', 'could not read sizes.')
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
        H._csr.ptr = np.insert(np.cumsum(ncol, dtype=np.int32), 0, 0)
        # Correct fortran indices
        H._csr.col = col.astype(np.int32, copy=False) - 1
        H._csr._nnz = len(col)

        if orthogonal:
            H._csr._D = np.empty([nnz, spin], np.float64)
            H._csr._D[:, :] = dH[:, :]
        else:
            H._csr._D = np.empty([nnz, spin+1], np.float64)
            H._csr._D[:, :spin] = dH[:, :]
            H._csr._D[:, spin] = dS[:]

        # Convert to sisl supercell
        _csr_from_sc_off(H.geometry, isc, H._csr)

        # Find all indices where dS == 1 (remember col is in fortran indices)
        idx = col[np.isclose(dS, 1.).nonzero()[0]]
        if np.any(idx > no):
            print('Number of orbitals: {}'.format(no))
            print(idx)
            raise SileError(str(self) + '.read_hamiltonian could not assert '
                            'the supercell connections in the primary unit-cell.')

        return H

    def write_hamiltonian(self, H, **kwargs):
        """ Writes the Hamiltonian to a siesta.TSHS file """
        H.finalize()
        csr = H._csr.copy()
        if csr.nnz == 0:
            raise SileError(str(self) + '.write_hamiltonian cannot write '
                            'a zero element sparse matrix!')

        # Convert to siesta CSR
        _csr_to_siesta(H.geometry, csr)
        csr.finalize()

        # Extract the data to pass to the fortran routine
        cell = H.geometry.cell * Ang2Bohr
        xyz = H.geometry.xyz * Ang2Bohr

        # Get H and S
        if H.orthogonal:
            h = (csr._D * eV2Ry).astype(np.float64, 'C', copy=False)
            s = csr.diags(1., dim=1)
            # Ensure all data is correctly formatted (i.e. have the same sparsity pattern
            s.align(csr)
            s.finalize()
            if s.nnz != len(h):
                raise SislError('The diagonal elements of your orthogonal Hamiltonian '
                                'have not been defined, this is a requirement.')
            s = (s._D[:, 0]).astype(np.float64, 'C', copy=False)
        else:
            h = (csr._D[:, :H.S_idx] * eV2Ry).astype(np.float64, 'C', copy=False)
            s = (csr._D[:, H.S_idx]).astype(np.float64, 'C', copy=False)
        # Ensure shapes (say if only 1 spin)
        h.shape = (-1, len(H.spin))
        s.shape = (-1,)

        # Get shorter variants
        nsc = H.geometry.nsc[:].astype(np.int32)
        isc = _siesta.siesta_sc_off(*nsc)

        # I can't seem to figure out the usage of f2py
        # Below I get an error if xyz is not transposed and h is transposed,
        # however, they are both in C-contiguous arrays and this is indeed weird... :(
        _siesta.write_tshs_hs(self.file, nsc[0], nsc[1], nsc[2],
                              cell.T, xyz.T, H.geometry.firsto,
                              csr.ncol, csr.col + 1, h, s, isc)
        _bin_check(self, 'write_hamiltonian', 'could not write Hamiltonian and overlap matrix.')


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
        DM._csr.ptr = np.insert(np.cumsum(ncol, dtype=np.int32), 0, 0)
        # Correct fortran indices
        DM._csr.col = col.astype(np.int32, copy=False) - 1
        DM._csr._nnz = len(col)

        DM._csr._D = np.empty([nnz, spin+1], np.float64)
        DM._csr._D[:, :spin] = dDM[:, :]
        # DM file does not contain overlap matrix... so neglect it for now.
        DM._csr._D[:, spin] = 0.

        # Convert the supercells to sisl supercells
        if nsc[0] != 0 or geom.no_s >= col.max():
            _csr_from_siesta(geom, DM._csr)
        else:
            warn(str(self) + '.read_density_matrix may result in a wrong sparse pattern!')

        return DM

    def write_density_matrix(self, DM, **kwargs):
        """ Writes the density matrix to a siesta.DM file """
        DM.finalize()
        csr = DM._csr.copy()
        if csr.nnz == 0:
            raise SileError(str(self) + '.write_density_matrix cannot write '
                            'a zero element sparse matrix!')

        _csr_to_siesta(DM.geometry, csr)
        csr.finalize()

        # Get H and S
        if DM.orthogonal:
            dm = csr._D
        else:
            dm = csr._D[:, :DM.S_idx]

        # Ensure shapes (say if only 1 spin)
        dm.shape = (-1, len(DM.spin))

        nsc = DM.geometry.sc.nsc.astype(np.int32)

        _siesta.write_dm(self.file, nsc, csr.ncol, csr.col + 1, dm)
        _bin_check(self, 'write_density_matrix', 'could not write density matrix.')


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
        EDM._csr.ptr = np.insert(np.cumsum(ncol, dtype=np.int32), 0, 0)
        # Correct fortran indices
        EDM._csr.col = col.astype(np.int32, copy=False) - 1
        EDM._csr._nnz = len(col)

        EDM._csr._D = np.empty([nnz, spin+1], np.float64)
        EDM._csr._D[:, :spin] = dEDM[:, :]
        # EDM file does not contain overlap matrix... so neglect it for now.
        EDM._csr._D[:, spin] = 0.

        # Convert the supercells to sisl supercells
        if nsc[0] != 0 or geom.no_s >= col.max():
            _csr_from_siesta(geom, EDM._csr)
        else:
            warn(str(self) + '.read_energy_density_matrix may result in a wrong sparse pattern!')

        return EDM


class hsxSileSiesta(SileBinSiesta):
    """ Hamiltonian and overlap matrix file """

    def read_hamiltonian(self, **kwargs):
        """ Returns the electronic structure from the siesta.TSHS file """

        # Now read the sizes used...
        Gamma, spin, no, no_s, nnz = _siesta.read_hsx_sizes(self.file)
        _bin_check(self, 'read_hamiltonian', 'could not read Hamiltonian sizes.')
        ncol, col, dH, dS, dxij = _siesta.read_hsx_hsx(self.file, Gamma, spin, no, no_s, nnz)
        _bin_check(self, 'read_hamiltonian', 'could not read Hamiltonian.')

        # Try and immediately attach a geometry
        geom = kwargs.get('geometry', kwargs.get('geom', None))
        if geom is None:
            # We have *no* clue about the
            if np.allclose(dxij, 0.):
                # We truly, have no clue,
                # Just generate a boxed system
                xyz = [[x, 0, 0] for x in range(no)]
                geom = Geometry(xyz, Atom(1), sc=[no, 1, 1])
            else:
                # Try to figure out the supercell
                warn(self.__class__.__name__ + '.read_hamiltonian '
                     '(currently we can not calculate atomic positions from xij array)')
        if geom.no != no:
            raise SileError(str(self) + '.read_hamiltonian could not use the '
                            'passed geometry as the number of atoms or orbitals is '
                            'inconsistent with HSX file.')

        # Create the Hamiltonian container
        H = Hamiltonian(geom, spin, nnzpr=1, dtype=np.float32, orthogonal=False)

        # Create the new sparse matrix
        H._csr.ncol = ncol.astype(np.int32, copy=False)
        H._csr.ptr = np.insert(np.cumsum(ncol, dtype=np.int32), 0, 0)
        # Correct fortran indices
        H._csr.col = col.astype(np.int32, copy=False) - 1
        H._csr._nnz = len(col)

        H._csr._D = np.empty([nnz, spin+1], np.float32)
        H._csr._D[:, :spin] = dH[:, :]
        H._csr._D[:, spin] = dS[:]

        # Convert the supercells to sisl supercells
        if no_s // no == np.product(geom.nsc):
            _csr_from_siesta(geom, H._csr)

        return H

    def read_overlap(self, **kwargs):
        """ Returns the overlap matrix from the siesta.HSX file """
        # Now read the sizes used...
        Gamma, spin, no, no_s, nnz = _siesta.read_hsx_sizes(self.file)
        _bin_check(self, 'read_overlap', 'could not read overlap matrix sizes.')
        ncol, col, dS = _siesta.read_hsx_s(self.file, Gamma, spin, no, no_s, nnz)
        _bin_check(self, 'read_overlap', 'could not read overlap matrix.')

        geom = kwargs.get('geometry', kwargs.get('geom', None))
        if geom is None:
            warn(self.__class__.__name__ + ".read_overlap requires input geometry to assign S")
        if geom.no != no:
            raise SileError(str(self) + '.read_overlap could not use the '
                            'passed geometry as the number of atoms or orbitals is '
                            'inconsistent with HSX file.')

        # Create the Hamiltonian container
        S = SparseOrbitalBZ(geom, nnzpr=1)

        # Create the new sparse matrix
        S._csr.ncol = ncol.astype(np.int32, copy=False)
        S._csr.ptr = np.insert(np.cumsum(ncol, dtype=np.int32), 0, 0)
        # Correct fortran indices
        S._csr.col = col.astype(np.int32, copy=False) - 1
        S._csr._nnz = len(col)

        S._csr._D = np.empty([nnz, 1], np.float32)
        S._csr._D[:, 0] = dS[:]

        # Convert the supercells to sisl supercells
        if no_s // no == np.product(geom.nsc):
            _csr_from_siesta(geom, S._csr)

        return S


class _gridSileSiesta(SileBinSiesta):
    """ Binary real-space grid file

    The Siesta binary grid sile will automatically convert the units from Siesta
    units (Bohr, Ry) to sisl units (Ang, eV) provided the correct extension is present.
    """

    def read_supercell(self, *args, **kwargs):

        cell = _siesta.read_grid_cell(self.file)
        _bin_check(self, 'read_supercell', 'could not read cell.')
        cell = np.array(cell.T, np.float64)
        cell.shape = (3, 3)

        return SuperCell(cell)

    def read_grid(self, index=0, *args, **kwargs):
        """ Read grid contained in the Grid file

        Parameters
        ----------
        index : int or array_like, optional
           the spin-index for retrieving one of the components. If a vector
           is passed it refers to the fraction per indexed component. I.e.
           ``[0.5, 0.5]`` will return sum of half the first two components.
           Default to the first component.
        """
        # Read the sizes
        nspin, mesh = _siesta.read_grid_sizes(self.file)
        _bin_check(self, 'read_grid', 'could not read grid sizes.')
        # Read the cell and grid
        cell = _siesta.read_grid_cell(self.file)
        _bin_check(self, 'read_grid', 'could not read grid cell.')
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

        cell = np.array(cell.T, np.float64)
        cell.shape = (3, 3)

        # Simply create the grid (with no information)
        # We will overwrite the actual grid
        g = Grid([1, 1, 1], sc=SuperCell(cell))
        # NOTE: there is no need to swap-axes since the returned array is in F ordering
        #       and thus the first axis is the fast (x, y, z) is retained
        g.grid = (grid * self.grid_unit).astype(dtype=np.float32, order='C', copy=False)
        return g


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

    def _is_open(self):
        return self._iu != -1

    def _open_gf(self, mode):
        if mode == 'r':
            self._iu = _siesta.read_open_gf(self.file)
        elif mode == 'w':
            self._iu = _siesta.write_open_gf(self.file)
        _bin_check(self, '_open_gf', 'could not open for {}.'.format({'r': 'reading', 'w': 'writing'}[mode]))

        # Counters to keep track
        self._ie = 0
        self._ik = 0

    def _close_gf(self):
        if not self._is_open():
            return
        # Close it
        _siesta.close_gf(self._iu)
        self._iu = -1

        # Clean variables
        del self._ie
        del self._ik
        try:
            del self._E
            del self._k
        except:
            pass
        try:
            del self._no_u
        except:
            pass
        try:
            del self._nspin
        except:
            pass

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
        self._close_gf()
        self._open_gf('r')
        nspin, no_u, nkpt, NE = _siesta.read_gf_sizes(self._iu)
        _bin_check(self, 'read_header', 'could not read sizes.')

        # We need to re-read (because of k-points)
        self._close_gf()
        self._open_gf('r')

        k, E = _siesta.read_gf_header(self._iu, nkpt, NE)
        _bin_check(self, 'read_header', 'could not read header information.')

        k = k.T
        self._nspin = nspin
        if self._nspin > 2:
            self._no_u = no_u * 2
        else:
            self._no_u = no_u
        self._E = E
        self._k = k
        return nspin, no_u, k, E * Ry2eV

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
        HS = 2 * self._nspin * len(self._k)
        SE = HS / 2 * len(self._E)

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
        self._ik += 1
        self._ie = 1

        H, S = _siesta.read_gf_hs(self._iu, self._no_u)
        _bin_check(self, 'read_hamiltonian', 'could not read Hamiltonian and overlap matrices.')
        return H.T * Ry2eV, S.T

    def read_self_energy(self):
        r""" Read the currently reached bulk self-energy

        The returned self-energy is:

        .. math::
            \boldsymbol \Sigma_{\mathrm{bulk}}(E) = \mathbf S E - \mathbf H - \boldsymbol \Sigma(E)

        Returns
        -------
        complex128 : Self-energy matrix
        """
        SE = _siesta.read_gf_se(self._iu, self._no_u, self._ie).T * Ry2eV
        _bin_check(self, 'read_self_energy', 'could not read self-energy.')
        self._ie += 1
        return SE

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
        cell = obj.geometry.sc.cell * Ang2Bohr
        na_u = obj.geometry.na
        no_u = obj.geometry.no
        xa = obj.geometry.xyz * Ang2Bohr
        # The lasto in siesta requires lasto(0) == 0
        # and secondly, the Python index to fortran
        # index makes firsto behave like fortran lasto
        lasto = obj.geometry.firsto
        bloch = _a.onesi(3)
        mu = mu * eV2Ry
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
        self._E = E * eV2Ry
        self._k = np.copy(k)
        if self._nspin > 2:
            self._no_u = no_u * 2
        else:
            self._no_u = no_u

        # Ensure it is open (in write mode)
        self._close_gf()
        self._open_gf('w')

        # Now write to it...
        _siesta.write_gf_header(self._iu, nspin, cell.T, na_u, no_u, no_u, xa.T, lasto,
                                bloch, 0, mu, k.T, w, self._E, **sizes)
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
        self._ik += 1
        self._ie = 1
        no = len(H)
        if S is None:
            S = np.eye(no, dtype=np.complex128)
        _siesta.write_gf_hs(self._iu, self._ik, self._ie, self._E[self._ie-1],
                            H.astype(np.complex128, 'C', copy=False).T * eV2Ry,
                            S.astype(np.complex128, 'C', copy=False).T, no_u=no)
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
        _siesta.write_gf_se(self._iu, self._ik, self._ie,
                            self._E[self._ie-1],
                            SE.astype(np.complex128, 'C', copy=False).T * eV2Ry, no_u=no)
        _bin_check(self, 'write_self_energy', 'could not write self-energy.')

        self._ie += 1

    def __len__(self):
        return len(self._E) * len(self._k) * self._nspin

    def __iter__(self):
        """ Iterate through the energies and k-points that this GF file is associated with

        Yields
        ------
        bool, list of float, float
        """
        # get everything
        e = self._E * Ry2eV
        if self._nspin in [1, 2]:
            for ispin in range(self._nspin):
                for k in self._k:
                    yield ispin, True, k, e[0]
                    for E in e[1:]:
                        yield ispin, False, k, E

                # Reset counters for k and e
                self._ik = 0
                self._ie = 0
        else:
            for k in self._k:
                yield True, k, e[0]
                for E in e[1:]:
                    yield False, k, E

            self._ik = 0
            self._ie = 0

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
    # These have unit-conversions
    BohrC2AngC = Bohr2Ang ** 3
    add_sile('RHO', _type("rhoSileSiesta", _gridSileSiesta, {'grid_unit': 1./BohrC2AngC}))
    add_sile('RHOINIT', _type("rhoinitSileSiesta", _gridSileSiesta, {'grid_unit': 1./BohrC2AngC}))
    add_sile('RHOXC', _type("rhoxcSileSiesta", _gridSileSiesta, {'grid_unit': 1./BohrC2AngC}))
    add_sile('DRHO', _type("drhoSileSiesta", _gridSileSiesta, {'grid_unit': 1./BohrC2AngC}))
    add_sile('BADER', _type("baderSileSiesta", _gridSileSiesta, {'grid_unit': 1./BohrC2AngC}))
    add_sile('IOCH', _type("iorhoSileSiesta", _gridSileSiesta, {'grid_unit': 1./BohrC2AngC}))
    add_sile('TOCH', _type("totalrhoSileSiesta", _gridSileSiesta, {'grid_unit': 1./BohrC2AngC}))
    add_sile('VH', _type("hartreeSileSiesta", _gridSileSiesta, {'grid_unit': Ry2eV}))
    add_sile('VNA', _type("neutralatomhartreeSileSiesta", _gridSileSiesta, {'grid_unit': Ry2eV}))
    add_sile('VT', _type("totalhartreeSileSiesta", _gridSileSiesta, {'grid_unit': Ry2eV}))
