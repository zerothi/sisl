from __future__ import print_function

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
from sisl import Geometry, Atom, SuperCell, Grid
from sisl.unit.siesta import unit_convert
from sisl.physics.sparse import SparseOrbitalBZ
from sisl.physics import Hamiltonian, DensityMatrix, EnergyDensityMatrix
from ._help import *


Ang2Bohr = unit_convert('Ang', 'Bohr')
eV2Ry = unit_convert('eV', 'Ry')
Bohr2Ang = unit_convert('Bohr', 'Ang')
Ry2eV = unit_convert('Ry', 'eV')

__all__ = ['tshsSileSiesta', 'tsdeSileSiesta']
__all__ += ['hsxSileSiesta', 'dmSileSiesta']
__all__ += ['gridSileSiesta']
__all__ += ['tsgfSileSiesta']


class tshsSileSiesta(SileBinSiesta):
    """ Geometry, Hamiltonian and overlap matrix file """

    def read_supercell(self):
        """ Returns a SuperCell object from a siesta.TSHS file """
        n_s = _siesta.read_tshs_sizes(self.file)[3]
        arr = _siesta.read_tshs_cell(self.file, n_s)
        nsc = np.array(arr[0], np.int32)
        cell = np.array(arr[1].T, np.float64)
        cell.shape = (3, 3)
        return SuperCell(cell, nsc=nsc)

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
            atoms.append(Atom(Z+1, [-1] * orb))

        def get_atom(atoms, orbs):
            for atom in atoms:
                if atom.no == orbs:
                    return atom

        atom = []
        for _, orb in enumerate(orbs):
            atom.append(get_atom(atoms, orb))

        # Create and return geometry object
        geom = Geometry(xyz, atom, sc=sc)

        return geom

    def read_hamiltonian(self, **kwargs):
        """ Returns the electronic structure from the siesta.TSHS file """
        geom = self.read_geometry()

        # read the sizes used...
        sizes = _siesta.read_tshs_sizes(self.file)
        isc = _siesta.read_tshs_cell(self.file, sizes[3])[2].T
        spin = sizes[0]
        no = sizes[2]
        nnz = sizes[4]
        ncol, col, dH, dS = _siesta.read_tshs_hs(self.file, spin, no, nnz)

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
            raise SileError(self.__class__.__name__ + '.read_hamiltonian could not assert the '
                            'supercell connections in the primary unit-cell.')

        return H

    def read_overlap(self, **kwargs):
        """ Returns the overlap matrix from the siesta.TSHS file """
        geom = self.read_geometry()

        # read the sizes used...
        sizes = _siesta.read_tshs_sizes(self.file)
        isc = _siesta.read_tshs_cell(self.file, sizes[3])[2].T
        no = sizes[2]
        nnz = sizes[4]
        ncol, col, dS = _siesta.read_tshs_s(self.file, no, nnz)

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

    def write_hamiltonian(self, H, **kwargs):
        """ Writes the Hamiltonian to a siesta.TSHS file """
        H.finalize()
        csr = H._csr.copy()
        if csr.nnz == 0:
            raise SileError(str(self) + '.write_hamiltonian cannot write a zero element sparse matrix!')

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
                raise SislError("The diagonal elements of your orthogonal Hamiltonian have not been defined, "
                                "this is a requirement.")
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


class dmSileSiesta(SileBinSiesta):
    """ Density matrix file """

    def read_density_matrix(self, **kwargs):
        """ Returns the density matrix from the siesta.DM file """

        # Now read the sizes used...
        spin, no, nsc, nnz = _siesta.read_dm_sizes(self.file)
        ncol, col, dDM = _siesta.read_dm(self.file, spin, no, nsc, nnz)

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
            raise ValueError("Reading DM files requires the input geometry to have the "
                             "correct number of orbitals.")

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
            raise SileError(str(self) + '.write_density_matrix cannot write a zero element sparse matrix!')

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


class tsdeSileSiesta(dmSileSiesta):
    """ Non-equilibrium density matrix and energy density matrix file """

    def read_energy_density_matrix(self, **kwargs):
        """ Returns the energy density matrix from the siesta.DM file """

        # Now read the sizes used...
        spin, no, nsc, nnz = _siesta.read_tsde_sizes(self.file)
        ncol, col, dEDM = _siesta.read_tsde_edm(self.file, spin, no, nsc, nnz)

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
            raise ValueError("Reading EDM files requires the input geometry to have the "
                             "correct number of orbitals.")

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
        ncol, col, dH, dS, dxij = _siesta.read_hsx_hsx(self.file, Gamma, spin, no, no_s, nnz)

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
                warn(self.__class__.__name__ + ".read_hamiltonian (currently we can not calculate atomic positions from"
                     " xij array)")
        if geom.no != no:
            raise ValueError("Reading HSX files requires the input geometry to have the "
                             "correct number of orbitals {} / {}.".format(no, geom.no))

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
        ncol, col, dS = _siesta.read_hsx_s(self.file, Gamma, spin, no, no_s, nnz)

        geom = kwargs.get('geometry', kwargs.get('geom', None))
        if geom is None:
            warn(self.__class__.__name__ + ".read_overlap requires input geometry to assign S")
        if geom.no != no:
            raise ValueError("Reading HSX files requires the input geometry to have the "
                             "correct number of orbitals {} / {}.".format(no, geom.no))

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


class gridSileSiesta(SileBinSiesta):
    """ Binary real-space grid file """

    def read_supercell(self, *args, **kwargs):

        cell = _siesta.read_grid_cell(self.file)
        cell = np.array(cell.T, np.float64)
        cell.shape = (3, 3)

        return SuperCell(cell)

    def read_grid(self, spin=0, *args, **kwargs):
        """ Read grid contained in the Grid file

        Parameters
        ----------
        spin : int or array_like, optional
           the spin-index for retrieving one of the components. If a vector
           is passed it refers to the fraction per indexed component. I.e.
           ``[0.5, 0.5]`` will return sum of half the first two components.
           Default to the first component.
        """
        # Read the sizes
        nspin, mesh = _siesta.read_grid_sizes(self.file)
        # Read the cell and grid
        cell = _siesta.read_grid_cell(self.file)
        grid = _siesta.read_grid(self.file, nspin, mesh[0], mesh[1], mesh[2])

        if isinstance(spin, Integral):
            grid = grid[:, :, :, spin]
        else:
            if len(spin) > grid.shape[0]:
                raise ValueError(self.__class__.__name__ + '.read_grid requires spin to be an integer or '
                                 'an array of length equal to the number of spin components.')
            g = grid[:, :, :, 0] * spin[0]
            for i, scale in enumerate(spin[1:]):
                g += grid[:, :, :, 1+i] * scale
            grid = g

        cell = np.array(cell.T, np.float64)
        cell.shape = (3, 3)

        g = Grid(mesh, sc=SuperCell(cell), dtype=np.float32)
        g.grid = np.array(grid.swapaxes(0, 2), np.float32) * self.grid_unit
        return g


class _gfSileSiesta(SileBinSiesta):
    """ Surface Green function file containing, Hamiltonian, overlap matrix and self-energies """

    def _setup(self, *args, **kwargs):
        """ Simple setup that needs to be overwritten """
        self._iu = -1

    def _is_open(self):
        return self._iu > 0

    def _open_gf(self):
        self._iu = _siesta.open_gf(self.file)
        # Counters to keep track
        self._ie = 0
        self._ik = 0

    def _close_gf(self):
        if not self._is_open():
            return
        # Close it
        _siesta.close_gf(self._iu)
        del self._ie
        del self._ik
        try:
            del self._E
            del self._k
        except:
            pass

    def write_header(self, E, bz, obj, mu=0.):
        """ Write to the binary file the header of the file

        Parameters
        ----------
        E : array_like of cmplx or float
           the energy points. If `obj` is an instance of `SelfEnergy` where an
           associated ``eta`` is defined then `E` may be float, otherwise
           it *has* to be a complex array.
        bz : BrillouinZone
           contains the k-points and their weights
        obj : ...
           an object that contains the Hamiltonian definitions
        """
        nspin = len(obj.spin)
        cell = obj.geom.sc.cell * Ang2Bohr
        na_u = obj.geom.na
        no_u = obj.geom.no
        xa = obj.geom.xyz * Ang2Bohr
        # The lasto in siesta requires lasto(0) == 0
        # and secondly, the Python index to fortran
        # index makes firsto behave like fortran lasto
        lasto = obj.geom.firsto
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

        self._E = np.copy(E) * eV2Ry
        self._k = np.copy(k)

        # Ensure it is open
        self._close_gf()
        self._open_gf()

        # Now write to it...
        _siesta.write_gf_header(self._iu, nspin, cell.T, na_u, no_u, no_u, xa.T, lasto,
                                bloch, 0, mu, k.T, w, self._E, **sizes)

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
        # Step k
        self._ik += 1
        self._ie = 1
        no = len(H)
        if S is None:
            S = np.eye(no, dtype=np.complex128)
        _siesta.write_gf_hs(self._iu, self._ik, self._ie, self._E[self._ie-1],
                            H.astype(np.complex128, 'C', copy=False).T * eV2Ry,
                            S.astype(np.complex128, 'C', copy=False).T, no_u=no)

    def write_self_energy(self, SE):
        """ Write the current energy, k-point and H and S to the file

        Parameters
        ----------
        SE : matrix
           a square matrix corresponding to the self-energy (Green function)
        """
        no = len(SE)
        _siesta.write_gf_se(self._iu, self._ik, self._ie,
                            self._E[self._ie-1],
                            SE.astype(np.complex128, 'C', copy=False).T * eV2Ry, no_u=no)
        # Step energy counter
        self._ie += 1

    def __iter__(self):
        """ Iterate through the energies and k-points that this GF file is associated with

        Yields
        ------
        bool, list of float, float
        """
        for k in self._k:
            yield True, k, self._E[0] * Ry2eV
            for E in self._E[1:] * Ry2eV:
                yield False, k, E
        # We will automatically close once we hit the end
        self._close_gf()


def _type(name, obj, dic=None):
    if dic is None:
        dic = {}
    return type(name, (obj, ), dic)

# Faster than class ... \ pass
tsgfSileSiesta = _type("tsgfSileSiesta", _gfSileSiesta)

if found_module:
    add_sile('TSHS', tshsSileSiesta)
    add_sile('TSDE', tsdeSileSiesta)
    add_sile('DM', dmSileSiesta)
    add_sile('HSX', hsxSileSiesta)
    # These have unit-conversions
    BohrC2AngC = Bohr2Ang ** 3
    add_sile('RHO', _type("rhoSileSiesta", gridSileSiesta, {'grid_unit': 1./BohrC2AngC}))
    add_sile('RHOINIT', _type("rhoinitSileSiesta", gridSileSiesta, {'grid_unit': 1./BohrC2AngC}))
    add_sile('DRHO', _type("drhoSileSiesta", gridSileSiesta, {'grid_unit': 1./BohrC2AngC}))
    add_sile('IOCH', _type("iorhoSileSiesta", gridSileSiesta, {'grid_unit': 1./BohrC2AngC}))
    add_sile('TOCH', _type("totalrhoSileSiesta", gridSileSiesta, {'grid_unit': 1./BohrC2AngC}))
    add_sile('VH', _type("hartreeSileSiesta", gridSileSiesta, {'grid_unit': Ry2eV}))
    add_sile('VNA', _type("neutralatomhartreeSileSiesta", gridSileSiesta, {'grid_unit': Ry2eV}))
    add_sile('VT', _type("totalhartreeSileSiesta", gridSileSiesta, {'grid_unit': Ry2eV}))
    add_sile('TSGF', tsgfSileSiesta)
