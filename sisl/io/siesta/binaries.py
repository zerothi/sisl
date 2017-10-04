from __future__ import print_function

import numpy as np

try:
    from . import _siesta
    found_module = True
except Exception as e:
    found_module = False

# Import sile objects
from ..sile import add_sile
from .sile import SileBinSiesta

# Import the geometry object
import sisl._array as _a
from sisl import Geometry, Atom, SuperCell, Grid
from sisl.unit.siesta import unit_convert
from sisl.physics import Hamiltonian

Ang2Bohr = unit_convert('Ang', 'Bohr')
eV2Ry = unit_convert('eV', 'Ry')
Bohr2Ang = unit_convert('Bohr', 'Ang')
Ry2eV = unit_convert('Ry', 'eV')

__all__ = ['TSHSSileSiesta']
__all__ += ['GridSileSiesta', 'EnergyGridSileSiesta']
__all__ += ['_GFSileSiesta', 'TSGFSileSiesta']


class TSHSSileSiesta(SileBinSiesta):
    """ TranSiesta file object """

    def read_supercell(self):
        """ Returns a SuperCell object from a siesta.TSHS file """
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
        ncol, col, dH, dS = _siesta.read_tshs_hs(self.file, spin, no, nnz)

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

    def write_hamiltonian(self, H, **kwargs):
        """ Writes the Hamiltonian to a `TSHS` file """
        # Ensure the Hamiltonian is finalized
        H.finalize()

        # Extract the data to pass to the fortran routine

        cell = H.geom.cell * Ang2Bohr
        xyz = H.geom.xyz * Ang2Bohr

        # Pointer to CSR matrix
        csr = H._csr

        # Get H and S
        if H.orthogonal:
            h = csr._D * eV2Ry
            s = csr.diags(1., dim=1)
            # Ensure all data is correctly formatted (i.e. have the same sparsity pattern
            s.align(csr)
            s.finalize()
            if s.nnz != len(h):
                raise ValueError(("The diagonal elements of your orthogonal Hamiltonian have not been defined, "
                                  "this is a requirement."))
            s = s._D[:, 0]
        else:
            h = csr._D[:, :H.S_idx-1] * eV2Ry
            s = csr._D[:, H.S_idx]
        # Ensure shapes (say if only 1 spin)
        h.shape = (-1, len(H.spin))
        s.shape = (-1,)

        # Get shorter variants
        nsc = H.geom.nsc[:]
        isc = H.geom.sc.sc_off[:, :]

        # I can't seem to figure out the usage of f2py
        # Below I get an error if xyz is not transposed and h is transposed,
        # however, they are both in C-contiguous arrays and this is indeed weird... :(
        _siesta.write_tshs_hs(self.file, nsc[0], nsc[1], nsc[2],
                              cell.T, xyz.T, H.geom.firsto,
                              csr.ncol, csr.col + 1, h, s, isc.T,
                              nspin=len(H.spin), na_u=H.geom.na, no_u=H.geom.no, nnz=H.nnz)


class GridSileSiesta(SileBinSiesta):
    """ Grid file object from a binary Siesta output file """

    def _setup(self, *args, **kwargs):
        self.grid_unit = 1.

    def read_supercell(self, *args, **kwargs):

        cell = _siesta.read_grid_cell(self.file)
        cell = np.array(cell.T, np.float64)
        cell.shape = (3, 3)

        return SuperCell(cell)

    def read_grid(self, spin=0, *args, **kwargs):
        """ Read grid contained in the Grid file

        Parameters
        ----------
        spin : int, optional
           the returned spin
        """
        # Read the sizes
        nspin, mesh = _siesta.read_grid_sizes(self.file)
        # Read the cell and grid
        cell, grid = _siesta.read_grid(self.file, nspin, mesh[0], mesh[1], mesh[2])

        if grid.ndim == 4:
            grid = grid[spin, :, :, :]

        cell = np.array(cell.T, np.float64)
        cell.shape = (3, 3)

        g = Grid(mesh, sc=SuperCell(cell), dtype=np.float32)
        g.grid = np.array(grid.swapaxes(0, 2), np.float32) * self.grid_unit
        return g


class EnergyGridSileSiesta(GridSileSiesta):
    """ Energy grid file object from a binary Siesta output file """

    def _setup(self, *args, **kwargs):
        self.grid_unit = Ry2eV


class _GFSileSiesta(SileBinSiesta):
    """ Surface Green function file for inclusion in TranSiesta and TBtrans """

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

    def write_hamiltonian(self, H, S):
        """ Write the current energy, k-point and H and S to the file

        Parameters
        ----------
        H : matrix
           a square matrix corresponding to the Hamiltonian
        S : matrix
           a square matrix corresponding to the overlap
        """
        # Step k
        self._ik += 1
        self._ie = 1
        no = len(H)
        _siesta.write_gf_hs(self._iu, self._ik, self._ie, self._E[self._ie-1],
                            H.T * eV2Ry, S.T, no_u=no)

    def write_self_energy(self, SE):
        """ Write the current energy, k-point and H and S to the file

        Parameters
        ----------
        SE : matrix
           a square matrix corresponding to the self-energy (Green function)
        """
        no = len(SE)
        _siesta.write_gf_se(self._iu, self._ik, self._ie,
                            self._E[self._ie-1], SE.T * eV2Ry, no_u=no)
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


def _type(name, obj):
    return type(name, (obj, ), {})

# Faster than class ... \ pass
TSGFSileSiesta = _type("TSGFSileSiesta", _GFSileSiesta)

if found_module:
    add_sile('TSHS', TSHSSileSiesta)
    add_sile('RHO', _type("RhoSileSiesta", GridSileSiesta))
    add_sile('RHOINIT', _type("RhoInitSileSiesta", GridSileSiesta))
    add_sile('DRHO', _type("dRhoSileSiesta", GridSileSiesta))
    add_sile('IOCH', _type("IoRhoSileSiesta", GridSileSiesta))
    add_sile('TOCH', _type("TotalRhoSileSiesta", GridSileSiesta))
    add_sile('VH', _type("HartreeSileSiesta", EnergyGridSileSiesta))
    add_sile('VNA', _type("NeutralAtomHartreeSileSiesta", EnergyGridSileSiesta))
    add_sile('VT', _type("TotalHartreeSileSiesta", EnergyGridSileSiesta))

    add_sile('TSGF', TSGFSileSiesta)
