from __future__ import print_function, division

import numpy as np

# Import sile objects
from ..sile import add_sile, sile_raise_write, SileWarning
from .sile import SileCDFTBtrans
from sisl.utils import *
import sisl._array as _a

# Import the geometry object
from sisl import Geometry, Atom, SuperCell
from sisl import SparseOrbitalBZSpin
from sisl.messages import warn
from sisl._help import _range as range
from sisl.unit.siesta import unit_convert
from ..siesta._help import _mat_spin_convert


__all__ = ['deltancSileTBtrans']

Bohr2Ang = unit_convert('Bohr', 'Ang')
Ry2eV = unit_convert('Ry', 'eV')
eV2Ry = unit_convert('eV', 'Ry')


# The delta nc file
class deltancSileTBtrans(SileCDFTBtrans):
    r""" TBtrans :math:`\delta` file object

    The :math:`\delta` file object is an extension enabled in `TBtrans`_ which
    allows changing the Hamiltonian in transport problems.

    .. math::
        \mathbf H'(\mathbf k) = \mathbf H(\mathbf k) +
            \delta\mathbf H(E, \mathbf k) + \delta\mathbf\Sigma(E, \mathbf k)

    This file may either be used directly as the :math:`\delta\mathbf H` or the
    :math:`\delta\mathbf\Sigma`.

    When writing :math:`\delta` terms using `write_delta` one may add ``k`` or ``E`` arguments
    to make the :math:`\delta` dependent on ``k`` and/or ``E``.

    Refer to the TBtrans manual on how to use this feature.

    Examples
    --------
    >>> H = Hamiltonian(geom.graphene(), dtype=np.complex128)
    >>> H[0, 0] = 1j
    >>> dH = get_sile('deltaH.dH.nc', 'w')
    >>> dH.write_delta(H)
    >>> H[1, 1] = 1.
    >>> dH.write_delta(H, k=[0, 0, 0]) # Gamma only
    >>> H[0, 0] += 1.
    >>> dH.write_delta(H, E=1.) # only at 1 eV
    >>> H[1, 1] += 1.j
    >>> dH.write_delta(H, E=1., k=[0, 0, 0]) # only at 1 eV and Gamma-point
    """

    def read_supercell(self):
        """ Returns the `SuperCell` object from this file """
        cell = _a.arrayd(np.copy(self._value('cell')))
        cell.shape = (3, 3)

        nsc = self._value('nsc')
        sc = SuperCell(cell, nsc=nsc)
        try:
            sc.sc_off = self._value('isc_off')
        except:
            # This is ok, we simply do not have the supercell offsets
            pass

        return sc

    def read_geometry(self, *args, **kwargs):
        """ Returns the `Geometry` object from this file """
        sc = self.read_supercell()

        xyz = _a.arrayd(np.copy(self._value('xa')))
        xyz.shape = (-1, 3)

        # Create list with correct number of orbitals
        lasto = _a.arrayi(np.copy(self._value('lasto')))
        nos = np.append([lasto[0]], np.diff(lasto))
        nos = _a.arrayi(nos)

        if 'atom' in kwargs:
            # The user "knows" which atoms are present
            atms = kwargs['atom']
            # Check that all atoms have the correct number of orbitals.
            # Otherwise we will correct them
            for i in range(len(atms)):
                if atms[i].no != nos[i]:
                    atms[i] = Atom(atms[i].Z, [-1] * nos[i], tag=atms[i].tag)

        else:
            # Default to Hydrogen atom with nos[ia] orbitals
            # This may be counterintuitive but there is no storage of the
            # actual species
            atms = [Atom('H', [-1] * o) for o in nos]

        # Create and return geometry object
        geom = Geometry(xyz, atms, sc=sc)

        return geom

    def write_supercell(self, sc):
        """ Creates the NetCDF file and writes the supercell information """
        sile_raise_write(self)

        # Create initial dimensions
        self._crt_dim(self, 'one', 1)
        self._crt_dim(self, 'n_s', np.prod(sc.nsc))
        self._crt_dim(self, 'xyz', 3)

        # Create initial geometry
        v = self._crt_var(self, 'nsc', 'i4', ('xyz',))
        v.info = 'Number of supercells in each unit-cell direction'
        v[:] = sc.nsc[:]
        v = self._crt_var(self, 'isc_off', 'i4', ('n_s', 'xyz'))
        v.info = "Index of supercell coordinates"
        v[:] = sc.sc_off[:, :]
        v = self._crt_var(self, 'cell', 'f8', ('xyz', 'xyz'))
        v.info = 'Unit cell'
        v.unit = 'Bohr'
        v[:] = sc.cell[:, :] / Bohr2Ang

        # Create designation of the creation
        self.method = 'sisl'

    def write_geometry(self, geometry):
        """ Creates the NetCDF file and writes the geometry information """
        sile_raise_write(self)

        # Create initial dimensions
        self.write_supercell(geometry.sc)
        self._crt_dim(self, 'no_s', np.prod(geometry.nsc) * geometry.no)
        self._crt_dim(self, 'no_u', geometry.no)
        self._crt_dim(self, 'na_u', geometry.na)

        # Create initial geometry
        v = self._crt_var(self, 'lasto', 'i4', ('na_u',))
        v.info = 'Last orbital of equivalent atom'
        v = self._crt_var(self, 'xa', 'f8', ('na_u', 'xyz'))
        v.info = 'Atomic coordinates'
        v.unit = 'Bohr'

        # Save stuff
        self.variables['xa'][:] = geometry.xyz / Bohr2Ang

        bs = self._crt_grp(self, 'BASIS')
        b = self._crt_var(bs, 'basis', 'i4', ('na_u',))
        b.info = "Basis of each atom by ID"

        orbs = _a.emptyi([geometry.na])

        for ia, a, isp in geometry.iter_species():
            b[ia] = isp + 1
            orbs[ia] = a.no
            if a.tag in bs.groups:
                # Assert the file sizes
                if bs.groups[a.tag].Number_of_orbitals != a.no:
                    raise ValueError(('File {0}'
                                      ' has erroneous data in regards of '
                                      'of the alreay stored dimensions.').format(self.file))
            else:
                ba = bs.createGroup(a.tag)
                ba.ID = np.int32(isp + 1)
                ba.Atomic_number = np.int32(a.Z)
                ba.Mass = a.mass
                ba.Label = a.tag
                ba.Element = a.symbol
                ba.Number_of_orbitals = np.int32(a.no)

        # Store the lasto variable as the remaining thing to do
        self.variables['lasto'][:] = _a.cumsumi(orbs)

    def _get_lvl_k_E(self, **kwargs):
        """ Return level, k and E indices, in that order.

        The indices are negative if a new index needs to be created.
        """
        # Determine the type of dH we are storing...
        k = kwargs.get('k', None)
        if k is not None:
            k = _a.asarrayd(k).flatten()
        E = kwargs.get('E', None)

        if (k is None) and (E is None):
            ilvl = 1
        elif (k is not None) and (E is None):
            ilvl = 2
        elif (k is None) and (E is not None):
            ilvl = 3
            # Convert to Rydberg
            E = E * eV2Ry
        elif (k is not None) and (E is not None):
            ilvl = 4
            # Convert to Rydberg
            E = E * eV2Ry

        try:
            lvl = self._get_lvl(ilvl)
        except:
            return ilvl, -1, -1

        # Now determine the energy and k-indices
        iE = -1
        if ilvl in [3, 4]:
            if lvl.variables['E'].size != 0:
                Es = _a.arrayd(lvl.variables['E'][:])
                iE = np.argmin(np.abs(Es - E))
                if abs(Es[iE] - E) > 0.0001:
                    iE = -1

        ik = -1
        if ilvl in [2, 4]:
            if lvl.variables['kpt'].size != 0:
                kpt = _a.arrayd(lvl.variables['kpt'][:])
                kpt.shape = (-1, 3)
                ik = np.argmin(np.abs(kpt - k[None, :]).sum(axis=1))
                if not np.allclose(kpt[ik, :], k, atol=0.0001):
                    ik = -1

        return ilvl, ik, iE

    def _get_lvl(self, ilvl):
        slvl = 'LEVEL-'+str(ilvl)
        if slvl in self.groups:
            return self._crt_grp(self, slvl)
        raise ValueError("Level {0} does not exist in {1}.".format(ilvl, self.file))

    def _add_lvl(self, ilvl):
        """ Simply adds and returns a group if it does not exist it will be created """
        slvl = 'LEVEL-' + str(ilvl)
        if slvl in self.groups:
            lvl = self._crt_grp(self, slvl)
        else:
            lvl = self._crt_grp(self, slvl)
            if ilvl in [2, 4]:
                self._crt_dim(lvl, 'nkpt', None)
                self._crt_var(lvl, 'kpt', 'f8', ('nkpt', 'xyz'),
                              attr = {'info': 'k-points for delta values',
                                      'unit': 'b**-1'})
            if ilvl in [3, 4]:
                self._crt_dim(lvl, 'ne', None)
                self._crt_var(lvl, 'E', 'f8', ('ne',),
                              attr = {'info': 'Energy points for delta values',
                                      'unit': 'Ry'})

        return lvl

    def write_delta(self, delta, **kwargs):
        r""" Writes a :math:`\delta` Hamiltonian to the file

        This term may be of

        - level-1: no E or k dependence
        - level-2: k-dependent
        - level-3: E-dependent
        - level-4: k- and E-dependent

        Parameters
        ----------
        delta : SparseOrbitalBZSpin
           the model to be saved in the NC file
        k : array_like, optional
           a specific k-point :math:`\delta` term. I.e. only save the :math:`\delta` term for
           the given k-point. May be combined with `E` for a specific k and energy point.
        E : float, optional
           an energy dependent :math:`\delta` term. I.e. only save the :math:`\delta` term for
           the given energy. May be combined with `k` for a specific k and energy point.
        """
        # Ensure finalization
        delta.finalize()

        # Ensure that the geometry is written
        self.write_geometry(delta.geom)

        self._crt_dim(self, 'spin', len(delta.spin))

        # Determine the type of delta we are storing...
        k = kwargs.get('k', None)
        E = kwargs.get('E', None)

        ilvl, ik, iE = self._get_lvl_k_E(**kwargs)
        lvl = self._add_lvl(ilvl)

        # Append the sparsity pattern
        # Create basis group
        if 'n_col' in lvl.variables:
            if len(lvl.dimensions['nnzs']) != delta.nnz:
                raise ValueError("The sparsity pattern stored in delta *MUST* be equivalent for "
                                 "all delta entries [nnz].")
            if np.any(lvl.variables['n_col'][:] != delta._csr.ncol[:]):
                raise ValueError("The sparsity pattern stored in delta *MUST* be equivalent for "
                                 "all delta entries [n_col].")
            if np.any(lvl.variables['list_col'][:] != delta._csr.col[:]+1):
                raise ValueError("The sparsity pattern stored in delta *MUST* be equivalent for "
                                 "all delta entries [list_col].")
            if np.any(lvl.variables['isc_off'][:] != delta.geometry.sc.sc_off):
                raise ValueError("The sparsity pattern stored in delta *MUST* be equivalent for "
                                 "all delta entries [sc_off].")
        else:
            self._crt_dim(lvl, 'nnzs', delta.nnz)
            v = self._crt_var(lvl, 'n_col', 'i4', ('no_u',))
            v.info = "Number of non-zero elements per row"
            v[:] = delta._csr.ncol[:]
            v = self._crt_var(lvl, 'list_col', 'i4', ('nnzs',),
                              chunksizes=(delta.nnz,), **self._cmp_args)
            v.info = "Supercell column indices in the sparse format"
            v[:] = delta._csr.col[:] + 1  # correct for fortran indices
            v = self._crt_var(lvl, 'isc_off', 'i4', ('n_s', 'xyz'))
            v.info = "Index of supercell coordinates"
            v[:] = delta.geometry.sc.sc_off[:, :]

        warn_E = True
        if ilvl in [3, 4]:
            if iE < 0:
                # We need to add the new value
                iE = lvl.variables['E'].shape[0]
                lvl.variables['E'][iE] = E * eV2Ry
                warn_E = False

        warn_k = True
        if ilvl in [2, 4]:
            if ik < 0:
                ik = lvl.variables['kpt'].shape[0]
                lvl.variables['kpt'][ik, :] = k
                warn_k = False

        if ilvl == 4 and warn_k and warn_E and False:
            # As soon as we have put the second k-point and the first energy
            # point, this warning will proceed...
            # I.e. even though the variable has not been set, it will WARN
            # Hence we out-comment this for now...
            warn(SileWarning('Overwriting k-point {0} and energy point {1} correction.'.format(ik, iE)))
        elif ilvl == 3 and warn_E:
            warn(SileWarning('Overwriting energy point {0} correction.'.format(iE)))
        elif ilvl == 2 and warn_k:
            warn(SileWarning('Overwriting k-point {0} correction.'.format(ik)))

        if ilvl == 1:
            dim = ('spin', 'nnzs')
            sl = [slice(None)] * 2
            csize = [1] * 2
        elif ilvl == 2:
            dim = ('nkpt', 'spin', 'nnzs')
            sl = [slice(None)] * 3
            sl[0] = ik
            csize = [1] * 3
        elif ilvl == 3:
            dim = ('ne', 'spin', 'nnzs')
            sl = [slice(None)] * 3
            sl[0] = iE
            csize = [1] * 3
        elif ilvl == 4:
            dim = ('nkpt', 'ne', 'spin', 'nnzs')
            sl = [slice(None)] * 4
            sl[0] = ik
            sl[1] = iE
            csize = [1] * 4

        # Number of non-zero elements
        csize[-1] = delta.nnz

        if delta.spin.kind > delta.spin.POLARIZED:
            raise ValueError(self.__class__.__name__ + '.write_delta only allows spin-polarized delta values')

        if delta.dtype.kind == 'c':
            v1 = self._crt_var(lvl, 'Redelta', 'f8', dim,
                               chunksizes=csize,
                               attr = {'info': "Real part of delta",
                                       'unit': "Ry"}, **self._cmp_args)
            v2 = self._crt_var(lvl, 'Imdelta', 'f8', dim,
                               chunksizes=csize,
                               attr = {'info': "Imaginary part of delta",
                                       'unit': "Ry"}, **self._cmp_args)
            for i in range(len(delta.spin)):
                sl[-2] = i
                v1[sl] = delta._csr._D[:, i].real * eV2Ry
                v2[sl] = delta._csr._D[:, i].imag * eV2Ry

        else:
            v = self._crt_var(lvl, 'delta', 'f8', dim,
                              chunksizes=csize,
                              attr = {'info': "delta",
                                      'unit': "Ry"},  **self._cmp_args)
            for i in range(len(delta.spin)):
                sl[-2] = i
                v[sl] = delta._csr._D[:, i] * eV2Ry

    def _read_class(self, cls, **kwargs):
        """ Reads a class model from a file """

        # Ensure that the geometry is written
        geom = self.read_geometry()

        # Determine the type of delta we are storing...
        E = kwargs.get('E', None)

        ilvl, ik, iE = self._get_lvl_k_E(**kwargs)

        # Get the level
        lvl = self._get_lvl(ilvl)

        if iE < 0 and ilvl in [3, 4]:
            raise ValueError("Energy {0} eV does not exist in the file.".format(E))
        if ik < 0 and ilvl in [2, 4]:
            raise ValueError("k-point requested does not exist in the file.")

        if ilvl == 1:
            sl = [slice(None)] * 2
        elif ilvl == 2:
            sl = [slice(None)] * 3
            sl[0] = ik
        elif ilvl == 3:
            sl = [slice(None)] * 3
            sl[0] = iE
        elif ilvl == 4:
            sl = [slice(None)] * 4
            sl[0] = ik
            sl[1] = iE

        # Now figure out what data-type the delta is.
        if 'Redelta' in lvl.variables:
            # It *must* be a complex valued Hamiltonian
            is_complex = True
            dtype = np.complex128
        elif 'delta' in lvl.variables:
            is_complex = False
            dtype = np.float64

        # Get number of spins
        nspin = len(self.dimensions['spin'])

        # Now create the sparse matrix stuff (we re-create the
        # array, hence just allocate the smallest amount possible)
        C = cls(geom, nspin, nnzpr=1, dtype=dtype, orthogonal=True)

        C._csr.ncol = _a.arrayi(lvl.variables['n_col'][:])
        # Update maximum number of connections (in case future stuff happens)
        C._csr.ptr = np.insert(_a.cumsumi(C._csr.ncol), 0, 0)
        C._csr.col = _a.arrayi(lvl.variables['list_col'][:]) - 1

        # Copy information over
        C._csr._nnz = len(C._csr.col)
        C._csr._D = np.empty([C._csr.ptr[-1], nspin], dtype)
        if is_complex:
            for ispin in range(nspin):
                sl[-2] = ispin
                C._csr._D[:, ispin].real = lvl.variables['Redelta'][sl] * Ry2eV
                C._csr._D[:, ispin].imag = lvl.variables['Imdelta'][sl] * Ry2eV
        else:
            for ispin in range(nspin):
                sl[-2] = ispin
                C._csr._D[:, ispin] = lvl.variables['delta'][sl] * Ry2eV

        _mat_spin_convert(C)

        return C

    def read_delta(self, **kwargs):
        """ Reads a delta model from the file """
        return self._read_class(SparseOrbitalBZSpin, **kwargs)

add_sile('delta.nc', deltancSileTBtrans)
add_sile('dH.nc', deltancSileTBtrans)
add_sile('dSE.nc', deltancSileTBtrans)
