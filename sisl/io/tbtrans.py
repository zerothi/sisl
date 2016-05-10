"""
Sile object for reading TBtrans binary files
"""
from __future__ import print_function

# Import sile objects
from sisl.io.sile import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell
from sisl import Bohr

# The sparse matrix for the orbital/bond currents
from scipy.sparse import csr_matrix, lil_matrix

import numpy as np

# Check against integers
from numbers import Integral

__all__ = ['TBtransSile', 'PHtransSile']


class TBtransSile(NCSile):
    """ TBtrans file object """
    _trans_type = 'TBT'

    def _get_var(self, name, tree=None):
        """ Local method to get the NetCDF variable """
        if tree is None:
            return self.variables[name]

        g = self
        if isinstance(tree, list):
            for t in tree:
                g = g.groups[t]
        else:
            g = g.groups[tree]

        return g.variables[name]

    def _data(self, name, tree=None):
        """ Local method for obtaining the data from the NCSile.

        This method checks how the file is access, i.e. whether
        data is stored in the object or it should be read consequtively.
        """
        if self._access > 0:
            if name in self.__data:
                return self.__data[name]

        return self._get_var(name, tree=tree)[:]

    def _data_avg(self, name, tree=None, avg=False):
        """ Local method for obtaining the data from the NCSile.

        This method checks how the file is access, i.e. whether
        data is stored in the object or it should be read consequtively.
        """
        if self._access > 0:
            if name in self.__data:
                return self.__data[name]

        v = self._get_var(name, tree=tree)
        wkpt = self.wkpt

        # Perform normalization
        orig_shape = v.shape
        if isinstance(avg, bool):
            if avg:
                nk = len(wkpt)
                data = v[0, ...] * wkpt[0]
                for i in range(1, nk):
                    data += v[i, :] * wkpt[i]
                data.shape = orig_shape[1:]
            else:
                data = v[:]

        elif isinstance(avg, Integral):
            data = v[avg, ...] * wkpt[avg]
            data.shape = orig_shape[1:]

        else:
            # We assume avg is some kind of itterable
            data = v[avg[0], ...] * wkpt[avg[0]]
            for i in range(1, len(avg)):
                data += v[avg[i], ...] * wkpt[avg[i]]
            data.shape = orig_shape[1:]

        # Return data
        return data

    def _data_E(self, name, tree=None, avg=False, E=None):
        """ Local method for obtaining the data from the NCSile using an E index.

        """
        if E is None:
            return self._data_avg(name, tree, avg)

        # Ensure that it is an index
        iE = self.E2idx(E)

        if self._access > 0:
            raise RuntimeError(
                "data_E is not allowed for access-contained items.")

        v = self._get_var(name, tree=tree)
        wkpt = self.wkpt

        # Perform normalization
        orig_shape = v.shape

        if isinstance(avg, bool):
            if avg:
                nk = len(wkpt)
                data = np.array(v[0, iE, ...]) * wkpt[0]
                for i in range(1, nk):
                    data += v[i, iE, ...] * wkpt[i]
                data.shape = orig_shape[1:]
            else:
                data = np.array(v[:, iE, ...])

        elif isinstance(avg, Integral):
            data = np.array(v[avg, iE, ...]) * wkpt[avg]
            data.shape = orig_shape[1:]

        else:
            # We assume avg is some kind of itterable
            data = v[avg[0], iE, ...] * wkpt[avg[0]]
            for i in range(1, len(avg)):
                data += v[avg[i], iE, ...] * wkpt[avg[i]]
            data.shape = orig_shape[1:]

        # Return data
        return data

    def _setup(self):
        """ Setup the special object for data containing """
        self.__data = dict()

        if self._access > 0:

            # Fake double calls
            access = self._access
            self._access = 0

            # There are certain elements which should
            # be minimal on memory but allow for
            # fast access by the object.
            for d in ['cell', 'xa', 'lasto',
                      'a_dev', 'pivot',
                      'kpt', 'wkpt', 'E']:
                self.__data[d] = self._data(d)

            # Create the geometry in the data file
            self.__data['_geom'] = self.read_geom()

            # Reset the access pattern
            self._access = access

    def read_sc(self):
        """ Returns `SuperCell` object from a .TBT.nc file """
        if not hasattr(self, 'fh'):
            with self:
                return self.read_sc()

        cell = np.array(np.copy(self.cell), dtype=np.float64)
        cell.shape = (3, 3)

        return SuperCell(cell)

    def read_geom(self):
        """ Returns Geometry object from a .TBT.nc file """
        # Quick access to the geometry object
        if self._access > 0 and '_geom' in self.__data:
            return self.__data['_geom']

        if not hasattr(self, 'fh'):
            with self:
                return self.read_geom()

        sc = self.read_sc()

        xyz = np.array(np.copy(self.xa), dtype=np.float64)
        xyz.shape = (-1, 3)

        # Create list with correct number of orbitals
        lasto = np.array(np.copy(self.lasto), dtype=np.int32)
        nos = np.append([lasto[0]], np.diff(lasto))
        nos = np.array(nos, np.int32)

        # Default to Hydrogen atom with nos[ia] orbitals
        # This may be counterintuitive but there is no storage of the
        # actual species
        atms = [Atom(Z='H', orbs=o) for o in nos]

        # Create and return geometry object
        geom = Geometry(xyz, atoms=atms, sc=sc)

        return geom

    def write_geom(self):
        """ This does not work """
        raise ValueError(self.__class__.__name__ + " can not write a geometry")

    # This class also contains all the important quantities elements of the
    # file.

    @property
    def geom(self):
        """ Returns the associated geometry from the TBT file """
        return self.read_geom()

    @property
    def cell(self):
        """ Unit cell in file """
        return self._data('cell') / Bohr

    @property
    def na(self):
        """ Returns number of atoms in the cell """
        return int(len(self.dimensions['na_u']))
    na_u = na

    @property
    def no(self):
        """ Returns number of orbitals in the cell """
        return int(len(self.dimensions['no_u']))
    no_u = no

    @property
    def xa(self):
        """ Atomic coordinates in file """
        return self._data('xa') / Bohr
    xyz = xa

    # Device atoms and other quantities
    @property
    def na_d(self):
        """ Number of atoms in the device region """
        return len(self.dimensions['na_d'])
    na_dev = na_d

    @property
    def a_d(self):
        """ Atomic indices (1-based) of device atoms """
        return self._data('a_dev')
    a_dev = a_d

    @property
    def pivot(self):
        """ Pivot table of device orbitals to obtain input sorting """
        return self._data('pivot')
    pvt = pivot

    @property
    def lasto(self):
        """ Last orbital of corresponding atom """
        return self._data('lasto')

    @property
    def no_d(self):
        """ Number of orbitals in the device region """
        return int(len(self.dimensions['no_d']))

    @property
    def kpt(self):
        """ Sampled k-points in file """
        return self._data('kpt')

    @property
    def wkpt(self):
        """ Weights of k-points in file """
        return self._data('wkpt')

    @property
    def nkpt(self):
        """ Number of k-points in file """
        return len(self.dimensions['nkpt'])

    @property
    def E(self):
        """ Sampled energy-points in file """
        return self._data('E') / Ry

    def E2idx(self, E):
        """ Return the closest energy index corresponding to the energy `E`"""
        if isinstance(E, Integral):
            return E
        RyE = E * Ry
        return np.abs(self._data('E') - RyE).argmin()

    @property
    def ne(self):
        """ Number of energy-points in file """
        return len(self.dimensions['ne'])
    nE = ne

    @property
    def elecs(self):
        """ List of electrodes """
        elecs = self.groups.keys()

        # in cases of not calculating all
        # electrode transmissions we must ensure that
        # we add the last one
        var = self.groups[elecs[0]].variables.keys()
        for tvar in var:
            if tvar.endswith('.T'):
                tvar = tvar.split('.')[0]
                if tvar not in elecs:
                    elecs.append(tvar)
        return elecs
    electrodes = elecs
    Electrodes = elecs
    Elecs = elecs

    def chemical_potential(self, elec):
        """ Return the chemical potential associated with the electrode `elec` """
        return self._data('mu', elec)
    mu = chemical_potential

    def electronic_temperature(self, elec):
        """ Return temperature of the electrode electronic distribution """
        return self._data('kT', elec)
    kT = electronic_temperature

    def transmission(self, elec_from, elec_to, avg=True):
        """ Return the transmission from `from` to `to`.

        The transmission between two electrodes may be retrieved
        from the `Sile`.

        Parameters
        ==========
        elec_from: str
           the originating electrode
        elec_to: str
           the absorbing electrode (different from `elec_from`)
        avg: bool (True)
           whether the returned transmission is k-averaged
        """
        if elec_from == elec_to:
            raise ValueError(
                "Supplied elec_from and elec_to must not be the same.")

        return self._data_avg(elec_to + '.T', elec_from, avg=kavg)
    T = transmission

    def transmission_eig(self, elec_from, elec_to, avg=True):
        """ Return the transmission eigenvalues from `from` to `to`.

        The transmission eigenvalues between two electrodes may be retrieved
        from the `Sile`.

        Parameters
        ==========
        elec_from: str
           the originating electrode
        elec_to: str
           the absorbing electrode (different from `elec_from`)
        avg: bool (True)
           whether the returned eigenvalues are k-averaged
        """
        if elec_from == elec_to:
            raise ValueError(
                "Supplied elec_from and elec_to must not be the same.")

        return self._data_avg(elec_to + '.T.Eig', elec_from, avg=kavg)
    TEig = transmission_eig
    Teig = transmission_eig

    def transmission_bulk(self, elec, avg=True):
        """ Return the bulk transmission in the `elec` electrode

        Parameters
        ==========
        elec: str
           the bulk electrode
        avg: bool (True)
           whether the returned transmission is k-averaged
        """
        return self._data_avg('T', elec, avg=kavg)
    TBulk = transmission_bulk
    Tbulk = transmission_bulk

    def DOS(self, avg=True):
        """ Return the Green function DOS.

        Parameters
        ==========
        avg: bool (True)
           whether the returned DOS is k-averaged
        """
        return self._data_avg('DOS', avg=kavg)
    DOS_Gf = DOS

    def ADOS(self, elec, avg=True):
        """ Return the DOS of the spectral function from `elec`.

        Parameters
        ==========
        elec: str
           electrode originating spectral function
        avg: bool (True)
           whether the returned DOS is k-averaged
        """
        return self._data_avg('ADOS', elec, avg=kavg)
    DOS_A = ADOS

    def DOS_bulk(self, elec, avg=True):
        """ Return the bulk DOS of `elec`.

        Parameters
        ==========
        elec: str
           electrode where the bulk DOS is returned
        avg: bool (True)
           whether the returned DOS is k-averaged
        """
        return self._data_avg('DOS', elec, avg=kavg)
    BulkDOS = DOS_bulk

    def orbital_current(self, elec, E=None, avg=True):
        """ Return the orbital current originating from `elec`.

        This will return a sparse matrix (`scipy.sparse.csr_matrix`).
        The sparse matrix may be interacted with like a normal
        matrix although it enables extremely big matrices.

        Parameters
        ==========
        elec: str
           the electrode of originating electrons
        E: int (None)
           the energy index of the orbital current
           If `None` two objects will be returned, 1) the csr_matrix of the orbital currents , 2) all the currents (J), you may do:
            >>> J, mat = orbital_current(elec)
            >>> mat.data[:] = J[E,:]
           otherwise it will only return:
            >>> mat.data[:] = J[E,:]
           which is (far) less memory consuming.
        avg: bool (True)
           whether the orbital currents are k-averaged
        """

        # Get column indices
        col = np.array(self.variables['list_col'][:], np.int32) - 1
        # Create row-pointer
        tmp = np.cumsum(self.variables['n_col'][:])
        size = len(tmp)
        ptr = np.empty(size + 1, np.int32)
        mat_size = (size, size)
        ptr[0] = 0
        ptr[1:] = tmp[:]
        del tmp

        if E is None:
            # Return both the data and the corresponding
            # sparse matrix
            J = self._data_avg('J', elec, avg=avg)
            if len(J.shape) == 2:
                mat = csr_matrix((J[0, :], col, ptr), shape=mat_size)
            else:
                mat = csr_matrix((J[0, 0, :], col, ptr), shape=mat_size)
            return mat, J

        else:
            J = self._data_E('J', elec, avg, E)

        return csr_matrix((J, col, ptr), shape=mat_size)

    def bond_current(self, Jij, symmetry=True):
        """ Return the bond-current between atoms (sum of orbital currents)

        Parameters
        ==========
        Jij: scipy.sparse.csr_matrix
           the orbital currents as retrieved from `orbital_current`
        symmetry: bool (True)
           only return half of the bond currents (the upper triangle), otherwise return both
        """

        # We convert to atomic bond-currents
        J = lil_matrix((self.na_u, self.na_u), dtype=mat.dtype)

        # Create the iterator across the sparse pattern
        tmp = Jij.tocoo()
        it = np.nditer([self.o2a(tmp.row), self.o2a(tmp.col), tmp.data],
                       flags=['external_loop', 'buffered'],
                       op_flags=['readonly'])

        # Perform reduction
        if symmetry:
            for ja, ia, d in it:
                if ia <= ja:
                    continue

                J[ja, ia] += d
        else:
            for ja, ia, d in it:
                if ia == ja:
                    continue  # it is zero anyway

                J[ja, ia] += d

        # Delete iterator
        del it

        # Now we have the bond-currents
        # convert and sort
        mat = J.tocsr()
        # Rescale to correct magnitude
        J.data[:] *= .5
        mat.sort_indices()

        return mat


class PHtransSile(TBtransSile):
    """ PHtrans file object """
    _trans_type = 'PHT'
    pass
