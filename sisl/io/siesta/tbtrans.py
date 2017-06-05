"""
Sile object for reading TBtrans binary files
"""
from __future__ import print_function, division

import warnings
import numpy as np
import itertools

# The sparse matrix for the orbital/bond currents
from scipy.sparse import csr_matrix, lil_matrix

# Check against integers
from numbers import Integral

# Import sile objects
from .sile import SileCDFSIESTA
from ..sile import *
from sisl.utils import *


# Import the geometry object
from sisl import Geometry, Atom, Atoms, SuperCell
from sisl.sparse import ispmatrix, ispmatrixd
from sisl._help import _str
from sisl.units.siesta import unit_convert

__all__ = ['tbtncSileSiesta', 'phtncSileSiesta']
__all__ += ['dHncSileSiesta']

Bohr2Ang = unit_convert('Bohr', 'Ang')
Ry2eV = unit_convert('Ry', 'eV')
Ry2K = unit_convert('Ry', 'K')
eV2Ry = unit_convert('eV', 'Ry')


class tbtncSileSiesta(SileCDFSIESTA):
    """ TBtrans file object 

    This ``SileCDF`` implements the TBtrans output `*.TBT.nc` sile which contains
    calculated quantities related to the NEGF code TBtrans.

    Although the TBtrans code is in fortran and the resulting NetCDF file variables
    are in fortran indexing (1-based), everything is returned as Python indexing (0-based)
    when scripting.

    This is vital when using this ``Sile``.

    Note that when using the command-line utility ``sdata`` the indexing is fortran based 
    because the data handlers are meant for _easy_ use.
    """
    _trans_type = 'TBT'
    _k_avg = False

    def write_tbtav(self, **kwargs):
        """ Write the TBT.AV.nc equivalent of this TBtrans output

        This will create/overwrite the file with the ending TBT.AV.nc and thus
        will not take notice of any older files.
        """
        from .tbtrans_av import tbtavncSileSiesta as TBTAV
        TBTAV(self._file.replace('.nc', '.AV.nc'), mode='w', access=0).write(tbtav=self)

    def _value_avg(self, name, tree=None, avg=False):
        """ Local method for obtaining the data from the SileCDF.

        This method checks how the file is access, i.e. whether
        data is stored in the object or it should be read consequtively.
        """
        if self._access > 0:
            if name in self._data:
                return self._data[name]

        v = self._variable(name, tree=tree)
        if self._k_avg:
            return v[:]

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
            # We assume avg is some kind of iterable
            data = v[avg[0], ...] * wkpt[avg[0]]
            for i in range(1, len(avg)):
                data += v[avg[i], ...] * wkpt[avg[i]]
            data.shape = orig_shape[1:]

        # Return data
        return data

    def _value_E(self, name, tree=None, avg=False, E=None):
        """ Local method for obtaining the data from the SileCDF using an E index.

        """
        if E is None:
            return self._value_avg(name, tree, avg)

        # Ensure that it is an index
        iE = self.Eindex(E)

        v = self._variable(name, tree=tree)
        if self._k_avg:
            return v[iE, ...]

        wkpt = self.wkpt

        # Perform normalization
        orig_shape = v.shape

        if isinstance(avg, bool):
            if avg:
                nk = len(wkpt)
                data = np.array(v[0, iE, ...]) * wkpt[0]
                for i in range(1, nk):
                    data += v[i, iE, ...] * wkpt[i]
                data.shape = orig_shape[2:]
            else:
                data = np.array(v[:, iE, ...])

        elif isinstance(avg, Integral):
            data = np.array(v[avg, iE, ...]) * wkpt[avg]
            data.shape = orig_shape[2:]

        else:
            # We assume avg is some kind of itterable
            data = v[avg[0], iE, ...] * wkpt[avg[0]]
            for i in avg[1:]:
                data += v[i, iE, ...] * wkpt[i]
            data.shape = orig_shape[2:]

        # Return data
        return data

    def _setup(self):
        """ Setup the special object for data containing """
        self._data = dict()

        if self._access > 0:

            # Fake double calls
            access = self._access
            self._access = 0

            # There are certain elements which should
            # be minimal on memory but allow for
            # fast access by the object.
            for d in ['cell', 'xa', 'lasto',
                      'a_dev', 'pivot', 'E']:
                self._data[d] = self._value(d)
            # tbtrans does not store the k-points and weights
            # if the Gamma-point is used.
            try:
                self._data['kpt'] = self._value('kpt')
            except:
                self._data['kpt'] = np.zeros([3], dtype=np.float64)
            try:
                self._data['wkpt'] = self._value('wkpt')
            except:
                self._data['wkpt'] = np.ones([1], dtype=np.float64)

            # Create the geometry in the data file
            self._data['_geom'] = self.read_geometry()

            # Reset the access pattern
            self._access = access

    def read_supercell(self):
        """ Returns ``SuperCell`` object from a .TBT.nc file """
        cell = np.array(np.copy(self.cell), dtype=np.float64)
        cell.shape = (3, 3)

        try:
            nsc = self._value('nsc')
        except:
            nsc = None

        sc = SuperCell(cell, nsc=nsc)
        try:
            sc.sc_off = self._value('isc_off')
        except:
            pass

        return sc

    def read_geometry(self, *args, **kwargs):
        """ Returns ``Geometry`` object from a .TBT.nc file """
        sc = self.read_supercell()

        xyz = np.array(np.copy(self.xa), dtype=np.float64)
        xyz.shape = (-1, 3)

        # Create list with correct number of orbitals
        lasto = np.array(np.copy(self.lasto) + 1, dtype=np.int32)
        nos = np.append([lasto[0]], np.diff(lasto))
        nos = np.array(nos, np.int32)

        if 'atom' in kwargs:
            # The user "knows" which atoms are present
            atms = kwargs['atom']
            # Check that all atoms have the correct number of orbitals.
            # Otherwise we will correct them
            for i in range(len(atms)):
                if atms[i].orbs != nos[i]:
                    atms[i] = Atom(Z=atms[i].Z, orbs=nos[i], tag=atms[i].tag)

        else:
            # Default to Hydrogen atom with nos[ia] orbitals
            # This may be counterintuitive but there is no storage of the
            # actual species
            atms = [Atom(Z='H', orbs=o) for o in nos]

        # Create and return geometry object
        geom = Geometry(xyz, atms, sc=sc)

        return geom

    def write_geometry(self, *args, **kwargs):
        """ This is not meant to be used """
        raise ValueError(self.__class__.__name__ + " can not write a geometry")

    # This class also contains all the important quantities elements of the
    # file.

    @property
    def geom(self):
        """ Returns the associated geometry from the TBT file """
        return self.read_geometry()

    @property
    def cell(self):
        """ Unit cell in file """
        return self._value('cell') * Bohr2Ang

    @property
    def na(self):
        """ Returns number of atoms in the cell """
        return len(self._dimension('na_u'))
    na_u = na

    @property
    def no(self):
        """ Returns number of orbitals in the cell """
        return len(self._dimension('no_u'))
    no_u = no

    @property
    def xa(self):
        """ Atomic coordinates in file """
        return self._value('xa') * Bohr2Ang
    xyz = xa

    # Device atoms and other quantities
    @property
    def na_d(self):
        """ Number of atoms in the device region """
        return len(self._dimension('na_d'))
    na_dev = na_d

    @property
    def a_d(self):
        """ Atomic indices (1-based) of device atoms """
        return self._value('a_dev') - 1
    a_dev = a_d

    @property
    def pivot(self):
        """ Pivot table of device orbitals to obtain input sorting """
        return self._value('pivot') - 1
    pvt = pivot

    def a2p(self, atom):
        """ Return the pivoting indices (0-based) for the atoms 

        Parameters
        ----------
        atom : ``array_like``, ``int``
           atomic indices (0-based)
        """
        orbs = self.geom.a2o(atom, True)
        return self.o2p(orbs)

    def o2p(self, orbital):
        """ Return the pivoting indices (0-based) for the orbitals

        Parameters
        ----------
        orbital : ``array_like``, ``int``
           orbital indices (0-based)
        """
        return np.where(np.in1d(self.pivot, orbital))[0]

    @property
    def lasto(self):
        """ Last orbital of corresponding atom """
        return self._value('lasto') - 1

    @property
    def no_d(self):
        """ Number of orbitals in the device region """
        return int(len(self.dimensions['no_d']))

    @property
    def kpt(self):
        """ Sampled k-points in file """
        return self._value('kpt')

    @property
    def wkpt(self):
        """ Weights of k-points in file """
        return self._value('wkpt')

    @property
    def nkpt(self):
        """ Number of k-points in file """
        return len(self.dimensions['nkpt'])

    @property
    def E(self):
        """ Sampled energy-points in file """
        return self._value('E') * Ry2eV

    def Eindex(self, E):
        """ Return the closest energy index corresponding to the energy `E`"""
        if isinstance(E, Integral):
            return E
        elif isinstance(E, _str):
            # This will always be converted to a float
            E = float(E)
        return np.abs(self.E - E).argmin()

    @property
    def ne(self):
        """ Number of energy-points in file """
        return len(self._dimension('ne'))
    nE = ne

    @property
    def elecs(self):
        """ List of electrodes """
        elecs = list(self.groups.keys())

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
        return self._value('mu', elec)[0] * Ry2eV
    mu = chemical_potential

    def electronic_temperature(self, elec):
        """ Return temperature of the electrode electronic distribution in Kelvin """
        return self._value('kT', elec)[0] * Ry2K

    def kT(self, elec):
        """ Return temperature of the electrode electronic distribution in eV """
        return self._value('kT', elec)[0] * Ry2eV

    def transmission(self, elec_from, elec_to, avg=True):
        """ Return the transmission from `from` to `to`.

        The transmission between two electrodes may be retrieved
        from the `Sile`.

        Parameters
        ----------
        elec_from: ``str``
           the originating electrode
        elec_to: ``str``
           the absorbing electrode (different from `elec_from`)
        avg: ``bool`` (`True`)
           whether the returned transmission is k-averaged
        """
        if elec_from == elec_to:
            raise ValueError("Supplied elec_from and elec_to must not be the same.")

        return self._value_avg(elec_to + '.T', elec_from, avg=avg)
    T = transmission

    def transmission_eig(self, elec_from, elec_to, avg=True):
        """ Return the transmission eigenvalues from `from` to `to`.

        The transmission eigenvalues between two electrodes may be retrieved
        from the `Sile`.

        Parameters
        ----------
        elec_from: ``str``
           the originating electrode
        elec_to: ``str``
           the absorbing electrode (different from `elec_from`)
        avg: ``bool`` (`True`)
           whether the returned eigenvalues are k-averaged
        """
        if elec_from == elec_to:
            raise ValueError(
                "Supplied elec_from and elec_to must not be the same.")

        return self._value_avg(elec_to + '.T.Eig', elec_from, avg=avg)
    Teig = transmission_eig

    def transmission_bulk(self, elec, avg=True):
        """ Return the bulk transmission in the `elec` electrode

        Parameters
        ----------
        elec: ``str``
           the bulk electrode
        avg: ``bool`` (`True`)
           whether the returned transmission is k-averaged
        """
        return self._value_avg('T', elec, avg=avg)
    Tbulk = transmission_bulk

    def _DOS_atom_sum(self, DOS, atom):
        if atom is None:
            return DOS

        # Now we should sum per atom an retain the order...
        if isinstance(atom, Integral):
            return np.sum(DOS[..., self.a2p(atom)], axis=-1)

        # Create return array
        shp = list(DOS.shape)
        shp[-1] = len(atom)
        nDOS = np.zeros(shp, np.float64)

        # Sum for new return stuff
        for i, a in enumerate(atom):
            pvt = self.a2p(a)
            if len(pvt) == 0:
                continue
            nDOS[..., i] = np.sum(DOS[..., pvt], axis=-1)

        return nDOS

    def DOS(self, E=None, avg=True, atom=None):
        """ Return the Green function DOS (1/eV).

        Parameters
        ----------
        E : ``int`` (`None`)
           optionally only return the DOS of atoms at a given energy point
        avg: ``bool`` (`True`)
           whether the returned DOS is k-averaged
        atom : ``array_like``, ``int`` (_all_)
           only return for a given set of atoms (Python based indices)
        """
        return self._DOS_atom_sum(self._value_E('DOS', avg=avg, E=E), atom) * eV2Ry
    DOS_Gf = DOS

    def ADOS(self, elec, E=None, avg=True, atom=None):
        """ Return the DOS of the spectral function from `elec` (1/eV).

        Parameters
        ----------
        elec: ``str``
           electrode originating spectral function
        E : ``int`` (`None`)
           optionally only return the DOS of atoms at a given energy point
        avg: ``bool`` (`True`)
           whether the returned DOS is k-averaged
        atom : ``array_like``, ``int`` (_all_)
           only return for a given set of atoms (Python based indices)
        """
        return self._DOS_atom_sum(self._value_E('ADOS', elec, avg=avg, E=E), atom) * eV2Ry
    DOS_A = ADOS

    def BDOS(self, elec, E=None, avg=True):
        """ Return the bulk DOS of `elec` (1/eV).

        Parameters
        ----------
        elec: ``str``
           electrode where the bulk DOS is returned
        E : ``int`` (`None`)
           optionally only return the DOS of atoms at a given energy point
        avg: ``bool`` (`True`)
           whether the returned DOS is k-averaged
        """
        return self._value_E('DOS', elec, avg=avg, E=E) * eV2Ry
    DOS_bulk = BDOS
    BulkDOS = BDOS

    def _E_T_sorted(self, avg=True):
        """ Internal routine for returning energies and transmission in a sorted array """
        E = self.E
        idx_sort = np.argsort(E)
        # Get transmission
        T = self.transmission(elec_from, elec_to, avg)
        return E[idx_sort], T[idx_sort]

    def current(self, elec_from, elec_to, avg=True):
        """ Return the current from `from` to `to` using the weights in the file. """
        # Get energies
        E, T = self._E_T_sorted(avg)

        # Get chemical potentials
        mu_f = self.chemical_potential(elec_from)
        mu_t = self.chemical_potential(elec_to)

        # Calculate the current
        return NotImplemented

    def orbital_current(self, elec, E, avg=True, isc=None):
        """ Return the orbital current originating from `elec`.

        This will return a sparse matrix (``scipy.sparse.csr_matrix``).
        The sparse matrix may be interacted with like a normal
        matrix although it enables extremely big matrices.

        Parameters
        ----------
        elec: str
           the electrode of originating electrons
        E: float or int
           the energy or the energy index of the orbital current. If an integer
           is passed it is the index, otherwise the index corresponding to
           `Eindex(E)` is used.
        avg: bool, optional
           whether the orbital current returned is k-averaged, default to True
        isc: array_like (`[0, 0, 0]`)
           the returned bond currents from the unit-cell (`[0, 0, 0]`) to
           the given supercell, the default is only orbital currents *in* the unitcell.
        """
        # Get the geometry for obtaining the sparsity pattern.
        geom = self.geom

        # These are the row-pointers...
        rptr = np.cumsum(self._value('n_col'))

        # Get column indices
        col = self._value('list_col') - 1

        # Default matrix size
        mat_size = (geom.no, geom.no_s)

        # Figure out the super-cell indices that are requested
        # First we figure out the indices, then
        # we build the array of allowed columns
        if isc is None:
            isc = [0, 0, 0]

        if isc[0] is None and isc[1] is None and isc[2] is None:
            all_col = None

        else:
            # The user has requested specific supercells
            # Here we create a list of supercell interactions.

            nsc = geom.nsc[:]
            # Shorten to the unit-cell if there are no more
            for i in [0, 1, 2]:
                if nsc[i] == 1:
                    isc[i] = 0
                if not isc[i] is None:
                    nsc[i] = 1
            # Small function for creating the supercells allowed

            def ret_range(val, req):
                i = val // 2
                if req is None:
                    return range(-i, i+1)
                return [req]
            x = ret_range(nsc[0], isc[0])
            y = ret_range(nsc[1], isc[1])
            z = ret_range(nsc[2], isc[2])
            offsets = []
            for ix, iy, iz in itertools.product(x, y, z):
                offsets.append(geom.sc_index([ix, iy, iz]))

            # Make a shrinking logical array for selecting a subset of the
            # orbital currents...
            all_col = []
            for i in offsets:
                all_col.extend(range(i * geom.no, (i+1) * geom.no))
            all_col = np.array(all_col, np.int32)
            # Create a logical array for sub-indexing
            all_col = np.in1d(col, all_col)
            col = col[all_col]

            # recreate row-pointer (we have to fix it)
            tmp = np.empty([geom.no], np.int32)
            nsum = np.sum
            tmp[0] = nsum(all_col[0:rptr[0]])
            for i in range(1, len(tmp)):
                tmp[i] = nsum(all_col[rptr[i-1]:rptr[i]])
            rptr = np.cumsum(tmp)
            del tmp

        ptr = np.empty([geom.no + 1], np.int32)
        ptr[0] = 0
        ptr[1:] = rptr[:]

        if all_col is None:
            J = self._value_E('J', elec, avg, E)
        else:
            J = self._value_E('J', elec, avg, E)[..., all_col]

        return csr_matrix((J, col, ptr), shape=mat_size)

    def bond_current_from_orbital(self, Jij, sum='+', uc=None):
        """ Return the bond-current between atoms (sum of orbital currents) by passing the orbital
        currents.

        Parameters
        ----------
        Jij : ``scipy.sparse.*_matrix``
           the orbital currents as retrieved from `orbital_current`
        sum : ``str`` (`'+'`)
           this value may be "+"/"-"/"all"
           If "+" is supplied only the positive orbital currents are used,
           for "-", only the negative orbital currents are used,
           else return both.
        uc : `bool` (`Jij.shape[0] == Jij.shape[1]`)
           whether the returned bond-currents are only in the unit-cell.
           If `True` this will return a sparse matrix of `.shape = (self.na, self.na)`,
           else, it will return a sparse matrix of `.shape = (self.na, self.na * self.n_s)`.
           One may figure out the connections via `Geometry.sc_index`.
        """
        geom = self.geom
        o2a = geom.o2a

        if uc is None:
            # Default UC shape
            uc = Jij.shape[0] == Jij.shape[1]

        # We convert to atomic bond-currents
        if uc:
            na = geom.na
            J = lil_matrix((na, na), dtype=Jij.dtype)

            map_row = o2a

            def map_col(c):
                return o2a(c) % na

        else:
            J = lil_matrix((geom.na, geom.na * geom.n_s), dtype=Jij.dtype)

            map_row = o2a
            map_col = o2a

        # Perform reduction
        if "+" in sum:
            for ia, ja, b in ispmatrixd(Jij, map_row, map_col):
                if b > 0:
                    J[ia, ja] += b

        elif "-" in sum:
            for ia, ja, b in ispmatrixd(Jij, map_row, map_col):
                if b < 0:
                    J[ia, ja] -= b

        else:
            for ia, ja, b in ispmatrixd(Jij, map_row, map_col):
                J[ia, ja] += b

        # Rescale to correct magnitude
        J *= 0.5

        # Now we have the bond-currents convert and sort
        mat = J.tocsr()
        mat.sort_indices()

        return mat

    def bond_current(self, elec, E, avg=True, isc=None, sum='+', uc=False):
        """ Return the bond-current between atoms (sum of orbital currents)

        Parameters
        ----------
        elec : ``str``
           the electrode of originating electrons
        E : float or int
           A `float` for energy in eV, `int` for explicit energy index
           Unlike `orbital_current` this may not be `None` as the down-scaling of the
           orbital currents may not be equivalent for all energy points.
        avg : ``bool`` (`True`)
           whether the bond current returned is k-averaged
        isc : ``array_like`` (`[0, 0, 0]`)
           the returned bond currents from the unit-cell (`[0, 0, 0]`) to
           the given supercell. If `[None, None, None]` is passed all
           bond currents are returned.
        sum : ``str`` (`'+'`)
           this value may be "+"/"-"/"all"
           If "+" is supplied only the positive orbital currents are used,
           for "-", only the negative orbital currents are used,
           else return the sum of both.
        uc : ``bool`` (`False`)
           whether the returned bond-currents are only in the unit-cell.
           If `True` this will return a sparse matrix of `.shape = (self.na, self.na)`,
           else, it will return a sparse matrix of `.shape = (self.na, self.na * self.n_s)`.
           One may figure out the connections via `Geometry.sc_index`.
        """

        # First we retrieve the orbital currents
        Jorb = self.orbital_current(elec, E, avg, isc)

        return self.bond_current_from_orbital(Jorb, sum=sum, uc=uc)

    def atom_current_from_orbital(self, Jij, activity=True):
        r""" Return the atom-current of atoms.

        This takes a sparse matrix with size `self.geom.no, self.geom.no_s` as argument
        with the associated orbital currents.

        Please note that this returns the atomic current by folding all 
        orbital currents into the unit-cell.

        Parameters
        ----------
        Jij: ``scipy.sparse.*_matrix``
           the orbital currents as retrieved from `orbital_current`
        activity: ``bool`` (`True`)
           whether the activity current is returned.
           This is defined using these two equations:

           .. math::
              J_I^{|a|} &=\frac{1}{2} \sum_J \big| \sum_{\nu\in I}\sum_{\mu\in J} J_{\nu\mu} \big|
              J_I^{|o|} &=\frac{1}{2} \sum_J \sum_{\nu\in I}\sum_{\mu\in J} \big| J_{\nu\mu} \big|

           If `activity = False` it returns

           .. math::
              J_I^{|a|}

           and if `activity = True` it returns

           .. math::
              J_I^{\mathcal A} = \sqrt{ J_I^{|a|} J_I^{|o|} }

        """

        # Number of atoms
        na = self.na_u

        # atomic currents
        Ja = np.zeros([na], np.float64)

        # Create local orbital pointers
        o2a = self.geom.o2a
        sc2uc = self.geom.sc2uc
        firsto = self.geom.firsto
        lasto = self.geom.lasto[:] + 1

        # Faster function calls
        nabs = np.abs
        nsum = np.sum

        # Calculate individual bond-currents between atoms
        if activity:
            Jo = np.zeros([na], np.float64)
            for ia, ja in ispmatrix(Jij, o2a, o2a):

                # we also include ia == ja (that should be zero anyway)
                # Get unit-cell atom
                a = sc2uc(ja)
                # Get supercell index offset
                i = (ja // na) * na
                t = Jij[firsto[ia]:lasto[ia], firsto[a]:i+lasto[a]].data

                # Calculate both the orbital and atomic normalized current
                Jo[ia] += nsum(nabs(t))
                Ja[ia] += nabs(nsum(t))

            # If it is the activity current, we return the geometric mean...
            Ja = np.sqrt(Ja * Jo)
        else:
            for ia, ja in ispmatrix(Jij, o2a, o2a):
                a = sc2uc(ja)
                i = (ja // na) * na
                t = Jij[firsto[ia]:lasto[ia], i+firsto[a]:i+lasto[a]].data
                Ja[ia] += nabs(nsum(t))
        del t

        # Scale correctly
        Ja *= 0.5

        return Ja

    def atom_current(self, elec, E, avg=True, activity=True):
        r""" Return the atom-current of atoms. 

        This should *not* be confused with the bond-currents.

        Parameters
        ----------
        elec: ``str``
           the electrode of originating electrons
        E: float or int
           the energy or energy index of the atom current.
        avg: ``bool`` (`True`)
           whether the atom current returned is k-averaged
        activity: ``bool`` (`True`)
           whether the activity current is returned.
           This is defined using these two equations:

           .. math::
              J_I^{|a|} &=\frac{1}{2} \sum_J \big| \sum_{\nu\in I}\sum_{\mu\in J} J_{\nu\mu} \big|
              J_I^{|o|} &=\frac{1}{2} \sum_J \sum_{\nu\in I}\sum_{\mu\in J} \big| J_{\nu\mu} \big|

           If `activity = False` it returns

           .. math::
              J_I^{|a|}

           and if `activity = True` it returns

           .. math::
              J_I^{\mathcal A} = \sqrt{ J_I^{|a|} J_I^{|o|} }

        """
        Jorb = self.orbital_current(elec, E, avg, isc=[None, None, None])

        return self.atom_current_from_orbital(Jorb, activity=activity)

    def vector_current_from_orbital(self, Jij):
        """ Return the atom-current with vector components of atoms.

        This takes a sparse matrix with size `self.geom.no, self.geom.no_s` as argument
        with the associated orbital currents.

        Parameters
        ----------
        Jij: ``scipy.sparse.*_matrix``
           the orbital currents as retrieved from `orbital_current`
        """
        geom = self.geom

        na = geom.na
        # vector currents
        Ja = np.zeros([na, 3], np.float64)

        # Create local orbital look-up
        o2a = geom.o2a
        sc2uc = geom.sc2uc
        firsto = geom.firsto
        lasto = geom.lasto[:] + 1
        xyz = geom.xyz

        # Calculate individual bond-currents between atoms
        for ia, ja in ispmatrix(Jij, o2a, o2a):
            if ia == ja:
                # If we are on the same atom there is no direction
                # and hence we would get a division by zero, so skip
                continue

            # Get unit-cell index
            a = sc2uc(ja)
            # Get supercell index offset
            i = (ja // na) * na

            t = Jij[firsto[ia]:lasto[ia], i+firsto[a]:i+lasto[a]].data.sum()

            # calculate the vector between atom `ia` and `ja`
            v = geom.axyz(ja) - xyz[ia, :]

            # multiply by normalized vector
            Ja[ia, :] += t * v / (v[0]**2 + v[1]**2 + v[2]**2)**.5

        # Scale correctly
        Ja *= 0.5

        return Ja

    def vector_current(self, elec, E, avg=True):
        """ Return the atom-current with vector components of atoms.

        Parameters
        ----------
        elec: ``str``
           the electrode of originating electrons
        E: float or int
           the energy or energy index of the vector current.
           Unlike `orbital_current` this may not be `None` as the down-scaling of the
           orbital currents may not be equivalent for all energy points.
        avg: ``bool`` (`True`)
           whether the vector current returned is k-averaged
        """
        Jorb = self.orbital_current(elec, E, avg, isc=[None, None, None])

        return self.vector_current_from_orbital(Jorb)

    def read_data(self, *args, **kwargs):
        """ Read specific type of data. 

        This is a generic routine for reading different parts of the data-file.

        Parameters
        ----------
        geom: ``bool``
           return the geometry
        atom_current: ``bool``
           return the atomic current flowing through an atom (the *activity* current)
        vector_current: ``bool``
           return the orbital currents as vectors
        """
        val = []
        for kw in kwargs:

            if kw in ['geom', 'geometry']:
                if kwargs[kw]:
                    val.append(self.read_geometry())

            if kw == 'atom_current':
                if kwargs[kw]:
                    # TODO we need some way of handling arguments.
                    val.append(self.atom_current(*args))

            if kw == 'vector_current':
                if kwargs[kw]:
                    # TODO we need some way of handling arguments.
                    val.append(self.vector_current(*args))

        if len(val) == 0:
            val = None
        elif len(val) == 1:
            val = val[0]
        return val

    @dec_default_AP("Extract data from a TBT.nc file")
    def ArgumentParser(self, p=None, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """

        # We limit the import to occur here
        import argparse

        d = {
            "_tbt": self,
            "_geometry": self.geom,
            "_data_description": [],
            "_data_header": [],
            "_data": [],
            "_Orng": None,
            "_Oscale": 1. / len(self.pivot),
            "_Erng": None,
            "_krng": True,
        }
        namespace = default_namespace(**d)

        def dec_ensure_E(func):
            """ This decorater ensures that E is the first element in the _data container """

            def assign_E(self, *args, **kwargs):
                ns = args[1]
                if len(ns._data) == 0:
                    # We immediately extract the energies
                    if ns._Erng is None:
                        ns._data.append(ns._tbt.E[:])
                    else:
                        ns._data.append(ns._tbt.E[ns._Erng])
                    ns._data_header.append('Energy[eV]')
                return func(self, *args, **kwargs)
            return assign_E

        # Correct the geometry species information
        class Geometry(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):

                old_g = ns._geometry.copy()

                # Now read the file to read the geometry from
                g = get_sile(value).read_geometry()

                # Make sure g has the same # of orbitals
                atoms = [None] * len(old_g)
                for a, idx in g.atom:
                    for i in idx:
                        atoms[i] = a.copy(orbs=old_g.atom[i].orbs)
                g._atom = Atoms(atoms)

                ns._geometry = g
        p.add_argument('--geometry', '-G',
                       action=Geometry,
                       help=('Update the geometry of the output file, this enables one to set the species correctly,'
                             ' note this only affects output-files where species are important'))

        # Energy grabs
        class ERange(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                Emap = strmap(float, value)
                # Convert to actual indices
                E = []
                for begin, end in Emap:
                    E.append(range(ns._tbt.Eindex(begin), ns._tbt.Eindex(end)+1))
                ns._Erng = np.array(E, np.int32).flatten()
        p.add_argument('--energy', '-E',
                       action=ERange,
                       help='Denote the sub-section of energies that are extracted: "-1:0,1:2" [eV]')

        # k-range
        class kRange(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                ns._krng = lstranges(strmap(int, value))
        if not self._k_avg:
            p.add_argument('--kpoint', '-k',
                           action=kRange,
                           help='Denote the sub-section of k-indices that are extracted.')

        # Try and add the atomic specification
        class AtomRange(argparse.Action):

            @dec_collect_and_run_action
            def __call__(self, parser, ns, value, option_string=None):
                value = value.replace(' ', '')
                # Immediately convert to proper indices
                geom = ns._geometry

                # Sadly many shell interpreters does not
                # allow simple [] because they are expansion tokens
                # in the shell.
                # We bypass this by allowing *, [, {
                # * will "only" fail if files are named accordingly, else
                # it will be passed as-is.
                #       {    [    *
                sep = ['c', 'b', '*']
                failed = True
                while failed and len(sep) > 0:
                    try:
                        ranges = lstranges(strmap(int, value, sep.pop()))
                        failed = False
                    except:
                        pass
                if failed:
                    print(value)
                    raise ValueError("Could not parse the atomic/orbital ranges")

                # we have only a subset of the orbitals
                orbs = []
                no = 0
                asarray = np.asarray
                for atoms in ranges:
                    if isinstance(atoms, list):
                        # this will be
                        #  atoms[0] == atom
                        #  atoms[1] == list of orbitals on the atom

                        # Get atoms and orbitals
                        ob = geom.a2o(atoms[0] - 1, True)
                        # We normalize for the total number of orbitals
                        # on the requested atoms.
                        # In this way the user can compare directly the DOS
                        # for same atoms with different sets of orbitals and the
                        # total will add up.
                        no += len(ob)
                        ob = ob[asarray(atoms[1], np.int32) - 1]
                    else:
                        ob = geom.a2o(atoms - 1, True)
                        no += len(ob)
                    orbs.append(ob)

                # Add one to make the c-index equivalent to the f-index
                orbs = np.concatenate(orbs).flatten()
                pivot = np.where(np.in1d(ns._tbt.pivot, orbs))[0]

                if len(orbs) != len(pivot):
                    print('Device atoms:')
                    tmp = ns._tbt.a_dev[:] + 1
                    tmp.sort()
                    print(tmp[:])
                    print('Input atoms:')
                    print(value)
                    raise ValueError('Atomic/Orbital requests are not fully included in the device region.')

                ns._Ovalue = value
                ns._Orng = pivot
                # Correct the scale to the correct number of orbitals
                ns._Oscale = 1. / no
                #print('Updating Orng and Oscale: ',ns._Orng, ns._Oscale)

        p.add_argument('--atom', '-a', type=str, action=AtomRange,
                       help='Limit orbital resolved quantities to a sub-set of atoms/orbitals: "1-2[3,4]" will yield the 1st and 2nd atom and their 3rd and fourth orbital. Multiple comma-separated specifications are allowed. Note that some shells does not allow [] as text-input (due to expansion), {, [ or * are allowed orbital delimiters. ')

        class DataT(argparse.Action):

            @dec_collect_action
            @dec_ensure_E
            def __call__(self, parser, ns, values, option_string=None):
                e1 = values[0]
                if e1 not in ns._tbt.elecs:
                    raise ValueError('Electrode: "'+e1+'" cannot be found in the specified file.')
                e2 = values[1]
                if e2 not in ns._tbt.elecs:
                    if e2.strip() == '.':
                        for e2 in ns._tbt.elecs:
                            if e2 != e1:
                                try: # catches if T isn't calculated
                                    self(parser, ns, [e1, e2], option_string)
                                except:
                                    pass
                        return
                    raise ValueError('Electrode: "'+e2+'" cannot be found in the specified file.')

                # Grab the information
                data = ns._tbt.transmission(e1, e2, avg=ns._krng)[ns._Erng]
                data.shape = (-1,)
                ns._data.append(data)
                ns._data_header.append('T:{}-{}[G]'.format(e1, e2))
                ns._data_description.append('Column {} is transmission from {} to {}'.format(len(ns._data), e1, e2))
        p.add_argument('-T', '--transmission', nargs=2, metavar=('ELEC1', 'ELEC2'),
                       action=DataT,
                       help='Store the transmission between two electrodes.')

        class DataBT(argparse.Action):

            @dec_collect_action
            @dec_ensure_E
            def __call__(self, parser, ns, value, option_string=None):
                e = value[0]
                if e not in ns._tbt.elecs:
                    if e.strip() == '.':
                        for e in ns._tbt.elecs:
                            try: # catches if B isn't calculated
                                self(parser, ns, [e], option_string)
                            except:
                                pass
                        return
                    raise ValueError('Electrode: "'+e+'" cannot be found in the specified file.')

                # Grab the information
                data = ns._tbt.transmission_bulk(e, avg=ns._krng)[ns._Erng]
                data.shape = (-1,)
                ns._data.append(data)
                ns._data_header.append('BT:{}[G]'.format(e))
                ns._data_description.append('Column {} is bulk-transmission'.format(len(ns._data)))
        p.add_argument('-BT', '--transmission-bulk', nargs=1, metavar='ELEC',
                       action=DataBT,
                       help='Store the bulk transmission of an electrode.')

        class DataDOS(argparse.Action):

            @dec_collect_action
            @dec_ensure_E
            def __call__(self, parser, ns, value, option_string=None):
                if not value is None:
                    # we are storing the spectral DOS
                    e = value
                    if e not in ns._tbt.elecs:
                        if e.strip() == '.':
                            for e in ns._tbt.elecs:
                                try: # catches if DOS isn't calculated
                                    self(parser, ns, [e], option_string)
                                except:
                                    pass
                            return
                        raise ValueError('Electrode: "'+e+'" cannot be found in the specified file.')
                    # Grab the information
                    data = ns._tbt.ADOS(e, avg=ns._krng)
                    ns._data_header.append('ADOS:{}[1/eV]'.format(e))
                else:
                    data = ns._tbt.DOS(avg=ns._krng)
                    ns._data_header.append('DOS[1/eV]')

                # Grab out the orbital ranges
                if not ns._Orng is None:
                    orig_shape = data.shape
                    data = data[..., ns._Orng]
                # Select the energies, even if _Erng is None, this will work!
                data = np.sum(data[ns._Erng, ...], axis=-1).flatten()
                ns._data.append(data * ns._Oscale)
                if ns._Orng is None:
                    ns._data_description.append('Column {} is sum of all device atoms+orbitals with normalization 1/'.format(len(ns._data), int(1/ns._Oscale)))
                else:
                    ns._data_description.append('Column {} is atoms[orbs] {} with normalization 1/{}'.format(len(ns._data), ns._Ovalue, int(1/ns._Oscale)))

        p.add_argument('--dos', '-D', nargs='?', metavar='ELEC',
                       action=DataDOS, default=None,
                       help="""Store the DOS. If no electrode is specified, it is Green function, else it is the spectral function.""")
        p.add_argument('--ados', '-AD', metavar='ELEC',
                       action=DataDOS, default=None,
                       help="""Store the spectral DOS, same as --dos but requires an electrode-argument.""")

        class DataDOSBulk(argparse.Action):

            @dec_collect_action
            @dec_ensure_E
            def __call__(self, parser, ns, value, option_string=None):

                # we are storing the Bulk DOS
                e = value[0]
                if e not in ns._tbt.elecs:
                    if e.strip() == '.':
                        for e in ns._tbt.elecs:
                            try: # catches if BDOS isn't calculated
                                self(parser, ns, [e], option_string)
                            except:
                                pass
                        return
                    raise ValueError('Electrode: "'+e+'" cannot be found in the specified file.')
                # Grab the information
                data = ns._tbt.BDOS(e, avg=ns._krng)
                ns._data_header.append('BDOS:{}[1/eV]'.format(e))
                # Select the energies, even if _Erng is None, this will work!
                no = data.shape[-1]
                data = np.mean(data[ns._Erng, ...], axis=-1).flatten()
                ns._data.append(data)
                ns._data_description.append('Column {} is sum of all electrode[{}] atoms+orbitals with normalization 1/{}'.format(len(ns._data), e, no))
        p.add_argument('--bulk-dos', '-BD', nargs=1, metavar='ELEC',
                       action=DataDOSBulk, default=None,
                       help="""Store the bulk DOS of an electrode.""")

        class DataTEig(argparse.Action):

            @dec_collect_action
            @dec_ensure_E
            def __call__(self, parser, ns, values, option_string=None):
                e1 = values[0]
                if e1 not in ns._tbt.elecs:
                    raise ValueError('Electrode: "'+e1+'" cannot be found in the specified file.')
                e2 = values[1]
                if e2 not in ns._tbt.elecs:
                    if e2.strip() == '.':
                        for e2 in ns._tbt.elecs:
                            if e1 != e2:
                                try: # catches if T-eig isn't calculated
                                    self(parser, ns, [e1, e2], option_string)
                                except:
                                    pass
                        return
                    raise ValueError('Electrode: "'+e2+'" cannot be found in the specified file.')

                # Grab the information
                data = ns._tbt.transmission_eig(e1, e2, avg=ns._krng)
                # The shape is: k, E, neig
                neig = data.shape[-1]
                for eig in range(neig):
                    ns._data.append(data[ns._Erng, eig])
                    ns._data_header.append('Teig({}):{}-{}[G]'.format(eig+1, e1, e2))
                    ns._data_description.append('Column {} is transmission eigenvalues from electrode {} to {}'.format(len(ns._data), e1, e2))
        p.add_argument('--transmission-eig', '-Teig', nargs=2, metavar=('ELEC1', 'ELEC2'),
                       action=DataTEig,
                       help='Store the transmission eigenvalues between two electrodes.')

        class Info(argparse.Action):
            """ Action to print information contained in the TBT.nc file, helpful before performing actions """

            def __call__(self, parser, ns, value, option_string=None):
                # First short-hand the file
                tbt = ns._tbt

                def truefalse(bol, string, fdf=None):
                    if bol:
                        print("  + " + string + ": true")
                    elif fdf is None:
                        print("  - " + string + ": false")
                    else:
                        print("  - " + string + ": false\t\t["+', '.join(fdf) + ']')

                # Retrieve the device atoms
                dev_rng = list2range(tbt.a_dev + 1)
                print("Device information:")
                if tbt._k_avg:
                    print("  - all data is k-averaged")
                else:
                    # Print out some more information related to the
                    # k-point sampling.
                    # However, we still do not know whether TRS is
                    # applied.
                    kpt = tbt.kpt
                    nA = len(np.unique(kpt[:, 0]))
                    nB = len(np.unique(kpt[:, 1]))
                    nC = len(np.unique(kpt[:, 2]))
                    print(("  - number of kpoints: {} <- "
                           "[ A = {} , B = {} , C = {} ] (time-reversal unknown)").format(tbt.nkpt, nA, nB, nC))
                print("  - energy range:")
                E = tbt.E
                Em, EM = np.amin(E), np.amax(E)
                dE = np.diff(E)
                dEm, dEM = np.amin(dE) * 1000, np.amax(dE) * 1000 # convert to meV
                if (dEM - dEm) < 1e-3: # 0.001 meV
                    print("     {:.5f} -- {:.5f} eV  [{:.3f} meV]".format(Em, EM, dEm))
                else:
                    print("     {:.5f} -- {:.5f} eV  [{:.3f} -- {:.3f} meV]".format(Em, EM, dEm, dEM))
                print("  - atoms with DOS (fortran indices):")
                print("     " + dev_rng)
                truefalse('DOS' in tbt.variables, "DOS Green function", ['TBT.DOS.Gf'])
                print()
                print("Electrodes (*=DOS/ADOS/orbital-current not possible):")
                for elec in tbt.elecs:
                    if elec in tbt.groups:
                        print("   " + elec)
                    else:
                        print("  *" + elec)
                # Print out information for each electrode
                for elec in tbt.groups:
                    print() # new-line
                    print("Electrode: " + elec)
                    gelec = tbt.groups[elec]
                    print("  - chemical potential: {:.4f} eV".format(tbt.mu(elec)))
                    print("  - electronic temperature: {:.2f} K".format(tbt.electronic_temperature(elec)))
                    truefalse('DOS' in gelec.variables, "DOS bulk", ['TBT.DOS.Elecs'])
                    truefalse('ADOS' in gelec.variables, "DOS spectral", ['TBT.DOS.A'])
                    truefalse('J' in gelec.variables, "Orbital-current", ['TBT.DOS.A', 'TBT.Current.Orb'])
                    truefalse('T' in gelec.variables, "transmission bulk", ['TBT.T.Bulk'])
                    truefalse(elec + '.T' in gelec.variables, "transmission out", ['TBT.T.Out'])
                    truefalse(elec + '.C' in gelec.variables, "transmission out correction", ['TBT.T.Out'])
                    truefalse(elec + '.C.Eig' in gelec.variables, "transmission out correction (eigen)", ['TBT.T.Out', 'TBT.T.Eig'])
                    for elec2 in tbt.elecs:
                        # Skip it self, checked above in .T and .C
                        if elec2 == elec: continue
                        truefalse(elec2 + '.T' in gelec.variables, "transmission -> " + elec2)
                        truefalse(elec2 + '.T.Eig' in gelec.variables, "transmission (eigen) -> " + elec2, ['TBT.T.Eig'])

        p.add_argument('--info', '-i', action=Info, nargs=0,
                       help='Print out what information is contained in the TBT.nc file.')

        class Out(argparse.Action):

            @dec_run_actions
            def __call__(self, parser, ns, value, option_string=None):

                out = value[0]

                from sisl.io import TableSile
                try:
                    # We figure out if the user wants to write
                    # to a geometry
                    obj = get_sile(out, mode='w')
                    if hasattr(obj, 'write_geometry'):
                        obj.write_geometry(ns._geometry)
                        return
                    raise NotImplementedError
                except:
                    pass

                if len(ns._data) == 0:
                    # do nothing if data has not been collected
                    print("No data has been collected in the arguments, nothing will be written, have you forgotten arguments?")
                    return

                TableSile(out, mode='w').write(np.array(ns._data),
                                               comment=ns._data_description, header=ns._data_header)
                # Clean all data
                ns._data_description = []
                ns._data_header = []
                ns._data = []
                # These are expert options
                ns._Ovalue = ''
                ns._Orng = None
                ns._Oscale = 1. / len(ns._tbt.pivot)
                ns._Erng = None
                ns._krng = True
        p.add_argument('--out', '-o', nargs=1, action=Out,
                       help='Store the currently collected information (at its current invocation) to the out file.')

        class AVOut(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                ns._tbt.write_tbtav()
        p.add_argument('--tbt-av', action=AVOut, nargs=0,
                       help='Create "{0}" with the k-averaged quantities of this file.'.format(self.file.replace('TBT.nc', 'TBT.AV.nc')))

        class Plot(argparse.Action):

            @dec_run_actions
            def __call__(self, parser, ns, value, option_string=None):

                if len(ns._data) == 0:
                    # do nothing if data has not been collected
                    return

                from matplotlib import pyplot as plt

                for i in range(1, len(ns._data)):
                    plt.plot(ns._data[0], ns._data[i], label=ns._data_header[i])

                plt.legend(loc=8, ncol=3, bbox_to_anchor=(0.5, 1.0))
                plt.show()

                # Clean all data
                ns._data_description = []
                ns._data_header = []
                ns._data = []
                # These are expert options
                ns._Ovalue = ''
                ns._Orng = None
                ns._Oscale = 1. / len(ns._tbt.pivot)
                ns._Erng = None
                ns._krng = True
        p.add_argument('--plot', '-p', nargs=0, action=Plot,
                       help='Plot the currently collected information (at its current invocation).')

        return p, namespace


add_sile('TBT.nc', tbtncSileSiesta)
# Add spin-dependent files
add_sile('TBT_DN.nc', tbtncSileSiesta)
add_sile('TBT_UP.nc', tbtncSileSiesta)


class phtncSileSiesta(tbtncSileSiesta):
    """ PHtrans file object """
    _trans_type = 'PHT'
    pass

add_sile('PHT.nc', phtncSileSiesta)


# The deltaH nc file
class dHncSileSiesta(SileCDFSIESTA):
    """ TBtrans delta-H file object """

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

        bs = self._crt_grp(self, 'BASIS')
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
                        'of the alreay stored dimensions.')
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

    def _add_lvl(self, ilvl):
        """
        Simply adds and returns a group if it does not
        exist it will be created
        """
        slvl = 'LEVEL-'+str(ilvl)
        if slvl in self.groups:
            lvl = self._crt_grp(self, slvl)
        else:
            lvl = self._crt_grp(self, slvl)
            if ilvl in [2, 4]:
                self._crt_dim(lvl, 'nkpt', None)
                v = self._crt_var(lvl, 'kpt', 'f8', ('nkpt', 'xyz'),
                                  attr = {'info': 'k-points for dH values',
                                          'unit': 'b**-1'})
            if ilvl in [3, 4]:
                self._crt_dim(lvl, 'ne', None)
                v = self._crt_var(lvl, 'E', 'f8', ('ne',),
                                  attr = {'info': 'Energy points for dH values',
                                          'unit': 'Ry'})

        return lvl

    def write_hamiltonian(self, ham, **kwargs):
        """ Writes Hamiltonian model to file

        Parameters
        ----------
        ham : `Hamiltonian` model
           the model to be saved in the NC file
        spin : int, 0
           the spin-index of the Hamiltonian object that is stored.
        """
        # Ensure finalizations
        ham.finalize()

        # Ensure that the geometry is written
        self.write_geometry(ham.geom)

        self._crt_dim(self, 'spin', ham._spin)

        # Determine the type of dH we are storing...
        k = kwargs.get('k', None)
        E = kwargs.get('E', None)

        if (k is None) and (E is None):
            ilvl = 1
        elif (k is not None) and (E is None):
            ilvl = 2
        elif (k is None) and (E is not None):
            ilvl = 3
        elif (k is not None) and (E is not None):
            ilvl = 4
        else:
            print(k, E)
            raise ValueError("This is wrongly implemented!!!")

        lvl = self._add_lvl(ilvl)

        # Append the sparsity pattern
        # Create basis group
        if 'n_col' in lvl.variables:
            if len(lvl.dimensions['nnzs']) != ham.nnz:
                raise ValueError("The sparsity pattern stored in dH *MUST* be equivalent for "
                                 "all dH entries [nnz].")
            if np.any(lvl.variables['n_col'][:] != ham._data.ncol[:]):
                raise ValueError("The sparsity pattern stored in dH *MUST* be equivalent for "
                                 "all dH entries [n_col].")
            if np.any(lvl.variables['list_col'][:] != ham._data.col[:]+1):
                raise ValueError("The sparsity pattern stored in dH *MUST* be equivalent for "
                                 "all dH entries [list_col].")
        else:
            self._crt_dim(lvl, 'nnzs', ham._data.col.shape[0])
            v = self._crt_var(lvl, 'n_col', 'i4', ('no_u',))
            v.info = "Number of non-zero elements per row"
            v[:] = ham._data.ncol[:]
            v = self._crt_var(lvl, 'list_col', 'i4', ('nnzs',),
                              chunksizes=(len(ham._data.col),), **self._cmp_args)
            v.info = "Supercell column indices in the sparse format"
            v[:] = ham._data.col[:] + 1  # correct for fortran indices
            v = self._crt_var(lvl, 'isc_off', 'i4', ('n_s', 'xyz'))
            v.info = "Index of supercell coordinates"
            v[:] = ham.geom.sc.sc_off[:, :]

        warn_E = True
        if ilvl in [3, 4]:
            Es = np.array(lvl.variables['E'][:]) * eV2Ry

            iE = 0
            if len(Es) > 0:
                iE = np.argmin(np.abs(Es - E))
                if abs(Es[iE] - E) > 0.0001:
                    # accuracy of 0.1 meV

                    # create a new entry
                    iE = len(Es)
                    lvl.variables['E'][iE] = E * Ry2eV
                    warn_E = False
            else:
                warn_E = False

        warn_k = True
        if ilvl in [2, 4]:
            kpt = np.array(lvl.variables['kpt'][:])

            ik = 0
            if len(kpt) > 0:
                ik = np.argmin(np.sum(np.abs(kpt - k[None, :]), axis=1))
                if np.allclose(kpt[ik, :], k, atol=0.0001):
                    # accuracy of 0.0001

                    # create a new entry
                    ik = len(kpt)
                    lvl.variables['kpt'][ik, :] = k
                    warn_k = False
            else:
                warn_k = False

        if ilvl == 4 and warn_k and warn_E and False:
            # As soon as we have put the second k-point and the first energy
            # point, this warning will proceed...
            # I.e. even though the variable has not been set, it will WARN
            # Hence we out-comment this for now...
            warnings.warn('Overwriting k-point {0} and energy point {1} correction.'.format(ik, iE), UserWarning)
        elif ilvl == 3 and warn_E:
            warnings.warn('Overwriting energy point {0} correction.'.format(iE), UserWarning)
        elif ilvl == 2 and warn_k:
            warnings.warn('Overwriting k-point {0} correction.'.format(ik), UserWarning)

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
        csize[-1] = ham.nnz

        if ham._data._D.dtype.kind == 'c':
            v1 = self._crt_var(lvl, 'RedH', 'f8', dim,
                               chunksizes=csize,
                               attr = {'info': "Real part of dH",
                                       'unit': "Ry"}, **self._cmp_args)
            for i in range(ham.spin):
                sl[-2] = i
                v1[sl] = ham._data._D[:, i].real * eV2Ry ** ham._E_order

            v2 = self._crt_var(lvl, 'ImdH', 'f8', dim,
                               chunksizes=csize,
                               attr = {'info': "Imaginary part of dH",
                                       'unit': "Ry"}, **self._cmp_args)
            for i in range(ham.spin):
                sl[-2] = i
                v2[sl] = ham._data._D[:, i].imag * eV2Ry ** ham._E_order

        else:
            v = self._crt_var(lvl, 'dH', 'f8', dim,
                              chunksizes=csize,
                              attr = {'info': "dH",
                                      'unit': "Ry"},  **self._cmp_args)
            for i in range(ham.spin):
                sl[-2] = i
                v[sl] = ham._data._D[:, i] * eV2Ry ** ham._E_order


add_sile('dH.nc', dHncSileSiesta)
