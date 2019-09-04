from __future__ import print_function, division

from numbers import Integral

import numpy as np
from numpy import in1d
from numpy import sort as npsort

# Import sile objects
from ..sile import SileWarning
from .sile import SileCDFTBtrans
from sisl.messages import warn, info
from sisl.utils import *
import sisl._array as _a
from sisl._indices import indices

# Import the geometry object
from sisl import Geometry, Atom, SuperCell
from sisl._help import _str
from sisl.unit.siesta import unit_convert

__all__ = ['_ncSileTBtrans', '_devncSileTBtrans']


Bohr2Ang = unit_convert('Bohr', 'Ang')
Ry2eV = unit_convert('Ry', 'eV')
Ry2K = unit_convert('Ry', 'K')
eV2Ry = unit_convert('eV', 'Ry')


class _ncSileTBtrans(SileCDFTBtrans):
    r""" Common TBtrans NetCDF file object due to a lot of the files having common entries

    This enables easy read of the Geometry and SuperCells etc.
    """

    def _setup(self, *args, **kwargs):
        """ Setup the special object for data containing """
        self._data = dict()

        if self._access > 0:

            # Fake double calls
            access = self._access
            self._access = 0

            # There are certain elements which should
            # be minimal on memory but allow for
            # fast access by the object.
            for d in ['cell', 'xa', 'lasto', 'E']:
                self._data[d] = self._value(d)
            # tbtrans does not store the k-points and weights
            # if the Gamma-point is used.
            try:
                self._data['kpt'] = self._value('kpt')
            except:
                self._data['kpt'] = _a.zerosd([3])
            try:
                self._data['wkpt'] = self._value('wkpt')
            except:
                self._data['wkpt'] = _a.onesd([1])

            # Create the geometry in the data file
            self._data['_geom'] = self.read_geometry()

            # Reset the access pattern
            self._access = access

    def read_supercell(self):
        """ Returns `SuperCell` object from this file """
        cell = _a.arrayd(np.copy(self.cell))
        cell.shape = (3, 3)

        nsc = self._value('nsc')
        sc = SuperCell(cell, nsc=nsc)
        sc.sc_off = self._value('isc_off')
        return sc

    def read_geometry(self, *args, **kwargs):
        """ Returns `Geometry` object from this file """
        sc = self.read_supercell()

        xyz = _a.arrayd(np.copy(self.xa))
        xyz.shape = (-1, 3)

        # Create list with correct number of orbitals
        lasto = _a.arrayi(np.copy(self.lasto) + 1)
        nos = np.append([lasto[0]], np.diff(lasto))
        nos = _a.arrayi(nos)

        if 'atom' in kwargs:
            # The user "knows" which atoms are present
            atms = kwargs['atom']
            # Check that all atoms have the correct number of orbitals.
            # Otherwise we will correct them
            for i in range(len(atms)):
                if atms[i].no != nos[i]:
                    atms[i] = Atom(atms[i].Z, [-1] *nos[i], tag=atms[i].tag)

        else:
            # Default to Hydrogen atom with nos[ia] orbitals
            # This may be counterintuitive but there is no storage of the
            # actual species
            atms = [Atom('H', [-1] * o) for o in nos]

        # Create and return geometry object
        geom = Geometry(xyz, atms, sc=sc)

        return geom

    # This class also contains all the important quantities elements of the
    # file.

    @property
    def geometry(self):
        """ The associated geometry from this file """
        return self.read_geometry()
    geom = geometry

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
    def xyz(self):
        """ Atomic coordinates in file """
        return self._value('xa') * Bohr2Ang
    xa = xyz

    @property
    def lasto(self):
        """ Last orbital of corresponding atom """
        return self._value('lasto') - 1

    @property
    def k(self):
        """ Sampled k-points in file """
        return self._value('kpt')
    kpt = k

    @property
    def wk(self):
        """ Weights of k-points in file """
        return self._value('wkpt')
    wkpt = wk

    @property
    def nk(self):
        """ Number of k-points in file """
        return len(self.dimensions['nkpt'])
    nkpt = nk

    @property
    def E(self):
        """ Sampled energy-points in file """
        return self._value('E') * Ry2eV

    @property
    def ne(self):
        """ Number of energy-points in file """
        return len(self._dimension('ne'))
    nE = ne

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
        idxE = np.abs(self.E - E).argmin()
        ret_E = self.E[idxE]
        if abs(ret_E - E) > 5e-3:
            warn(self.__class__.__name__ + " requesting energy " +
                 "{0:.5f} eV, found {1:.5f} eV as the closest energy!".format(E, ret_E))
        elif abs(ret_E - E) > 1e-3:
            info(self.__class__.__name__ + " requesting energy " +
                 "{0:.5f} eV, found {1:.5f} eV as the closest energy!".format(E, ret_E))
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
        ik = np.sum(np.abs(self.k - _a.asarrayd(k)[None, :]), axis=1).argmin()
        ret_k = self.k[ik, :]
        if not np.allclose(ret_k, k, atol=0.0001):
            warn(SileWarning(self.__class__.__name__ + " requesting k-point " +
                             "[{0:.3f}, {1:.3f}, {2:.3f}]".format(*k) +
                             " found " +
                             "[{0:.3f}, {1:.3f}, {2:.3f}]".format(*ret_k)))
        return ik


class _devncSileTBtrans(_ncSileTBtrans):
    r""" Common TBtrans NetCDF file object due to a lot of the files having common entries

    This one also enables device region atoms and pivoting tables.
    """

    def _setup(self, *args, **kwargs):
        """ Setup the special object for data containing """
        super(_devncSileTBtrans, self)._setup(*args, **kwargs)

        if self._access > 0:

            # Fake double calls
            access = self._access
            self._access = 0

            # There are certain elements which should
            # be minimal on memory but allow for
            # fast access by the object.
            for d in ['a_dev', 'pivot']:
                self._data[d] = self._value(d)

            # Reset the access pattern
            self._access = access

    def read_geometry(self, *args, **kwargs):
        g = super(_devncSileTBtrans, self).read_geometry(*args, **kwargs)
        try:
            g['Buffer'] = self.a_buf[:]
        except:
            # Then no buffer atoms
            pass
        g['Device'] = self.a_dev[:]
        try:
            for elec in self.elecs:
                g[elec] = self._value('a', [elec]) - 1
                g[elec + '+'] = self._value('a_down', [elec]) - 1
        except:
            pass
        return g

    @property
    def na_b(self):
        """ Number of atoms in the buffer region """
        return len(self._dimension('na_b'))
    na_buffer = na_b

    @property
    def a_buf(self):
        """ Atomic indices (0-based) of device atoms """
        return self._value('a_buf') - 1

    # Device atoms and other quantities
    @property
    def na_d(self):
        """ Number of atoms in the device region """
        return len(self._dimension('na_d'))
    na_dev = na_d

    @property
    def a_dev(self):
        """ Atomic indices (0-based) of device atoms (sorted) """
        return self._value('a_dev') - 1

    def a_elec(self, elec):
        """ Electrode atomic indices for the full geometry (sorted)

        Parameters
        ----------
        elec : str or int
           electrode to retrieve indices for
        """
        return self._value('a', self._elec(elec)) - 1

    def a_down(self, elec, bulk=False):
        """ Down-folding atomic indices for a given electrode

        Parameters
        ----------
        elec : str or int
           electrode to retrieve indices for
        bulk : bool, optional
           whether the returned indices are *only* in the pristine electrode,
           or the down-folding region (electrode + downfolding region, not in device)
        """
        if bulk:
            return self.a_elec(elec)
        return  self._value('a_down', self._elec(elec)) - 1

    @property
    def o_dev(self):
        """ Orbital indices (0-based) of device orbitals (sorted)

        See Also
        --------
        pivot : retrieve the device orbitals, non-sorted
        """
        return self.pivot(sort=True)

    @property
    def no_d(self):
        """ Number of orbitals in the device region """
        return len(self.dimensions['no_d'])

    def _elec(self, elec):
        """ Converts a string or integer to the corresponding electrode name

        Parameters
        ----------
        elec : str or int
           if `str` it is the *exact* electrode name, if `int` it is the electrode
           index

        Returns
        -------
        str
            the electrode name
        """
        try:
            elec = int(elec)
            return self.elecs[elec]
        except:
            return elec

    @property
    def elecs(self):
        """ List of electrodes """
        return list(self.groups.keys())

    def chemical_potential(self, elec):
        """ Return the chemical potential associated with the electrode `elec` """
        return self._value('mu', self._elec(elec))[0] * Ry2eV
    mu = chemical_potential

    def eta(self, elec=None):
        """ The imaginary part used when calculating the self-energies in eV (or for the device

        Parameters
        ----------
        elec : str, int, optional
           electrode to extract the eta value from. If not specified (or None) the device
           region eta will be returned.
        """
        try:
            return self._value('eta', self._elec(elec))[0] * self._E2eV
        except:
            return 0. # unknown!

    def electron_temperature(self, elec):
        """ Electron bath temperature [Kelvin]

        Parameters
        ----------
        elec : str or int
           electrode to extract the temperature from

        See Also
        --------
        kT: bath temperature in [eV]
        """
        return self._value('kT', self._elec(elec))[0] * Ry2K

    def kT(self, elec):
        """ Electron bath temperature [eV]

        Parameters
        ----------
        elec : str or int
           electrode to extract the temperature from

        See Also
        --------
        electron_temperature: bath temperature in [K]
        """
        return self._value('kT', self._elec(elec))[0] * Ry2eV

    def bloch(self, elec):
        """ Bloch-expansion coefficients for an electrode

        Parameters
        ----------
        elec : str or int
           bloch expansions of electrode
        """
        return self._value('bloch', self._elec(elec))

    def n_btd(self, elec=None):
        """ Number of blocks in the BTD partioning

        Parameters
        ----------
        elec : str or int, optional
           if None the number of blocks in the device region BTD matrix. Else
           the number of BTD blocks in the electrode down-folding.
        """
        return len(self._dimension('n_btd', self._elec(elec)))

    def btd(self, elec=None):
        """ Block-sizes for the BTD method in the device/electrode region

        Parameters
        ----------
        elec : str or int, optional
           the BTD block sizes for the device (if none), otherwise the downfolding
           BTD block sizes for the electrode
        """
        return self._value('btd', self._elec(elec))

    def no_down(self, elec):
        """ Number of orbitals in the downfolding region (plus device downfolded region)

        Parameters
        ----------
        elec : str or int
           Number of downfolding orbitals for electrode `elec`
        """
        return len(self._dimension('no_down', self._elec(elec)))

    def pivot_down(self, elec):
        """ Pivoting orbitals for the downfolding region of a given electrode

        Parameters
        ----------
        elec : str or int
           the corresponding electrode to get the pivoting indices for
        """
        return self._value('pivot_down', self._elec(elec)) - 1

    def pivot(self, elec=None, in_device=False, sort=False):
        """ Return the pivoting indices for a specific electrode (in the device region) or the device

        Parameters
        ----------
        elec : str or int
           the corresponding electrode to return the pivoting indices from
        in_device : bool, optional
           If ``True`` the pivoting table will be translated to the device region orbitals.
           If `sort` is also true, this would correspond to the orbitals directly translated
           to the geometry ``self.geometry.sub(self.a_dev)``.
        sort : bool, optional
           Whether the returned indices are sorted. Mostly useful if you want to handle
           the device in a non-pivoted order.

        Examples
        --------
        >>> se = tbtncSileTBtrans(...)
        >>> se.pivot()
        [3, 4, 6, 5, 2]
        >>> se.pivot(sort=True)
        [2, 3, 4, 5, 6]
        >>> se.pivot(0)
        [2, 3]
        >>> se.pivot(0, in_device=True)
        [4, 0]
        >>> se.pivot(0, in_device=True, sort=True)
        [0, 1]
        >>> se.pivot(0, sort=True)
        [2, 3]

        See Also
        --------
        pivot_down : for the pivot table for electrodes down-folding regions
        """
        if elec is None:
            if in_device and sort:
                return _a.arangei(self.no_d)
            pvt = self._value('pivot') - 1
            if in_device:
                # Count number of elements that we need to subtract from each orbital
                subn = _a.onesi(self.no)
                subn[pvt] = 0
                pvt -= _a.cumsumi(subn)[pvt]
            elif sort:
                pvt = npsort(pvt)
            return pvt

        # Get electrode pivoting elements
        se_pvt = self._value('pivot', tree=self._elec(elec)) - 1
        if sort:
            # Sort pivoting indices
            # Since we know that pvt is also sorted, then
            # the resulting in_device would also return sorted
            # indices
            se_pvt = npsort(se_pvt)

        if in_device:
            pvt = self._value('pivot') - 1
            if sort:
                pvt = npsort(pvt)
            # translate to the device indices
            se_pvt = indices(pvt, se_pvt, 0)
        return se_pvt

    def a2p(self, atom, elec=None):
        """ Return the pivoting orbital indices (0-based) for the atoms, possibly on an electrode

        This is equivalent to:

        >>> p = self.o2p(self.geom.a2o(atom, True))

        Will warn if an atom requested is not in the device list of atoms.

        Parameters
        ----------
        atom : array_like or int
           atomic indices (0-based)
        elec : str or int or None, optional
           electrode to return pivoting indices of (if None it is the
           device pivoting indices).
        """
        return self.o2p(self.geom.a2o(atom, True))

    def o2p(self, orbital, elec=None):
        """ Return the pivoting indices (0-based) for the orbitals, possibly on an electrode

        Will warn if an orbital requested is not in the device list of orbitals.

        Parameters
        ----------
        orbital : array_like or int
           orbital indices (0-based)
        elec : str or int or None, optional
           electrode to return pivoting indices of (if None it is the
           device pivoting indices).
        """
        # We need asarray, otherwise taking len of an int will fail
        orbital = np.asarray(orbital).ravel()
        porb = in1d(self.pivot(elec), orbital).nonzero()[0]
        d = len(orbital) - len(porb)
        if d != 0:
            warn('{}.o2p requesting an orbital outside the device region, '
                 '{} orbitals will be removed from the returned list'.format(self.__class__.__name__, d))
        return porb
