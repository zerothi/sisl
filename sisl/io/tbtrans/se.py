from __future__ import print_function, division

import numpy as np
from numpy import in1d, argsort

# Import sile objects
from sisl._indices import indices
from sisl.utils import *
import sisl._array as _a
from ..sile import add_sile
from ._cdf import _devncSileTBtrans

# Import the geometry object
from sisl.unit.siesta import unit_convert


__all__ = ['tbtsencSileTBtrans', 'phtsencSilePHtrans']


Bohr2Ang = unit_convert('Bohr', 'Ang')
Ry2eV = unit_convert('Ry', 'eV')


class tbtsencSileTBtrans(_devncSileTBtrans):
    r""" TBtrans self-energy file object with downfolded self-energies to the device region

    The :math:`\Sigma` object contains all self-energies on the specified k- and energy grid projected
    into the device region.

    This is mainly an output file object from TBtrans and can be used as a post-processing utility for
    testing various things in Python.

    Note that *anything* returned from this object are the self-energies in eV.

    Examples
    --------
    >>> H = Hamiltonian(device)
    >>> se = tbtsencSileTBtrans(...)
    >>> # Return the self-energy for the left electrode (unsorted)
    >>> se_unsorted = se.self_energy('Left', 0.1, [0, 0, 0])
    >>> # Return the self-energy for the left electrode (sorted)
    >>> se_sorted = se.self_energy('Left', 0.1, [0, 0, 0], sort=True)
    >>> #
    >>> # Query the indices in the full Hamiltonian
    >>> pvt_unsorted = se.pivot('Left').reshape(-1, 1)
    >>> pvt_sorted = se.pivot('Left', sort=True).reshape(-1, 1)
    >>> # The following two lines are equivalent
    >>> Hfull[pvt_unsorted, pvt_unsorted.T] -= se_unsorted[:, :]
    >>> Hfull[pvt_sorted, pvt_sorted.T] -= se_sorted[:, :]
    >>> # Query the indices in the device Hamiltonian
    >>> dpvt_unsorted = se.pivot('Left', in_device=True).reshape(-1, 1)
    >>> dpvt_sorted = se.pivot('Left', in_device=True, sort=True).reshape(-1, 1)
    >>> # Following inserts are equivalent
    >>> Hdev[dpvt_unsorted, dpvt_unsorted.T] -= se_unsorted[:, :]
    >>> Hdev[dpvt_sorted, dpvt_sorted.T] -= se_sorted[:, :]
    """
    _trans_type = 'TBT'
    _E2eV = Ry2eV

    def _elec(self, elec):
        """ Converts a string or integer to the corresponding electrode name

        Parameters
        ----------
        elec : str or int
           if `str` it is the *exact* electrode name, if `int` it is the electrode
           index

        Returns
        -------
        str : the electrode name
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

    def eta(self, elec):
        """ The imaginary part used when calculating the self-energies in eV """
        try:
            return self._value('eta', self._elec(elec))[0] * self._E2eV
        except:
            return 0.

    def pivot(self, elec=None, in_device=False, sort=False):
        """ Return the pivoting indices for a specific electrode

        Parameters
        ----------
        elec : str or int
           the corresponding electrode to return the self-energy from
        in_device : bool, optional
           If ``True`` the pivoting table will be translated to the device region orbitals
        sort : bool, optional
           Whether the returned indices are sorted. Mostly useful if the self-energies are returned
           sorted as well.

        Examples
        --------
        >>> se = tbtsencSileTBtrans(...)
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
                pvt = np.sort(pvt)
            return pvt

        # Get electrode pivoting elements
        se_pvt = self._value('pivot', tree=self._elec(elec)) - 1
        if sort:
            # Sort pivoting indices
            # Since we know that pvt is also sorted, then
            # the resulting in_device would also return sorted
            # indices
            se_pvt = np.sort(se_pvt)

        if in_device:
            pvt = self._value('pivot') - 1
            if sort:
                pvt = np.sort(pvt)
            # translate to the device indices
            se_pvt = indices(pvt, se_pvt, 0)
        return se_pvt

    def a2p(self, atom, elec=None):
        """ Return the pivoting orbital indices (0-based) for the atoms, possibly on an electrode

        This is equivalent to:

        >>> p = self.o2p(self.geom.a2o(atom, True))

        Parameters
        ----------
        atom : array_like or int
           atomic indices (0-based)
        elec : str or int or None
           electrode to return pivoting indices of (if None it is the
           device pivoting indices).
        """
        orbs = self.geom.a2o(atom, True)
        return self.o2p(orbs, elec=elec)

    def o2p(self, orbital, elec=None):
        """ Return the pivoting indices (0-based) for the orbitals, possibly on an electrode

        Parameters
        ----------
        orbital : array_like or int
           orbital indices (0-based)
        elec : str or int or None
           electrode to return pivoting indices of (if None it is the
           device pivoting indices).
        """
        return in1d(self.pivot(elec=elec), orbital).nonzero()[0]

    def self_energy(self, elec, E, k=0, sort=False):
        """ Return the self-energy from the electrode `elec`

        Parameters
        ----------
        elec : str or int
           the corresponding electrode to return the self-energy from
        E : float or int
           energy to retrieve the self-energy at, if a floating point the closest
           energy value will be found and returned, if an integer it will correspond
           to the exact index
        k : array_like or int
           k-point to retrieve, if an integer it is the k-index in the file
        sort : bool, optional
           if ``True`` the returned self-energy will be sorted (equivalent to pivoting the self-energy)
        """
        tree = self._elec(elec)
        ik = self.kindex(k)
        iE = self.Eindex(E)

        re = self._variable('ReSelfEnergy', tree=tree)
        im = self._variable('ImSelfEnergy', tree=tree)

        SE = (re[ik, iE, :, :] + 1j * im[ik, iE, :, :])
        if sort:
            pvt = self.pivot(elec)
            idx = argsort(pvt).reshape(-1, 1)

            # pivot for sorted device region
            return SE[idx, idx.T] * self._E2eV

        return SE * self._E2eV

    def scattering_matrix(self, elec, E, k=0, sort=False):
        r""" Return the scattering matrix from the electrode `elec`

        The scattering matrix is calculated as:

        .. math::
            \Gamma(E) = i [\Sigma(E) - \Sigma^\dagger(E)]

        Parameters
        ----------
        elec : str or int
           the corresponding electrode to return the scattering matrix from
        E : float or int
           energy to retrieve the scattering matrix at, if a floating point the closest
           energy value will be found and returned, if an integer it will correspond
           to the exact index
        k : array_like or int
           k-point to retrieve, if an integer it is the k-index in the file
        sort : bool, optional
           if ``True`` the returned scattering matrix will be sorted (equivalent to pivoting the scattering matrix)
        """
        tree = self._elec(elec)
        ik = self.kindex(k)
        iE = self.Eindex(E)

        re = self._variable('ReSelfEnergy', tree=tree)[ik, iE, :, :]
        im = self._variable('ImSelfEnergy', tree=tree)[ik, iE, :, :]

        G = - (im + im.T) + 1j * (re - re.T)
        if sort:
            pvt = self.pivot(elec)
            idx = argsort(pvt)
            idx.shape = (-1, 1)

            # pivot for sorted device region
            return G[idx, idx.T] * self._E2eV

        return G * self._E2eV

    def self_energy_average(self, elec, E, sort=False):
        """ Return the k-averaged average self-energy from the electrode `elec`

        Parameters
        ----------
        elec : str or int
           the corresponding electrode to return the self-energy from
        E : float or int
           energy to retrieve the self-energy at, if a floating point the closest
           energy value will be found and returned, if an integer it will correspond
           to the exact index
        sort : bool, optional
           if ``True`` the returned self-energy will be sorted but not necessarily consecutive
           in the device region.
        """
        tree = self._elec(elec)
        iE = self.Eindex(E)

        re = self._variable('ReSelfEnergyMean', tree=tree)
        im = self._variable('ImSelfEnergyMean', tree=tree)

        SE = (re[ik, iE, :, :] + 1j * im[ik, iE, :, :])
        if sort:
            pvt = self.pivot(elec)
            idx = argsort(pvt)
            idx.shape = (-1, 1)

            # pivot for sorted device region
            return SE[idx, idx.T] * self._E2eV

        return SE * self._E2eV


add_sile('TBT.SE.nc', tbtsencSileTBtrans)
# Add spin-dependent files
add_sile('TBT_UP.SE.nc', tbtsencSileTBtrans)
add_sile('TBT_DN.SE.nc', tbtsencSileTBtrans)


class phtsencSilePHtrans(tbtsencSileTBtrans):
    """ PHtrans file object """
    _trans_type = 'PHT'
    _E2eV = Ry2eV ** 2

add_sile('PHT.SE.nc', phtsencSilePHtrans)
