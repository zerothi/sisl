# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""Phonon related functions and classes
=======================================

In sisl phonon calculations are relying on routines
specific for phonons. For instance density of states calculations from
phonon eigenvalues and other quantities.

This module implements the necessary tools required for calculating
DOS, PDOS, group-velocities and real-space displacements.

   DOS
   PDOS


Supporting classes
------------------

Certain classes aid in the usage of the above methods by implementing them
using automatic arguments.

   CoefficientPhonon
   ModePhonon
   ModeCPhonon
   EigenvaluePhonon
   EigenvectorPhonon
   EigenmodePhonon

"""

import numpy as np
from numpy import delete, fabs

import sisl._array as _a
from sisl import constant, units
from sisl._internal import set_module
from sisl.typing import DistributionType

from .distribution import get_distribution
from .electron import DOS as electron_DOS
from .electron import PDOS as electron_PDOS
from .state import Coefficient, State, StateC

__all__ = ["DOS", "PDOS"]
__all__ += ["CoefficientPhonon", "ModePhonon", "ModeCPhonon"]
__all__ += ["EigenvaluePhonon", "EigenvectorPhonon", "EigenmodePhonon"]


@set_module("sisl.physics.phonon")
def DOS(E, hw, distribution: DistributionType = "gaussian"):
    r"""Calculate the density of modes (DOS) for a set of energies, `E`, with a distribution function

    The :math:`\mathrm{DOS}(E)` is calculated as:

    .. math::
       \mathrm{DOS}(E) = \sum_i D(E-\hbar\omega_i) \approx\delta(E-\hbar\omega_i)

    where :math:`D(\Delta E)` is the distribution function used. Note that the distribution function
    used may be a user-defined function. Alternatively a distribution function may
    be retrieved from `~sisl.physics.distribution`.

    Parameters
    ----------
    E : array_like
       energies to calculate the DOS at
    hw : array_like
       phonon eigenvalues
    distribution :
       a function that accepts :math:`E` as argument and calculates the
       distribution function.

    See Also
    --------
    :ref:`physics.distribution` : a selected set of implemented distribution functions
    PDOS : projected DOS (same as this, but projected onto each direction)

    Returns
    -------
    numpy.ndarray
        DOS calculated at energies, has same length as `E`
    """
    return electron_DOS(E, hw, distribution)


@set_module("sisl.physics.phonon")
def PDOS(E, mode, hw, distribution: DistributionType = "gaussian"):
    r"""Calculate the projected density of modes (PDOS) onto each each atom and direction for a set of energies, `E`, with a distribution function

    The :math:`\mathrm{PDOS}(E)` is calculated as:

    .. math::
       \mathrm{PDOS}_\alpha(E) = \sum_i \epsilon^*_{i,\alpha} \epsilon_{i,\alpha} D(E-\hbar\omega_i)

    where :math:`D(\Delta E)` is the distribution function used. Note that the distribution function
    used may be a user-defined function. Alternatively a distribution function may
    be aquired from `~sisl.physics.distribution`.

    .. math::
       \mathrm{DOS}(E) = \sum_\alpha\mathrm{PDOS}_\alpha(E)

    Parameters
    ----------
    E : array_like
       energies to calculate the projected-DOS from
    mode : array_like
       eigenvectors
    hw : array_like
       eigenvalues
    distribution :
       a function that accepts :math:`E-\epsilon` as argument and calculates the
       distribution function.

    See Also
    --------
    :ref:`physics.distribution` : a selected set of implemented distribution functions
    DOS : total DOS (same as summing over atoms and directions)

    Returns
    -------
    numpy.ndarray
        projected DOS calculated at energies, has dimension ``(mode.shape[1], len(E))``.
    """
    return electron_PDOS(E, hw, mode, distribution=distribution)[0]


# dDk is in [Ang * eV ** 2]
# velocity units in Ang/ps
_velocity_const = 1 / constant.hbar("eV ps")


_displacement_const = (
    2 * units("Ry", "eV") * (constant.m_e / constant.m_p)
) ** 0.5 * units("Bohr", "Ang")


@set_module("sisl.physics.phonon")
class CoefficientPhonon(Coefficient):
    """Coefficients describing some physical quantity related to phonons"""

    __slots__ = []


@set_module("sisl.physics.phonon")
class ModePhonon(State):
    """A mode describing a physical quantity related to phonons"""

    __slots__ = []

    @property
    def mode(self):
        """Eigenmodes (states)"""
        return self.state


@set_module("sisl.physics.phonon")
class ModeCPhonon(StateC):
    """A mode describing a physical quantity related to phonons, with associated coefficients of the mode"""

    __slots__ = []

    @property
    def mode(self):
        """Eigenmodes (states)"""
        return self.state

    def velocity(self, *args, **kwargs):
        r"""Calculate velocity of the modes

        This routine calls `derivative` with appropriate arguments (1st order derivative)
        and returns the velocity for the modes.

        Note that the coefficients associated with the `ModeCPhonon` *must* correspond
        to the energies of the modes.

        See `derivative` for details and possible arguments. One cannot pass the ``order`` argument
        as that is fixed to ``1`` in this call.

        Notes
        -----
        The states and energies for the modes *may* have changed after calling this routine.
        This is because of the velocity un-folding for degenerate modes. I.e. calling
        `displacement` and/or `PDOS` after this method *may* change the result.

        See Also
        --------
        derivative : for details of the implementation
        """
        d = self.derivative(1, *args, **kwargs)
        axes = tuple(i for i in range(0, d.ndim, 2))
        c = np.expand_dims(self.c, axis=axes) * (2 / _velocity_const)
        return np.divide(d, c, where=(c != 0))


@set_module("sisl.physics.phonon")
class EigenvaluePhonon(CoefficientPhonon):
    """Eigenvalues of phonon modes, no eigenmodes retained

    This holds routines that enable the calculation of density of states.
    """

    __slots__ = []

    @property
    def hw(self):
        r"""Eigenmode values in units of :math:`\hbar \omega` [eV]"""
        return self.c

    def occupation(self, distribution: DistributionType = "bose_einstein"):
        """Calculate the occupations for the states according to a distribution function

        Parameters
        ----------
        distribution :
           distribution used to find occupations

        Returns
        -------
        numpy.ndarray
             ``len(self)`` with occupation values
        """
        if isinstance(distribution, str):
            distribution = get_distribution(distribution)
        return distribution(self.hw)

    def DOS(self, E, distribution: DistributionType = "gaussian"):
        r"""Calculate DOS for provided energies, `E`.

        This routine calls `~sisl.physics.phonon.DOS` with appropriate arguments
        and returns the DOS.

        See `~sisl.physics.phonon.DOS` for argument details.
        """
        return DOS(E, self.hw, distribution)


@set_module("sisl.physics.phonon")
class EigenvectorPhonon(ModePhonon):
    """Eigenvectors of phonon modes, no eigenvalues retained"""

    __slots__ = []


@set_module("sisl.physics.phonon")
class EigenmodePhonon(ModeCPhonon):
    """Eigenmodes of phonons with eigenvectors and eigenvalues.

    This holds routines that enable the calculation of (projected) density of states.
    """

    __slots__ = []

    @property
    def hw(self):
        r"""Eigenmode values in units of :math:`\hbar \omega` [eV]"""
        return self.c

    def occupation(self, distribution: DistributionType = "bose_einstein"):
        """Calculate the occupations for the states according to a distribution function

        Parameters
        ----------
        distribution :
           distribution used to find occupations

        Returns
        -------
        numpy.ndarray
             ``len(self)`` with occupation values
        """
        if isinstance(distribution, str):
            distribution = get_distribution(distribution)
        return distribution(self.hw)

    def DOS(self, E, distribution: DistributionType = "gaussian"):
        r"""Calculate DOS for provided energies, `E`.

        This routine calls `~sisl.physics.phonon.DOS` with appropriate arguments
        and returns the DOS.

        See `~sisl.physics.phonon.DOS` for argument details.
        """
        return DOS(E, self.hw, distribution)

    def PDOS(self, E, distribution: DistributionType = "gaussian"):
        r"""Calculate PDOS for provided energies, `E`.

        This routine calls `~sisl.physics.phonon.PDOS` with appropriate arguments
        and returns the PDOS.

        See `~sisl.physics.phonon.PDOS` for argument details.
        """
        return PDOS(E, self.mode, self.hw, distribution)

    def displacement(self, atol: float = 1e-9):
        r"""Calculate real-space displacements for a given mode (in units of the characteristic length)

        The displacements per mode may be written as:

        .. math::

            \mathbf{u}_{I\alpha} = \epsilon_{I\alpha}\sqrt{\frac{\hbar}{m_I \omega}}

        where :math:`I` is the atomic index.

        Even for negative frequencies the characteristic length is calculated for use of non-equilibrium
        modes.

        Parameters
        ----------
        atol :
            absolute tolerance for whether a phonon is 0 or not.
            Since the phonon energy is used in the calculation of the displacement vector
            we have to remove phonon modes with 0 energy.
            The displacements for phonon modes with an absolute energy below `atol` will
            be 0.

        Returns
        -------
        numpy.ndarray
            displacements per mode with final dimension ``(len(self), self.parent.na, 3)``, displacements are in Ang
        """
        # get indices for the zero modes
        idx = (np.fabs(self.c) <= atol).nonzero()[0]
        mode = self.mode
        U = mode.copy()
        U[idx, :] = 0.0

        # Now create the remaining displacements
        idx = delete(_a.arange(U.shape[0]), idx)

        # Generate displacement factor
        factor = _displacement_const / fabs(self.c[idx]).reshape(-1, 1) ** 0.5

        U.shape = (U.shape[0], -1, 3)
        U[idx] = (mode[idx, :] * factor).reshape(
            len(idx), -1, 3
        ) / self._geometry().mass.reshape(1, -1, 1) ** 0.5
        return U
