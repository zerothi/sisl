from __future__ import print_function, division

import numpy as np
from numpy import add, conj

from sisl._help import dtype_complex_to_real, _range as range
from .distributions import distribution as dist_func
from .state import Coefficient, State, StateC


def DOS(E, eig, distribution='gaussian'):
    r""" Calculate the density of states (DOS) for a set of energies, `E`, with a distribution function

    The :math:`\mathrm{DOS}(E)` is calculated as:
    .. math::
       \mathrm{DOS}(E) = \sum_i D(E-\epsilon_i) \approx\delta(E-\epsilon_i)

    where :math:`D(\Delta E)` is the distribution function used. Note that the distribution function
    used may be a user-defined function. Alternatively a distribution function may
    be retrieved from `sisl.physics.distribution`.

    Parameters
    ----------
    E : array_like
       energies to calculate the DOS at
    eig : array_like
       eigenvalues
    distribution : func or str, optional
       a function that accepts :math:`E-\epsilon` as argument and calculates the
       distribution function.

    See Also
    --------
    sisl.physics.distribution : a selected set of implemented distribution functions
    PDOS : projected DOS (same as this, but projected onto each orbital)
    spin_moment: spin moment of eigenvectors

    Returns
    -------
    numpy.ndarray : DOS calculated at energies, has same length as `E`
    """
    if isinstance(distribution, str):
        distribution = dist_func(distribution)

    DOS = distribution(E - eig[0])
    for i in range(1, len(self)):
        DOS += distribution(E - eig[i])
    return DOS


def PDOS(E, eig, eig_v, S=None, distribution='gaussian', spin=None):
    r""" Calculate the projected density of states (PDOS) for a set of energies, `E`, with a distribution function

    The :math:`\mathrm{PDOS}(E)` is calculated as:
    .. math::
       \mathrm{PDOS}_\nu(E) = \sum_i \psi^*_{i,\nu} [\mathbf S | \psi_{i}\rangle]_\nu D(E-\epsilon_i)

    where :math:`D(\Delta E)` is the distribution function used. Note that the distribution function
    used may be a user-defined function. Alternatively a distribution function may
    be aquired from `sisl.physics.distribution`.

    In case of an orthogonal basis set :math:`\mathbf S` is equal to the identity matrix.
    Note that `DOS` is the sum of the orbital projected DOS:

    .. math::
       \mathrm{DOS}(E) = \sum_\nu\mathrm{PDOS}_\nu(E)

    For non-colinear calculations (this includes spin-orbit calculations) the PDOS is additionally
    separated into 4 components (in this order):

    - Total projected DOS
    - Projected spin magnetic moment along :math:`x` direction
    - Projected spin magnetic moment along :math:`y` direction
    - Projected spin magnetic moment along :math:`z` direction

    These are calculated using the Pauli matrices :math:`\sigma_x`, :math:`\sigma_y` and :math:`\sigma_z`:

    .. math::
       \mathrm{PDOS}_\nu^\sigma(E) &= \sum_i \psi^*_{i,\nu} \sigma_z \sigma_z [\mathbf S | \psi_{i}\rangle]_\nu D(E-\epsilon_i)
       \\
       \mathrm{PDOS}_\nu^x(E) &= \sum_i \psi^*_{i,\nu} \sigma_x [\mathbf S | \psi_{i}\rangle]_\nu D(E-\epsilon_i)
       \\
       \mathrm{PDOS}_\nu^y(E) &= \sum_i \psi^*_{i,\nu} \sigma_y [\mathbf S | \psi_{i}\rangle]_\nu D(E-\epsilon_i)
       \\
       \mathrm{PDOS}_\nu^z(E) &= \sum_i \psi^*_{i,\nu} \sigma_z [\mathbf S | \psi_{i}\rangle]_\nu D(E-\epsilon_i)

    Note that the total PDOS may be calculated using :math:`\sigma_i\sigma_i` where :math:`i` may be either of :math:`x`,
    :math:`y` or :math:`z`.

    Parameters
    ----------
    E : array_like
       energies to calculate the projected-DOS from
    eig : array_like
       eigenvalues
    eig_v : array_like
       eigenvectors
    S : array_like, optional
       overlap matrix used in the :math:`\rangle\psi^*|\mathbf S|\psi\langle` calculation. If `None` the identity
       matrix is assumed. For non-colinear calculations this matrix may be halve the size of ``len(eig_v[0, :])`` to
       trigger the non-colinear calculation of PDOS.
    distribution : func or str, optional
       a function that accepts :math:`E-\epsilon` as argument and calculates the
       distribution function.
    spin : str or Spin, optional
       the spin configuration. This is generally only needed when the eigenvectors correspond to a non-colinear
       calculation.

    See Also
    --------
    sisl.physics.distribution : a selected set of implemented distribution functions
    DOS : total DOS (same as summing over orbitals)
    spin_moment: spin moment of eigenvectors

    Returns
    -------
    numpy.ndarray
        projected DOS calculated at energies, has dimension ``(eig_v.shape[1], len(E))``.
        For non-colinear calculations it will be ``(4, eig_v.shape[1] // 2, len(E))``, ordered as
        indicated in the above list.
    """
    if isinstance(distribution, str):
        distribution = dist_func(distribution)

    # Figure out whether we are dealing with a non-colinear calculation
    if S is None:
        class _S(object):
            @property
            def shape(self):
                return eig_v.shape
            def dot(self, v):
                return v
        S = _S()

    if spin is None:
        if S.shape[1] == eig_v.shape[1] // 2:
            spin = Spin('nc')
            S = S[::2, ::2]
        else:
            spin = Spin()

    # check for non-colinear (or SO)
    if spin.kind > Spin.POLARIZED:
        # Non colinear eigenvectors
        if S.shape[1] == eig_v.shape[1]:
            # Since we are going to reshape the eigen-vectors
            # to more easily get the mixed states, we can reduce the overlap matrix
            S = S[::2, ::2]

        # Initialize
        PDOS = np.empty([4, eig_v.shape[0], len(E)], dtype=dtype_complex_to_real(eig_v.dtype))

        d = distribution(E - eig[0]).reshape(1, -1)
        v = S.dot(eig_v[0].reshape(-1, 2))
        D = (conj(eig_v[0]) * v.ravel()).reshape(-1, 2) # diagonal PDOS
        PDOS[0, :, :] = D.sum(1).reshape(-1, 1) * d # total DOS
        PDOS[3, :, :] = (D[:, 0] - D[:, 1]).reshape(-1, 1) * d # z-dos
        D = (conj(eig_v[0, 1::2]) * 2 * v[:, 0]).reshape(-1, 1) # psi_down * psi_up * 2
        PDOS[1, :, :] = D.real * d # x-dos
        PDOS[2, :, :] = D.imag * d # y-dos
        for i in range(1, len(eig)):
            d = distribution(E - eig[i]).reshape(1, -1)
            v = S.dot(eig_v[i].reshape(-1, 2))
            D = (conj(eig_v[i]) * v.ravel()).reshape(-1, 2)
            PDOS[0, :, :] += D.sum(1).reshape(-1, 1) * d
            PDOS[3, :, :] += (D[:, 0] - D[:, 1]).reshape(-1, 1) * d
            D = (conj(eig_v[i, 1::2]) * 2 * v[:, 0]).reshape(-1, 1)
            PDOS[1, :, :] += D.real * d
            PDOS[2, :, :] += D.imag * d

    else:
        PDOS = (conj(eig_v[0]) * S.dot(eig_v[0])).real.reshape(-1, 1) \
               * distribution(E - eig[0]).reshape(1, -1)
        for i in range(1, len(eig)):
            PDOS[:, :] += (conj(eig_v[i]) * S.dot(eig_v[i])).real.reshape(-1, 1) \
                          * distribution(E - eig[i]).reshape(1, -1)

    return PDOS


def spin_moment(eig_v, S=None):
    r""" Calculate the spin magnetic moment (also known as spin texture)

    This calculation only makes sense for non-colinear calculations.

    The returned quantities are given in this order:

    - Total spin magnetic moment
    - Spin magnetic moment along :math:`x` direction
    - Spin magnetic moment along :math:`y` direction
    - Spin magnetic moment along :math:`z` direction

    These are calculated using the Pauli matrices :math:`\sigma_x`, :math:`\sigma_y` and :math:`\sigma_z`:

    .. math::
       \mathbf{S}_i^\sigma &= \sum_i \langle \psi^*_i |\sigma_z \mathbf S \sigma_z | \psi_i \rangle
       \\
       \mathbf{S}_i^x(E) &= \sum_i \langle \psi^*_i | \sigma_x \mathbf S | \psi_i \rangle
       \\
       \mathbf{S}_i^y(E) &= \sum_i \langle \psi^*_i | \sigma_y \mathbf S | \psi_i \rangle
       \\
       \mathbf{S}_i^z(E) &= \sum_i \langle \psi^*_i | \sigma_z \mathbf S | \psi_i \rangle

    Note that the total spin magnetic moment, S may be calculated using
    :math:`\sigma_i\sigma_i` where :math:`i` may be either of :math:`x`, :math:`y` or :math:`z`.

    Parameters
    ----------
    eig_v : array_like
       eigenvectors
    S : array_like, optional
       overlap matrix used in the :math:`\rangle\psi^*|\mathbf S|\psi\langle` calculation. If `None` the identity
       matrix is assumed. The overlap matrix should correspond to the system and :math:`k` point the eigenvectors
       have been evaluated at.

    See Also
    --------
    DOS : total DOS
    PDOS : projected DOS

    Returns
    -------
    numpy.ndarray
        spin moments per eigenvector with dimension ``(4, self.size, len(E))``.
        For non-colinear calculations it will be ``(4, self.size, len(E))``, ordered as
        indicated it the above list.
    """
    if S is None:
        class _S(object):
            @property
            def shape(self):
                return eig_v.shape
            def dot(self, v):
                return v
        S = _S()

    if S.shape[1] == eig_v.shape[1] // 2:
        S = S[::2, ::2]

    # Initialize
    s = np.empty([4, eig_v.shape[0]], dtype=dtype_complex_to_real(eig_v.dtype))

    v = S.dot(eig_v[0].reshape(-1, 2))
    D = (conj(eig_v[0]) * v.ravel()).reshape(-1, 2) # diagonal elements
    s[0, :] = D.sum(1) # total spin moment
    s[3, :] = D[:, 0] - D[:, 1] # S_z
    D = (conj(eig_v[0, 1::2]) * 2 * v[:, 0]) # psi_down * psi_up * 2
    s[1, :] = D.real # S_x
    s[2, :] = D.imag # S_y
    for i in range(1, len(eig_v)):
        v = S.dot(eig_v[i].reshape(-1, 2))
        D = (conj(eig_v[i]) * v.ravel()).reshape(-1, 2)
        s[0, :] += D.sum(1)
        s[3, :] += D[:, 0] - D[:, 1]
        D = conj(eig_v[i, 1::2]) * 2 * v[:, 0]
        s[1, :] += D.real
        s[2, :] += D.imag

    return s
