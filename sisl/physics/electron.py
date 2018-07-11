"""Electron related functions and classes
=========================================

.. module:: sisl.physics.electron
   :noindex:

In sisl electronic structure calculations are relying on routines
specific for electrons. For instance density of states calculations from
electronic eigenvalues and other quantities.

This module implements the necessary tools required for calculating
DOS, PDOS, band-velocities and spin moments of non-collinear calculations.
One may also plot real-space wavefunctions.

.. autosummary::
   :toctree:

   DOS
   PDOS
   velocity
   wavefunction
   spin_moment


Supporting classes
------------------

Certain classes aid in the usage of the above methods by implementing them
using automatic arguments.

For instance, the PDOS method requires the overlap matrix in non-orthogonal
basis sets at the :math:`k`-point corresponding to the eigenstates. Hence, the
argument ``S`` must be :math:`\mathbf S(\mathbf k)`. The `EigenstateElectron` class
automatically passes the correct ``S`` because it knows the states :math:`k`-point.

.. autosummary::
   :toctree:

   CoefficientElectron
   StateElectron
   StateCElectron
   EigenvalueElectron
   EigenvectorElectron
   EigenstateElectron

"""
from __future__ import print_function, division

import numpy as np
from numpy import floor, ceil
from numpy import conj, dot, ogrid
from numpy import cos, sin, pi, int32
from numpy import add

from sisl import units, constant
from sisl.supercell import SuperCell
from sisl.geometry import Geometry
from sisl._indices import indices_le
from sisl._math_small import xyz_to_spherical_cos_phi
import sisl._array as _a
from sisl.linalg import eigh_destroy
from sisl.messages import info, warn, SislError, tqdm_eta
from sisl._help import dtype_complex_to_real, _range as range
from .distribution import get_distribution
from .spin import Spin
from .sparse import SparseOrbitalBZSpin
from .state import Coefficient, State, StateC


__all__ = ['DOS', 'PDOS', 'spin_moment', 'velocity', 'inv_eff_mass_tensor', 'wavefunction']
__all__ += ['CoefficientElectron', 'StateElectron', 'StateCElectron']
__all__ += ['EigenvalueElectron', 'EigenvectorElectron', 'EigenstateElectron']


def DOS(E, eig, distribution='gaussian'):
    r""" Calculate the density of states (DOS) for a set of energies, `E`, with a distribution function

    The :math:`\mathrm{DOS}(E)` is calculated as:

    .. math::
       \mathrm{DOS}(E) = \sum_i D(E-\epsilon_i) \approx\delta(E-\epsilon_i)

    where :math:`D(\Delta E)` is the distribution function used. Note that the distribution function
    used may be a user-defined function. Alternatively a distribution function may
    be retrieved from `~sisl.physics.distribution`.

    Parameters
    ----------
    E : array_like
       energies to calculate the DOS at
    eig : array_like
       electronic eigenvalues
    distribution : func or str, optional
       a function that accepts :math:`\Delta E` as argument and calculates the
       distribution function.

    See Also
    --------
    sisl.physics.distribution : a selected set of implemented distribution functions
    PDOS : projected DOS (same as this, but projected onto each orbital)
    spin_moment: spin moment for states

    Returns
    -------
    numpy.ndarray : DOS calculated at energies, has same length as `E`
    """
    if isinstance(distribution, str):
        distribution = get_distribution(distribution)

    DOS = distribution(E - eig[0])
    for i in range(1, len(eig)):
        DOS += distribution(E - eig[i])
    return DOS


def PDOS(E, eig, state, S=None, distribution='gaussian', spin=None):
    r""" Calculate the projected density of states (PDOS) for a set of energies, `E`, with a distribution function

    The :math:`\mathrm{PDOS}(E)` is calculated as:

    .. math::
       \mathrm{PDOS}_\nu(E) = \sum_i \psi^*_{i,\nu} [\mathbf S | \psi_{i}\rangle]_\nu D(E-\epsilon_i)

    where :math:`D(\Delta E)` is the distribution function used. Note that the distribution function
    used may be a user-defined function. Alternatively a distribution function may
    be aquired from `~sisl.physics.distribution`.

    In case of an orthogonal basis set :math:`\mathbf S` is equal to the identity matrix.
    Note that `DOS` is the sum of the orbital projected DOS:

    .. math::
       \mathrm{DOS}(E) = \sum_\nu\mathrm{PDOS}_\nu(E)

    For non-collinear calculations (this includes spin-orbit calculations) the PDOS is additionally
    separated into 4 components (in this order):

    - Total projected DOS
    - Projected spin magnetic moment along :math:`x` direction
    - Projected spin magnetic moment along :math:`y` direction
    - Projected spin magnetic moment along :math:`z` direction

    These are calculated using the Pauli matrices :math:`\boldsymbol\sigma_x`, :math:`\boldsymbol\sigma_y` and :math:`\boldsymbol\sigma_z`:

    .. math::

       \mathrm{PDOS}_\nu^\Sigma(E) &= \sum_i \psi^*_{i,\nu} \boldsymbol\sigma_z \boldsymbol\sigma_z [\mathbf S | \psi_{i}\rangle]_\nu D(E-\epsilon_i)
       \\
       \mathrm{PDOS}_\nu^x(E) &= \sum_i \psi^*_{i,\nu} \boldsymbol\sigma_x [\mathbf S | \psi_{i}\rangle]_\nu D(E-\epsilon_i)
       \\
       \mathrm{PDOS}_\nu^y(E) &= \sum_i \psi^*_{i,\nu} \boldsymbol\sigma_y [\mathbf S | \psi_{i}\rangle]_\nu D(E-\epsilon_i)
       \\
       \mathrm{PDOS}_\nu^z(E) &= \sum_i \psi^*_{i,\nu} \boldsymbol\sigma_z [\mathbf S | \psi_{i}\rangle]_\nu D(E-\epsilon_i)

    Note that the total PDOS may be calculated using :math:`\boldsymbol\sigma_i\boldsymbol\sigma_i` where :math:`i` may be either of :math:`x`,
    :math:`y` or :math:`z`.

    Parameters
    ----------
    E : array_like
       energies to calculate the projected-DOS from
    eig : array_like
       eigenvalues
    state : array_like
       eigenvectors
    S : array_like, optional
       overlap matrix used in the :math:`\langle\psi|\mathbf S|\psi\rangle` calculation. If `None` the identity
       matrix is assumed. For non-collinear calculations this matrix may be halve the size of ``len(state[0, :])`` to
       trigger the non-collinear calculation of PDOS.
    distribution : func or str, optional
       a function that accepts :math:`E-\epsilon` as argument and calculates the
       distribution function.
    spin : str or Spin, optional
       the spin configuration. This is generally only needed when the eigenvectors correspond to a non-collinear
       calculation.

    See Also
    --------
    sisl.physics.distribution : a selected set of implemented distribution functions
    DOS : total DOS (same as summing over orbitals)
    spin_moment: spin moment for states

    Returns
    -------
    numpy.ndarray
        projected DOS calculated at energies, has dimension ``(state.shape[1], len(E))``.
        For non-collinear calculations it will be ``(4, state.shape[1] // 2, len(E))``, ordered as
        indicated in the above list.
    """
    if isinstance(distribution, str):
        distribution = get_distribution(distribution)

    # Figure out whether we are dealing with a non-collinear calculation
    if S is None:
        class S(object):
            __slots__ = []
            shape = (state.shape[1], state.shape[1])
            @staticmethod
            def dot(v):
                return v

    if spin is None:
        if S.shape[1] == state.shape[1] // 2:
            spin = Spin('nc')
            S = S[::2, ::2]
        else:
            spin = Spin()

    # check for non-collinear (or SO)
    if spin.kind > Spin.POLARIZED:
        # Non colinear eigenvectors
        if S.shape[1] == state.shape[1]:
            # Since we are going to reshape the eigen-vectors
            # to more easily get the mixed states, we can reduce the overlap matrix
            S = S[::2, ::2]

        # Initialize data
        PDOS = np.empty([4, state.shape[1] // 2, len(E)], dtype=dtype_complex_to_real(state.dtype))

        d = distribution(E - eig[0]).reshape(1, -1)
        v = S.dot(state[0].reshape(-1, 2))
        D = (conj(state[0]) * v.ravel()).real.reshape(-1, 2) # diagonal PDOS
        PDOS[0, :, :] = D.sum(1).reshape(-1, 1) * d # total DOS
        PDOS[3, :, :] = (D[:, 0] - D[:, 1]).reshape(-1, 1) * d # z-dos
        D = (conj(state[0, 1::2]) * 2 * v[:, 0]).reshape(-1, 1) # psi_down * psi_up * 2
        PDOS[1, :, :] = D.real * d # x-dos
        PDOS[2, :, :] = D.imag * d # y-dos
        for i in range(1, len(eig)):
            d = distribution(E - eig[i]).reshape(1, -1)
            v = S.dot(state[i].reshape(-1, 2))
            D = (conj(state[i]) * v.ravel()).real.reshape(-1, 2)
            PDOS[0, :, :] += D.sum(1).reshape(-1, 1) * d
            PDOS[3, :, :] += (D[:, 0] - D[:, 1]).reshape(-1, 1) * d
            D = (conj(state[i, 1::2]) * 2 * v[:, 0]).reshape(-1, 1)
            PDOS[1, :, :] += D.real * d
            PDOS[2, :, :] += D.imag * d

    else:
        PDOS = (conj(state[0]) * S.dot(state[0])).real.reshape(-1, 1) \
               * distribution(E - eig[0]).reshape(1, -1)
        for i in range(1, len(eig)):
            PDOS[:, :] += (conj(state[i]) * S.dot(state[i])).real.reshape(-1, 1) \
                          * distribution(E - eig[i]).reshape(1, -1)

    return PDOS


def spin_moment(state, S=None):
    r""" Calculate the spin magnetic moment (also known as spin texture)

    This calculation only makes sense for non-collinear calculations.

    The returned quantities are given in this order:

    - Spin magnetic moment along :math:`x` direction
    - Spin magnetic moment along :math:`y` direction
    - Spin magnetic moment along :math:`z` direction

    These are calculated using the Pauli matrices :math:`\boldsymbol\sigma_x`, :math:`\boldsymbol\sigma_y` and :math:`\boldsymbol\sigma_z`:

    .. math::

       \mathbf{S}_i^x &= \langle \psi_i | \boldsymbol\sigma_x \mathbf S | \psi_i \rangle
       \\
       \mathbf{S}_i^y &= \langle \psi_i | \boldsymbol\sigma_y \mathbf S | \psi_i \rangle
       \\
       \mathbf{S}_i^z &= \langle \psi_i | \boldsymbol\sigma_z \mathbf S | \psi_i \rangle

    Parameters
    ----------
    state : array_like
       vectors describing the electronic states, 2nd dimension contains the states
    S : array_like, optional
       overlap matrix used in the :math:`\langle\psi|\mathbf S|\psi\rangle` calculation. If `None` the identity
       matrix is assumed. The overlap matrix should correspond to the system and :math:`k` point the eigenvectors
       have been evaluated at.

    Notes
    -----
    This routine cannot check whether the input eigenvectors originate from a non-collinear calculation.
    If a non-polarized eigenvector is passed to this routine, the output will have no physical meaning.

    See Also
    --------
    DOS : total DOS
    PDOS : projected DOS

    Returns
    -------
    numpy.ndarray
        spin moments per state with final dimension ``(state.shape[0], 3)``.
    """
    if state.ndim == 1:
        return spin_moment(state.reshape(1, -1), S).ravel()

    if S is None:
        class S(object):
            __slots__ = []
            shape = (state.shape[1] // 2, state.shape[1] // 2)
            @staticmethod
            def dot(v):
                return v

    if S.shape[1] == state.shape[1]:
        S = S[::2, ::2]

    # Initialize
    s = np.empty([state.shape[0], 3], dtype=dtype_complex_to_real(state.dtype))

    # TODO consider doing this all in a few lines
    # TODO Since there are no energy dependencies here we can actually do all
    # TODO dot products in one go and then use b-casting rules. Should be much faster
    # TODO but also way more memory demanding!
    for i in range(len(state)):
        Sstate = S.dot(state[i].reshape(-1, 2))
        D = (conj(state[i]) * Sstate.ravel()).real.reshape(-1, 2)
        s[i, 2] = (D[:, 0] - D[:, 1]).sum()
        D = 2 * (conj(state[i, 1::2]) * Sstate[:, 0]).sum()
        s[i, 0] = D.real
        s[i, 1] = D.imag

    return s


def velocity(state, dHk, energy=None, dSk=None, degenerate=None):
    r""" Calculate the velocity of a set of states

    These are calculated using the analytic expression (:math:`\alpha` corresponding to the Cartesian directions):

    .. math::

       \mathbf{v}_{i\alpha} = \frac1\hbar \langle \psi_i |
                \frac{\partial}{\partial\mathbf k}_\alpha \mathbf H(\mathbf k) | \psi_i \rangle

    In case of non-orthogonal basis the equations substitutes :math:`\mathbf H(\mathbf k)` by
    :math:`\mathbf H(\mathbf k) - \epsilon_i\mathbf S(\mathbf k)`.

    Parameters
    ----------
    state : array_like
       vectors describing the electronic states, 2nd dimension contains the states. In case of degenerate
       states the vectors *may* be rotated upon return.
    dHk : list of array_like
       Hamiltonian derivative with respect to :math:`\mathbf k`. This needs to be a tuple or
       list of the Hamiltonian derivative along the 3 Cartesian directions.
    energy : array_like, optional
       energies of the states. Required for non-orthogonal basis together with `dSk`. In case of degenerate
       states the eigenvalues of the states will be averaged in the degenerate sub-space.
    dSk : list of array_like, optional
       :math:`\delta \mathbf S_k` matrix required for non-orthogonal basis. This and `energy` *must* both be
       provided in a non-orthogonal basis (otherwise the results will be wrong).
       Same derivative as `dHk`
    degenerate: list of array_like, optional
       a list containing the indices of degenerate states. In that case a prior diagonalization
       is required to decouple them. This is done 3 times along each of the Cartesian directions.

    See Also
    --------
    inv_eff_mass_tensor : inverse effective mass tensor

    Returns
    -------
    numpy.ndarray
        velocities per state with final dimension ``(state.shape[0], 3)``, the velocity unit is Ang/ps
        Units *may* change in future releases.
    """
    if state.ndim == 1:
        return velocity(state.reshape(1, -1), dHk, energy, dSk, degenerate).ravel()

    if dSk is None:
        return _velocity_ortho(state, dHk, degenerate)
    return _velocity_non_ortho(state, dHk, energy, dSk, degenerate)


# dHk is in [Ang eV]
# velocity units in [Ang/ps]
_velocity_const = units('ps', 's') / constant.hbar('eV s')


def _velocity_non_ortho(state, dHk, energy, dSk, degenerate):
    r""" For states in a non-orthogonal basis """
    # Along all directions
    v = np.empty([state.shape[0], 3], dtype=dtype_complex_to_real(state.dtype))

    # Decouple the degenerate states
    if not degenerate is None:
        for deg in degenerate:
            # Set the average energy
            e = np.average(energy[deg])
            energy[deg] = e

            # Now diagonalize to find the contributions from individual states
            # then re-construct the seperated degenerate states
            # Since we do this for all directions we should decouple them all
            vv = conj(state[deg, :]).dot((dHk[0] - e * dSk[0]).dot(state[deg, :].T))
            S = eigh_destroy(vv)[1].T.dot(state[deg, :])
            vv = conj(S).dot((dHk[1] - e * dSk[1]).dot(S.T))
            S = eigh_destroy(vv)[1].T.dot(S)
            vv = conj(S).dot((dHk[2] - e * dSk[2]).dot(S.T))
            state[deg, :] = eigh_destroy(vv)[1].T.dot(S)

    # Since they depend on the state energies and dSk we have to loop them individually.
    for s, e in enumerate(energy):

        # Since dHk *may* be a csr_matrix or sparse, we have to do it like
        # this. A sparse matrix cannot be re-shaped with an extra dimension.
        v[s, 0] = (conj(state[s]) * (dHk[0] - e * dSk[0]).dot(state[s])).sum().real
        v[s, 1] = (conj(state[s]) * (dHk[1] - e * dSk[1]).dot(state[s])).sum().real
        v[s, 2] = (conj(state[s]) * (dHk[2] - e * dSk[2]).dot(state[s])).sum().real

    return v * _velocity_const


def _velocity_ortho(state, dHk, degenerate):
    r""" For states in an orthogonal basis """

    # Along all directions
    v = np.empty([state.shape[0], 3], dtype=dtype_complex_to_real(state.dtype))

    # Decouple the degenerate states
    if not degenerate is None:
        for deg in degenerate:
            # Now diagonalize to find the contributions from individual states
            # then re-construct the seperated degenerate states
            # Since we do this for all directions we should decouple them all
            vv = conj(state[deg, :]).dot(dHk[0].dot(state[deg, :].T))
            S = eigh_destroy(vv)[1].T.dot(state[deg, :])
            vv = conj(S).dot((dHk[1]).dot(S.T))
            S = eigh_destroy(vv)[1].T.dot(S)
            vv = conj(S).dot((dHk[2]).dot(S.T))
            state[deg, :] = eigh_destroy(vv)[1].T.dot(S)

    v[:, 0] = (conj(state.T) * dHk[0].dot(state.T)).sum(0).real
    v[:, 1] = (conj(state.T) * dHk[1].dot(state.T)).sum(0).real
    v[:, 2] = (conj(state.T) * dHk[2].dot(state.T)).sum(0).real

    return v * _velocity_const


def inv_eff_mass_tensor(state, ddHk, energy=None, ddSk=None, degenerate=None, as_matrix=False):
    r""" Calculate the effective mass tensor for a set of states

    These are calculated using the analytic expression (:math:`\alpha,\beta` corresponds to Cartesian directions):

    .. math::

        \mathbf M^{-1}_{i\alpha\beta} = \frac1{\hbar^2} \langle \psi_i |
             \frac{\partial}{\partial\mathbf k}_\alpha\frac{\partial}{\partial\mathbf k}_\beta \mathbf H(\mathbf k)
             | \psi_i \rangle

    In case of non-orthogonal basis the equations substitutes :math:`\mathbf H(\mathbf k)` by
    :math:`\mathbf H(\mathbf k) - \epsilon_i\mathbf S(\mathbf k)`.

    The matrix :math:`\mathbf M` is known as the effective mass tensor, remark that this function returns the inverse
    of :math:`\mathbf M`.

    Notes
    -----
    The reason for not inverting the mass-tensor is that for systems with limited
    periodicities some of the diagonal elements of the inverse mass tensor matrix
    will be 0, in which case the matrix is singular and non-invertible. Therefore
    it is the users responsibility to remove any of the non-periodic elements from
    the matrix before inverting.

    Parameters
    ----------
    state : array_like
       vectors describing the electronic states, 2nd dimension contains the states. In case of degenerate
       states the vectors *may* be rotated upon return.
    ddHk : (6,) of array_like
       Hamiltonian double derivative with respect to :math:`\mathbf k`. The input must be in Voigt order.
    energy : array_like, optional
       energies of the states. Required for non-orthogonal basis together with `ddSk`. In case of degenerate
       states the eigenvalues of the states will be averaged in the degenerate sub-space.
    ddSk : (6,) of array_like, optional
       overlap matrix required for non-orthogonal basis. This and `energy` *must* both be
       provided when the states are defined in a non-orthogonal basis (otherwise the results will be wrong).
       Same order as `ddHk`.
    degenerate: list of array_like, optional
       a list containing the indices of degenerate states. In that case a subsequent diagonalization
       is required to decouple them. This is done 3 times along the diagonal Cartesian directions.
    as_matrix : bool, optional
       if true the returned tensor will be a symmetric matrix, otherwise the Voigt tensor is returned.

    See Also
    --------
    velocity : band velocity

    Returns
    -------
    numpy.ndarray
        inverse effective mass tensor of each state in units of inverse electron mass
    """
    if state.ndim == 1:
        return inv_eff_mass_tensor(state.reshape(1, -1), ddHk, energy, ddSk, degenerate, as_matrix).ravel()

    if ddSk is None:
        return _inv_eff_mass_tensor_ortho(state, ddHk, degenerate, as_matrix)
    return _inv_eff_mass_tensor_non_ortho(state, ddHk, energy, ddSk, degenerate, as_matrix)


# inverse electron mass units in 1/m_e (atomic units)!
# ddHk is in [Ang ^ 2 eV]
_inv_eff_mass_const = units('Ang', 'Bohr') ** 2 * units('eV', 'Ha')


def _inv_eff_mass_tensor_non_ortho(state, ddHk, energy, ddSk, degenerate, as_matrix):
    r""" For states in a non-orthogonal basis """
    if as_matrix:
        M = np.empty([state.shape[0], 9], dtype=dtype_complex_to_real(state.dtype))
    else:
        M = np.empty([state.shape[0], 6], dtype=dtype_complex_to_real(state.dtype))

    # Now decouple the degenerate states
    if not degenerate is None:
        for deg in degenerate:
            e = np.average(energy[deg])
            energy[deg] = e

            # Now diagonalize to find the contributions from individual states
            # then re-construct the seperated degenerate states
            # We only do this along the double derivative directions
            vv = conj(state[deg, :]).dot((ddHk[0] - e * ddSk[0]).dot(state[deg, :].T))
            S = eigh_destroy(vv)[1].T.dot(state[deg, :])
            vv = conj(S).dot((ddHk[1] - e * ddSk[1]).dot(S.T))
            S = eigh_destroy(vv)[1].T.dot(S)
            vv = conj(S).dot((ddHk[2] - e * ddSk[2]).dot(S.T))
            state[deg, :] = eigh_destroy(vv)[1].T.dot(S)

    # Since they depend on the state energies and ddSk we have to loop them individually.
    for s, e in enumerate(energy):

        # Since ddHk *may* be a csr_matrix or sparse, we have to do it like
        # this. A sparse matrix cannot be re-shaped with an extra dimension.
        for i in range(6):
            M[s, i] = (conj(state[s]) * (ddHk[i] - e * ddSk[i]).dot(state[s])).sum().real

    if as_matrix:
        M[:, 8] = M[:, 2] # zz
        M[:, 7] = M[:, 3] # zy
        M[:, 6] = M[:, 4] # zx
        M[:, 3] = M[:, 5] # xy
        M[:, 5] = M[:, 7] # yz
        M[:, 4] = M[:, 1] # yy
        M[:, 1] = M[:, 3] # xy
        M[:, 2] = M[:, 6] # zx
        M.shape = (-1, 3, 3)

    return M * _inv_eff_mass_const


def _inv_eff_mass_tensor_ortho(state, ddHk, degenerate, as_matrix):
    r""" For states in an orthogonal basis """

    # Along all directions
    if as_matrix:
        M = np.empty([state.shape[0], 9], dtype=dtype_complex_to_real(state.dtype))
    else:
        M = np.empty([state.shape[0], 6], dtype=dtype_complex_to_real(state.dtype))

    # Now decouple the degenerate states
    if not degenerate is None:
        for deg in degenerate:
            # Now diagonalize to find the contributions from individual states
            # then re-construct the seperated degenerate states
            # We only do this along the double derivative directions
            vv = conj(state[deg, :]).dot(ddHk[0].dot(state[deg, :].T))
            S = eigh_destroy(vv)[1].T.dot(state[deg, :])
            vv = conj(S).dot(ddHk[1].dot(S.T))
            S = eigh_destroy(vv)[1].T.dot(S)
            vv = conj(S).dot(ddHk[2].dot(S.T))
            state[deg, :] = eigh_destroy(vv)[1].T.dot(S)

    for i in range(6):
        M[:, i] = (conj(state.T) * ddHk[i].dot(state.T)).sum(0).real

    if as_matrix:
        M[:, 8] = M[:, 2] # zz
        M[:, 7] = M[:, 3] # zy
        M[:, 6] = M[:, 4] # zx
        M[:, 3] = M[:, 5] # xy
        M[:, 5] = M[:, 7] # yz
        M[:, 4] = M[:, 1] # yy
        M[:, 1] = M[:, 3] # xy
        M[:, 2] = M[:, 6] # zx
        M.shape = (-1, 3, 3)

    return M * _inv_eff_mass_const


def wavefunction(v, grid, geometry=None, k=None, spinor=0, spin=None, eta=False):
    r""" Add the wave-function (`Orbital.psi`) component of each orbital to the grid

    This routine calculates the real-space wave-function components in the
    specified grid.

    This is an *in-place* operation that *adds* to the current values in the grid.

    It may be instructive to check that an eigenstate is normalized:

    >>> grid = Grid(...) # doctest: +SKIP
    >>> psi(state, grid) # doctest: +SKIP
    >>> (np.abs(grid.grid) ** 2).sum() * grid.dvolume == 1. # doctest: +SKIP

    Note: To calculate :math:`\psi(\mathbf r)` in a unit-cell different from the
    originating geometry, simply pass a grid with a unit-cell smaller than the originating
    supercell.

    The wavefunctions are calculated in real-space via:

    .. math::
       \psi(\mathbf r) = \sum_i\phi_i(\mathbf r) |\psi\rangle_i \exp(-i\mathbf k \mathbf R)

    While for non-collinear/spin-orbit calculations the wavefunctions are determined from the
    spinor component (`spinor`)

    .. math::
       \psi_{\alpha/\beta}(\mathbf r) = \sum_i\phi_i(\mathbf r) |\psi_{\alpha/\beta}\rangle_i \exp(-i\mathbf k \mathbf R)

    where ``spinor in [0, 1]`` determines :math:`\alpha` or :math:`\beta`, respectively.

    Notes
    -----
    Currently this method only works for `v` being coefficients of the gauge='R' method. In case
    you are passing a `v` with the incorrect gauge you will find a phase-shift according to:

    .. math::
        \tilde v_j = e^{-i\mathbf k\mathbf r_j} v_j

    where :math:`j` is the orbital index and :math:`\mathbf r_j` is the orbital position.


    Parameters
    ----------
    v : array_like
       coefficients for the orbital expansion on the real-space grid.
       If `v` is a complex array then the `grid` *must* be complex as well. The coefficients
       must be using the ``R`` gauge.
    grid : Grid
       grid on which the wavefunction will be plotted.
       If multiple eigenstates are in this object, they will be summed.
    geometry : Geometry, optional
       geometry where the orbitals are defined. This geometry's orbital count must match
       the number of elements in `v`.
       If this is ``None`` the geometry associated with `grid` will be used instead.
    k : array_like, optional
       k-point associated with wavefunction, by default the inherent k-point used
       to calculate the eigenstate will be used (generally shouldn't be used unless the `EigenstateElectron` object
       has not been created via `Hamiltonian.eigenstate`).
    spinor : int, optional
       the spinor for non-collinear/spin-orbit calculations. This is only used if the
       eigenstate object has been created from a parent object with a `Spin` object
       contained, *and* if the spin-configuration is non-collinear or spin-orbit coupling.
       Default to the first spinor component.
    spin : Spin, optional
       specification of the spin configuration of the orbital coefficients. This only has
       influence for non-collinear wavefunctions where `spinor` choice is important.
    eta : bool, optional
       Display a console progressbar.
    """
    if geometry is None:
        geometry = grid.geometry
        warn('wavefunction was not passed a geometry associated, will use the geometry associated with the Grid.')
    if geometry is None:
        raise SislError('wavefunction did not find a usable Geometry through keywords or the Grid!')

    # In case the user has passed several vectors we sum them to plot the summed state
    if v.ndim == 2:
        v = v.sum(0)

    if spin is None:
        if len(v) // 2 == geometry.no:
            # We can see from the input that the vector *must* be a non-collinear calculation
            v = v.reshape(-1, 2)[:, spinor]
            info('wavefunction assumes the input wavefunction coefficients to originate from a non-collinear calculation!')

    elif spin.kind > Spin.POLARIZED:
        # For non-collinear cases the user selects the spinor component.
        v = v.reshape(-1, 2)[:, spinor]

    if len(v) != geometry.no:
        raise ValueError("wavefunction require wavefunction coefficients corresponding to number of orbitals in the geometry.")

    # Check for k-points
    k = _a.asarrayd(k)
    kl = (k ** 2).sum() ** 0.5
    has_k = kl > 0.000001
    if has_k:
        info('wavefunction for k != Gamma is currently untested!')

    # Check that input/grid makes sense.
    # If the coefficients are complex valued, then the grid *has* to be
    # complex valued.
    # Likewise if a k-point has been passed.
    is_complex = np.iscomplexobj(v) or has_k
    if is_complex and not np.iscomplexobj(grid.grid):
        raise SislError("wavefunction input coefficients are complex, while grid only contains real.")

    if is_complex:
        psi_init = _a.zerosz
    else:
        psi_init = _a.zerosd

    # Extract sub variables used throughout the loop
    shape = _a.asarrayi(grid.shape)
    dcell = grid.dcell
    ic = grid.sc.icell
    geom_shape = dot(ic, geometry.cell.T) * shape.reshape(1, -1)

    # In the following we don't care about division
    # So 1) save error state, 2) turn off divide by 0, 3) calculate, 4) turn on old error state
    old_err = np.seterr(divide='ignore', invalid='ignore')

    addouter = add.outer
    def idx2spherical(ix, iy, iz, offset, dc, R):
        """ Calculate the spherical coordinates from indices """
        rx = addouter(addouter(ix * dc[0, 0], iy * dc[1, 0]), iz * dc[2, 0] - offset[0]).ravel()
        ry = addouter(addouter(ix * dc[0, 1], iy * dc[1, 1]), iz * dc[2, 1] - offset[1]).ravel()
        rz = addouter(addouter(ix * dc[0, 2], iy * dc[1, 2]), iz * dc[2, 2] - offset[2]).ravel()

        # Total size of the indices
        n = rx.shape[0]
        # Reduce our arrays to where the radius is "fine"
        idx = indices_le(rx ** 2 + ry ** 2 + rz ** 2, R ** 2)
        rx = rx[idx]
        ry = ry[idx]
        rz = rz[idx]
        xyz_to_spherical_cos_phi(rx, ry, rz)
        return n, idx, rx, ry, rz

    # Figure out the max-min indices with a spacing of 1 radian
    rad1 = pi / 180
    theta, phi = ogrid[-pi:pi:rad1, 0:pi:rad1]
    cphi, sphi = cos(phi), sin(phi)
    ctheta_sphi = cos(theta) * sphi
    stheta_sphi = sin(theta) * sphi
    del sphi
    nrxyz = (theta.size, phi.size, 3)
    del theta, phi, rad1

    # First we calculate the min/max indices for all atoms
    idx_mm = _a.emptyi([geometry.na, 2, 3])
    rxyz = _a.emptyd(nrxyz)
    rxyz[..., 0] = ctheta_sphi
    rxyz[..., 1] = stheta_sphi
    rxyz[..., 2] = cphi
    # Reshape
    rxyz.shape = (-1, 3)
    idx = dot(ic, rxyz.T).T * shape.reshape(1, -1)
    idxm = idx.min(0).reshape(1, 3)
    idxM = idx.max(0).reshape(1, 3)
    del ctheta_sphi, stheta_sphi, cphi, idx, rxyz, nrxyz

    origo = grid.sc.origo.reshape(1, -1)
    for atom, ia in geometry.atom.iter(True):
        if len(ia) == 0:
            continue
        R = atom.maxR()

        # Now do it for all the atoms to get indices of the middle of
        # the atoms
        # The coordinates are relative to origo, so we need to shift (when writing a grid
        # it is with respect to origo)
        xyz = geometry.xyz[ia, :] - origo
        idx = dot(ic, xyz.T).T * shape.reshape(1, -1)

        # Get min-max for all atoms
        idx_mm[ia, 0, :] = idxm * R + idx
        idx_mm[ia, 1, :] = idxM * R + idx

    # Now we have min-max for all atoms
    # When we run the below loop all indices can be retrieved by looking
    # up in the above table.

    # Before continuing, we can easily clean up the temporary arrays
    del origo, idx

    arangei = _a.arangei

    # In case this grid does not have a Geometry associated
    # We can *perhaps* easily attach a geometry with the given
    # atoms in the unit-cell
    sc = grid.sc.copy()
    # Find the periodic directions
    pbc = [bc == grid.PERIODIC or geometry.nsc[i] > 1 for i, bc in enumerate(grid.bc[:, 0])]
    if grid.geometry is None:
        # Create the actual geometry that encompass the grid
        ia, xyz, _ = geometry.within_inf(sc, periodic=pbc)
        if len(ia) > 0:
            grid.set_geometry(Geometry(xyz, geometry.atom[ia], sc=sc))

    # Instead of looping all atoms in the supercell we find the exact atoms
    # and their supercell indices.
    add_R = _a.zerosd(3) + geometry.maxR()
    # Calculate the required additional vectors required to increase the fictitious
    # supercell by add_R in each direction.
    # For extremely skewed lattices this will be way too much, hence we make
    # them square.
    o = sc.toCuboid(True)
    sc = SuperCell(o._v, origo=o.origo - add_R) + np.diag(2 * add_R)

    # Retrieve all atoms within the grid supercell
    # (and the neighbours that connect into the cell)
    IA, XYZ, ISC = geometry.within_inf(sc, periodic=pbc, origo=grid.origo)

    r_k = dot(geometry.rcell, k)
    r_k_cell = dot(r_k, geometry.cell)
    phase = 1

    # Retrieve progressbar
    eta = tqdm_eta(len(IA), 'wavefunction', 'atom', eta)

    # Loop over all atoms in the grid-cell
    for ia, xyz, isc in zip(IA, XYZ, ISC):
        # Get current atom
        atom = geometry.atom[ia]

        # Extract maximum R
        R = atom.maxR()
        if R <= 0.:
            warn("Atom '{}' does not have a wave-function, skipping atom.".format(atom))
            eta.update()
            continue

        # Get indices in the supercell grid
        idx = (isc.reshape(3, 1) * geom_shape).sum(0)
        idxm = floor(idx_mm[ia, 0, :] + idx).astype(int32)
        idxM = ceil(idx_mm[ia, 1, :] + idx).astype(int32) + 1

        # Fast check whether we can skip this point
        if idxm[0] >= shape[0] or idxm[1] >= shape[1] or idxm[2] >= shape[2] or \
           idxM[0] <= 0 or idxM[1] <= 0 or idxM[2] <= 0:
            eta.update()
            continue

        # Truncate values
        if idxm[0] < 0:
            idxm[0] = 0
        if idxM[0] > shape[0]:
            idxM[0] = shape[0]
        if idxm[1] < 0:
            idxm[1] = 0
        if idxM[1] > shape[1]:
            idxM[1] = shape[1]
        if idxm[2] < 0:
            idxm[2] = 0
        if idxM[2] > shape[2]:
            idxM[2] = shape[2]

        # Now idxm/M contains min/max indices used
        # Convert to spherical coordinates
        n, idx, r, theta, phi = idx2spherical(arangei(idxm[0], idxM[0]),
                                              arangei(idxm[1], idxM[1]),
                                              arangei(idxm[2], idxM[2]), xyz, dcell, R)

        # Get initial orbital
        io = geometry.a2o(ia)

        if has_k:
            phase = np.exp(-1j * (dot(r_k_cell, isc)))

        # Allocate a temporary array where we add the psi elements
        psi = psi_init(n)

        # Loop on orbitals on this atom, grouped by radius
        for os in atom.iter(True):

            # Get the radius of orbitals (os)
            oR = os[0].R

            if oR <= 0.:
                warn("Orbital(s) '{}' does not have a wave-function, skipping orbital!".format(os))
                # Skip these orbitals
                io += len(os)
                continue

            # Downsize to the correct indices
            if R - oR < 1e-6:
                idx1 = idx.view()
                r1 = r.view()
                theta1 = theta.view()
                phi1 = phi.view()
            else:
                idx1 = indices_le(r, oR)
                # Reduce arrays
                r1 = r[idx1]
                theta1 = theta[idx1]
                phi1 = phi[idx1]
                idx1 = idx[idx1]

            # Loop orbitals with the same radius
            for o in os:
                # Evaluate psi component of the wavefunction and add it for this atom
                psi[idx1] += o.psi_spher(r1, theta1, phi1, cos_phi=True) * (v[io] * phase)
                io += 1

        # Clean-up
        del idx1, r1, theta1, phi1, idx, r, theta, phi

        # Convert to correct shape and add the current atom contribution to the wavefunction
        psi.shape = idxM - idxm
        grid.grid[idxm[0]:idxM[0], idxm[1]:idxM[1], idxm[2]:idxM[2]] += psi

        # Clean-up
        del psi

        # Step progressbar
        eta.update()

    eta.close()

    # Reset the error code for division
    np.seterr(**old_err)


class _electron_State(object):
    __slots__ = []

    def __is_nc(self):
        """ Internal routine to check whether this is a non-collinear calculation """
        try:
            return self.parent.spin > Spin.POLARIZED
        except:
            return False

    def Sk(self, format='csr', spin=None):
        r""" Retrieve the overlap matrix corresponding to the originating parent structure.

        When ``self.parent`` is a Hamiltonian this will return :math:`\mathbf S(k)` for the
        :math:`k`-point these eigenstates originate from

        Parameters
        ----------
        format: str, optional
           the returned format of the overlap matrix. This only takes effect for
           non-orthogonal parents.
        spin : Spin, optional
           for non-collinear spin configurations the *fake* overlap matrix returned
           will have halve the size of the input matrix. If you want the *full* overlap
           matrix, simply do not specify the `spin` argument.
        """

        if isinstance(self.parent, SparseOrbitalBZSpin):
            # Calculate the overlap matrix
            if not self.parent.orthogonal:
                opt = {'k': self.info.get('k', (0, 0, 0)),
                       'format': format}
                gauge = self.info.get('gauge', None)
                if not gauge is None:
                    opt['gauge'] = gauge
                return self.parent.Sk(**opt)

        class __FakeSk(object):
            """ Replacement object which superseedes a matrix """
            __slots__ = []
            shape = (self.shape[1], self.shape[1])
            @staticmethod
            def dot(v):
                return v

        if spin is None:
            return __FakeSk
        if spin.kind > Spin.POLARIZED:
            class __FakeSk(object):
                """ Replacement object which superseedes a matrix """
                __slots__ = []
                shape = (self.shape[1] // 2, self.shape[1] // 2)
                @staticmethod
                def dot(v):
                    return v
        return __FakeSk

    def norm2(self, sum=True):
        r""" Return a vector with the norm of each state :math:`\langle\psi|\psi\rangle`

        Parameters
        ----------
        sum : bool, optional
           if true the summed orbital square is returned (a vector). For false a matrix
           with normalization squared per orbital is returned.

        Returns
        -------
        numpy.ndarray
            the normalization on each orbital for each state
        """
        # Retrieve the overlap matrix (FULL S is required for NC)
        S = self.Sk()

        # TODO, perhaps check that it is correct... and fix multiple transposes
        if sum:
            if self.__is_nc():
                return (conj(self.state) * S.dot(self.state.T).T).real.reshape(len(self), -1, 2).sum(-1).sum(0)
            return (conj(self.state) * S.dot(self.state.T).T).real.sum(0)
        if self.__is_nc():
            return (conj(self.state) * S.dot(self.state.T).T).real.reshape(len(self), -1, 2).sum(-1)
        return (conj(self.state) * S.dot(self.state.T).T).real

    def spin_moment(self):
        r""" Calculate spin moment from the states

        This routine calls `~sisl.physics.electron.spin_moment` with appropriate arguments
        and returns the spin moment for the states.

        See `~sisl.physics.electron.spin_moment` for details.
        """
        try:
            spin = self.parent.spin
        except:
            spin = None
        return spin_moment(self.state, self.Sk(spin=spin))

    def wavefunction(self, grid, spinor=0, eta=False):
        r""" Expand the coefficients as the wavefunction on `grid` *as-is*

        See `~sisl.physics.electron.wavefunction` for argument details.
        """
        try:
            spin = self.parent.spin
        except:
            spin = None

        if isinstance(self.parent, Geometry):
            geometry = self.parent
        else:
            try:
                geometry = self.parent.geometry
            except:
                geometry = None

        # Ensure we are dealing with the R gauge
        self.change_gauge('R')

        # Retrieve k
        k = self.info.get('k', _a.zerosd(3))

        wavefunction(self.state, grid, geometry=geometry, k=k, spinor=spinor,
                     spin=spin, eta=eta)

    def change_gauge(self, gauge):
        r""" In-place change of the gauge of the state coefficients

        The two gauges are related through:

        .. math::

            \tilde C_j = e^{-i\mathbf k\mathbf r_j} C_j

        where :math:`C_j` belongs to the gauge ``R`` and :math:`\tilde C_j` is in the gauge
        ``r``.

        Parameters
        ----------
        gauge : {'R', 'r'}
            specify the new gauge for the state coefficients
        """
        # These calls will fail if the gauge is not specified.
        # In that case we simply don't care about the raised issues.
        if self.info.get('gauge') == gauge:
            # Quick return
            return

        # Update gauge value
        self.info['gauge'] = gauge

        # Check that we can do a gauge transformation
        k = _a.asarrayd(self.info.get('k'))
        if (k ** 2).sum() ** 0.5 <= 0.000001:
            return

        g = self.parent.geometry
        k = dot(g.rcell, k)
        phase = dot(g.xyz[g.o2a(_a.arangei(g.no)), :], k)

        if gauge == 'r':
            self.state *= np.exp(-1j * phase).reshape(1, -1)
        elif gauge == 'R':
            self.state *= np.exp(1j * phase).reshape(1, -1)

    # TODO to be deprecated
    psi = wavefunction


class CoefficientElectron(Coefficient):
    """ Coefficients describing some physical quantity related to electrons """
    __slots__ = []


class StateElectron(_electron_State, State):
    """ A state describing a physical quantity related to electrons """
    __slots__ = []


class StateCElectron(_electron_State, StateC):
    """ A state describing a physical quantity related to electrons, with associated coefficients of the state """
    __slots__ = []

    def velocity(self, eps=1e-4):
        r""" Calculate velocity for the states

        This routine calls `~sisl.physics.electron.velocity` with appropriate arguments
        and returns the velocity for the states. I.e. for non-orthogonal basis the overlap
        matrix and energy values are also passed.

        Note that the coefficients associated with the `StateCElectron` *must* correspond
        to the energies of the states.

        See `~sisl.physics.electron.velocity` for details.

        Parameters
        ----------
        eps : float, optional
           precision used to find degenerate states.
        """
        try:
            opt = {'k': self.info.get('k', (0, 0, 0))}
            gauge = self.info.get('gauge', None)
            if not gauge is None:
                opt['gauge'] = gauge

            # Get dSk before spin
            if self.parent.orthogonal:
                dSk = None
            else:
                dSk = self.parent.dSk(**opt)

            if 'spin' in self.info:
                opt['spin'] = self.info.get('spin', None)
            deg = self.degenerate(eps)
        except:
            raise SislError(self.__class__.__name__ + '.velocity requires the parent to have a spin associated.')
        return velocity(self.state, self.parent.dHk(**opt), self.c, dSk, degenerate=deg)

    def inv_eff_mass_tensor(self, as_matrix=False, eps=1e-3):
        r""" Calculate inverse effective mass tensor for the states

        This routine calls `~sisl.physics.electron.inv_eff_mass` with appropriate arguments
        and returns the state inverse effective mass tensor. I.e. for non-orthogonal basis the overlap
        matrix and energy values are also passed.

        Note that the coefficients associated with the `StateCElectron` *must* correspond
        to the energies of the states.

        See `~sisl.physics.electron.inv_eff_mass_tensor` for details.

        Notes
        -----
        The reason for not inverting the mass-tensor is that for systems with limited
        periodicities some of the diagonal elements of the inverse mass tensor matrix
        will be 0, in which case the matrix is singular and non-invertible. Therefore
        it is the users responsibility to remove any of the non-periodic elements from
        the matrix.

        Parameters
        ----------
        as_matrix : bool, optional
           if true the returned tensor will be a symmetric matrix, otherwise the Voigt tensor is returned.
        eps : float, optional
           precision used to find degenerate states.
        """
        try:
            opt = {'k': self.info.get('k', (0, 0, 0))}
            gauge = self.info.get('gauge', None)
            if not gauge is None:
                opt['gauge'] = gauge

            # Get dSk before spin
            if self.parent.orthogonal:
                ddSk = None
            else:
                ddSk = self.parent.ddSk(**opt)

            if 'spin' in self.info:
                opt['spin'] = self.info.get('spin', None)
            degenerate = self.degenerate(eps)
        except:
            raise SislError(self.__class__.__name__ + '.inv_eff_mass_tensor requires the parent to have a spin associated.')
        return inv_eff_mass_tensor(self.state, self.parent.ddHk(**opt), self.c, ddSk, degenerate, as_matrix)


class EigenvalueElectron(CoefficientElectron):
    """ Eigenvalues of electronic states, no eigenvectors retained

    This holds routines that enable the calculation of density of states.
    """
    __slots__ = []

    @property
    def eig(self):
        return self.c

    def occupation(self, distribution='fermi_dirac'):
        """ Calculate the occupations for the states according to a distribution function

        Parameters
        ----------
        distribution : str or func, optional
           distribution used to find occupations

        Returns
        -------
        numpy.ndarray : len(self) with occupation values
        """
        if isinstance(distribution, str):
            distribution = get_distribution(distribution)
        return distribution(self.eig)

    def DOS(self, E, distribution='gaussian'):
        r""" Calculate DOS for provided energies, `E`.

        This routine calls `sisl.physics.electron.DOS` with appropriate arguments
        and returns the DOS.

        See `~sisl.physics.electron.DOS` for argument details.
        """
        return DOS(E, self.eig, distribution)


class EigenvectorElectron(StateElectron):
    """ Eigenvectors of electronic states, no eigenvalues retained

    This holds routines that enable the calculation of spin moments.
    """
    __slots__ = []


class EigenstateElectron(StateCElectron):
    """ Eigen states of electrons with eigenvectors and eigenvalues.

    This holds routines that enable the calculation of (projected) density of states,
    spin moments (spin texture).
    """
    __slots__ = []

    @property
    def eig(self):
        return self.c

    def occupation(self, distribution='fermi_dirac'):
        """ Calculate the occupations for the states according to a distribution function

        Parameters
        ----------
        distribution : str or func, optional
           distribution used to find occupations

        Returns
        -------
        numpy.ndarray : len(self) with occupation values
        """
        if isinstance(distribution, str):
            distribution = get_distribution(distribution)
        return distribution(self.eig)

    def DOS(self, E, distribution='gaussian'):
        r""" Calculate DOS for provided energies, `E`.

        This routine calls `sisl.physics.electron.DOS` with appropriate arguments
        and returns the DOS.

        See `~sisl.physics.electron.DOS` for argument details.
        """
        return DOS(E, self.c, distribution)

    def PDOS(self, E, distribution='gaussian'):
        r""" Calculate PDOS for provided energies, `E`.

        This routine calls `~sisl.physics.electron.PDOS` with appropriate arguments
        and returns the PDOS.

        See `~sisl.physics.electron.PDOS` for argument details.
        """
        try:
            spin = self.parent.spin
        except:
            spin = None
        return PDOS(E, self.c, self.state, self.Sk(spin=spin), distribution, spin)
