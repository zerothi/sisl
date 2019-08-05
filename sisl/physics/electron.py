"""Electron related functions and classes
=========================================

.. module:: sisl.physics.electron
   :noindex:

In sisl electronic structure calculations are relying on routines
specific for electrons. For instance density of states calculations from
electronic eigenvalues and other quantities.

This module implements the necessary tools required for calculating
DOS, PDOS, band-velocities and spin moments of non-colinear calculations.
One may also plot real-space wavefunctions.

.. autosummary::
   :toctree:

   DOS
   PDOS
   velocity
   velocity_matrix
   berry_phase
   wavefunction
   spin_moment
   spin_orbital_moment
   spin_squared


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

from functools import reduce
import numpy as np
from numpy import find_common_type
from numpy import zeros, empty
from numpy import floor, ceil
from numpy import conj, dot, ogrid
from numpy import cos, sin, pi
from numpy import int32, complex128
from numpy import add, angle, sort

from sisl import units, constant
from sisl.supercell import SuperCell
from sisl.geometry import Geometry
from sisl._indices import indices_le
from sisl.oplist import oplist
from sisl._math_small import xyz_to_spherical_cos_phi
import sisl._array as _a
from sisl.linalg import svd_destroy, eigvals_destroy
from sisl.linalg import eigh_destroy, det_destroy
from sisl.messages import info, warn, SislError, tqdm_eta
from sisl._help import dtype_complex_to_real, _range as range
from .distribution import get_distribution
from .spin import Spin
from .sparse import SparseOrbitalBZSpin
from .state import Coefficient, State, StateC


__all__ = ['DOS', 'PDOS']
__all__ += ['velocity', 'velocity_matrix']
__all__ += ['spin_moment', 'spin_orbital_moment']
__all__ += ['inv_eff_mass_tensor', 'berry_phase']
__all__ += ['wavefunction']
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
    spin_moment : spin moment for states
    spin_orbital_moment : orbital resolved spin-moment

    Returns
    -------
    numpy.ndarray
        DOS calculated at energies, has same length as `E`
    """
    if isinstance(distribution, str):
        distribution = get_distribution(distribution)

    return reduce(lambda DOS, eig: DOS + distribution(E - eig), eig, 0.)


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

    For non-colinear calculations (this includes spin-orbit calculations) the PDOS is additionally
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
       matrix is assumed. For non-colinear calculations this matrix may be halve the size of ``len(state[0, :])`` to
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
    spin_moment : spin moment for states
    spin_orbital_moment : orbital resolved spin-moment

    Returns
    -------
    numpy.ndarray
        projected DOS calculated at energies, has dimension ``(state.shape[1], len(E))``.
        For non-colinear calculations it will be ``(4, state.shape[1] // 2, len(E))``, ordered as
        indicated in the above list.
    """
    if isinstance(distribution, str):
        distribution = get_distribution(distribution)

    # Figure out whether we are dealing with a non-colinear calculation
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

    # check for non-colinear (or SO)
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

    This calculation only makes sense for non-colinear calculations.

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
       has been evaluated at.

    Notes
    -----
    This routine cannot check whether the input eigenvectors originate from a non-colinear calculation.
    If a non-polarized eigenvector is passed to this routine, the output will have no physical meaning.

    See Also
    --------
    DOS : total DOS
    PDOS : projected DOS
    spin_orbital_moment : orbital resolved spin-moment

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


def spin_orbital_moment(state, S=None):
    r""" Calculate the spin magnetic moment per orbital site (equivalent to spin-moment per orbital)

    This calculation only makes sense for non-colinear calculations.

    The returned quantities are given in this order:

    - Spin magnetic moment along :math:`x` direction
    - Spin magnetic moment along :math:`y` direction
    - Spin magnetic moment along :math:`z` direction

    These are calculated using the Pauli matrices :math:`\boldsymbol\sigma_x`, :math:`\boldsymbol\sigma_y` and :math:`\boldsymbol\sigma_z`:

    .. math::

       \mathbf{S}_{i,o}^x &= \psi_{i,o}^* | \boldsymbol\sigma_x \mathbf S | \psi_i \rangle
       \\
       \mathbf{S}_{i,o}^y &= \psi_{i,o}^* | \boldsymbol\sigma_y \mathbf S | \psi_i \rangle
       \\
       \mathbf{S}_{i,o}^z &= \psi_{i,o}^* | \boldsymbol\sigma_z \mathbf S | \psi_i \rangle

    Note that this is equivalent to `PDOS` *without* the distribution function and energy grid.

    Parameters
    ----------
    state : array_like
       vectors describing the electronic states, 2nd dimension contains the states
    S : array_like, optional
       overlap matrix used in the :math:`\mathbf S|\psi\rangle` calculation. If `None` the identity
       matrix is assumed. The overlap matrix should correspond to the system and :math:`k` point the eigenvectors
       has been evaluated at.

    Notes
    -----
    This routine cannot check whether the input eigenvectors originate from a non-colinear calculation.
    If a non-polarized eigenvector is passed to this routine, the output will have no physical meaning.

    See Also
    --------
    DOS : total DOS
    PDOS : projected DOS
    spin_moment : spin moment for states (equivalent to the ``spin_orbital_moment.sum(1)``)

    Returns
    -------
    numpy.ndarray
        spin moments per state with final dimension ``(state.shape[0], state.shape[1] // 2, 3)``.
    """
    if state.ndim == 1:
        return spin_orbital_moment(state.reshape(1, -1), S)[0]

    if S is None:
        class S(object):
            __slots__ = []
            shape = (state.shape[1] // 2, state.shape[1] // 2)
            @staticmethod
            def dot(v):
                return v

    if S.shape[1] == state.shape[1]:
        S = S[::2, ::2]

    s = np.empty([state.shape[0], state.shape[1] // 2, 3], dtype=dtype_complex_to_real(state.dtype))

    for i in range(len(state)):
        Sstate = S.dot(state[i].reshape(-1, 2))
        D = (conj(state[i]) * Sstate.ravel()).real.reshape(-1, 2)
        s[i, :, 2] = D[:, 0] - D[:, 1]
        D = 2 * conj(state[i, 1::2]) * Sstate[:, 0]
        s[i, :, 0] = D.real
        s[i, :, 1] = D.imag

    return s


def spin_squared(state_alpha, state_beta, S=None):
    r""" Calculate the spin squared expectation value between two spin states

    This calculation only makes sense for spin-polarized calculations.

    The expectation value is calculated using the following formula:

    .. math::
       S^2_{\alpha,i} &= \sum_j |\langle \psi_j^\beta | \mathbf S | \psi_i^\alpha \rangle|^2
       \\
       S^2_{\beta,j} &= \sum_i |\langle \psi_i^\alpha | \mathbf S | \psi_j^\beta \rangle|^2

    where :math:`\alpha` and :math:`\beta` are different spin-components.

    The arrays :math:`S^2_\alpha` and :math:`S^2_\beta` are returned.

    Parameters
    ----------
    state_alpha : array_like
       vectors describing the electronic states of spin-channel :math:`\alpha`, 2nd dimension contains the states
    state_beta : array_like
       vectors describing the electronic states of spin-channel :math:`\beta`, 2nd dimension contains the states
    S : array_like, optional
       overlap matrix used in the :math:`\langle\psi|\mathbf S|\psi\rangle` calculation. If `None` the identity
       matrix is assumed. The overlap matrix should correspond to the system and :math:`k` point the eigenvectors
       have been evaluated at.

    Notes
    -----
    `state_alpha` and `state_beta` need not have the same number of states.

    Returns
    -------
    ~sisl.oplist.oplist
         list of spin squared expectation value per state for spin state :math:`\alpha` and :math:`\beta`
    """
    if state_alpha.ndim == 1:
        if state_beta.ndim == 1:
            Sa, Sb = spin_squared(state_alpha.reshape(1, -1), state_beta.reshape(1, -1), S)
            return oplist((Sa[0], Sb[0]))
        return spin_squared(state_alpha.reshape(1, -1), state_beta, S)
    elif state_beta.ndim == 1:
        return spin_squared(state_alpha, state_beta.reshape(1, -1), S)

    if state_alpha.shape[1] != state_beta.shape[1]:
        raise ValueError('spin_squared requires alpha and beta states to have same number of orbitals')

    if S is None:
        class S(object):
            __slots__ = []
            shape = (state_alpha.shape[1], state_alpha.shape[1])
            @staticmethod
            def dot(v):
                return v

    n_alpha = state_alpha.shape[0]
    n_beta = state_beta.shape[0]

    if n_alpha > n_beta:
        # Loop beta...
        Sa = zeros([n_alpha], dtype=dtype_complex_to_real(state_alpha.dtype))
        Sb = empty([n_beta], dtype=Sa.dtype)

        S_state_alpha = S.dot(state_alpha.T)
        for i in range(n_beta):
            D = dot(conj(state_beta[i]), S_state_alpha)
            D *= conj(D)
            Sa += D.real
            Sb[i] = D.sum().real

    else:
        # Loop alpha...
        Sa = empty([n_alpha], dtype=dtype_complex_to_real(state_alpha.dtype))
        Sb = zeros([n_beta], dtype=Sa.dtype)

        S_state_beta = S.dot(state_beta.T)
        for i in range(n_alpha):
            D = dot(conj(state_alpha[i]), S_state_beta)
            D *= conj(D)
            Sb += D.real
            Sa[i] = D.sum().real

    return oplist((Sa, Sb))


def velocity(state, dHk, energy=None, dSk=None, degenerate=None):
    r""" Calculate the velocity of a set of states

    These are calculated using the analytic expression (:math:`\alpha` corresponding to the Cartesian directions):

    .. math::

       \mathbf{v}_{i\alpha} = \frac1\hbar \langle \psi_i |
                \frac{\partial}{\partial\mathbf k}_\alpha \mathbf H(\mathbf k) | \psi_i \rangle

    In case of non-orthogonal basis the equations substitutes :math:`\mathbf H(\mathbf k)` by
    :math:`\mathbf H(\mathbf k) - \epsilon_i\mathbf S(\mathbf k)`.

    The velocities calculated are without the Berry curvature contributions.

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
    degenerate : list of array_like, optional
       a list containing the indices of degenerate states. In that case a prior diagonalization
       is required to decouple them. This is done 3 times along each of the Cartesian directions.

    Returns
    -------
    numpy.ndarray
        velocities per state with final dimension ``(state.shape[0], 3)``, the velocity unit is Ang/ps. Units *may* change in future releases.
    """
    if state.ndim == 1:
        return velocity(state.reshape(1, -1), dHk, energy, dSk, degenerate).ravel()

    if dSk is None:
        return _velocity_ortho(state, dHk, degenerate)
    return _velocity_non_ortho(state, dHk, energy, dSk, degenerate)


# dHk is in [Ang eV]
# velocity units in [Ang/ps]
_velocity_const = 1 / constant.hbar('eV ps')


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


def velocity_matrix(state, dHk, energy=None, dSk=None, degenerate=None):
    r""" Calculate the velocity matrix of a set of states

    These are calculated using the analytic expression (:math:`\alpha` corresponding to the Cartesian directions):

    .. math::

       \mathbf{v}_{ij\alpha} = \frac1\hbar \langle \psi_j |
                \frac{\partial}{\partial\mathbf k}_\alpha \mathbf H(\mathbf k) | \psi_i \rangle

    In case of non-orthogonal basis the equations substitutes :math:`\mathbf H(\mathbf k)` by
    :math:`\mathbf H(\mathbf k) - \epsilon_i\mathbf S(\mathbf k)`.

    Although this matrix should be Hermitian it is not checked, and we explicitly calculate
    all elements.

    The velocities calculated are without the Berry curvature contributions.

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
    degenerate : list of array_like, optional
       a list containing the indices of degenerate states. In that case a prior diagonalization
       is required to decouple them. This is done 3 times along each of the Cartesian directions.

    See Also
    --------
    velocity : only calculate the diagonal components of this matrix

    Returns
    -------
    numpy.ndarray
        velocity matrixstate with final dimension ``(state.shape[0], state.shape[0], 3)``, the velocity unit is Ang/ps. Units *may* change in future releases.
    """
    if state.ndim == 1:
        return velocity_matrix(state.reshape(1, -1), dHk, energy, dSk, degenerate).ravel()

    dtype = find_common_type([state.dtype, dHk[0].dtype], [])
    if dSk is None:
        return _velocity_matrix_ortho(state, dHk, degenerate, dtype)
    return _velocity_matrix_non_ortho(state, dHk, energy, dSk, degenerate, dtype)


def _velocity_matrix_non_ortho(state, dHk, energy, dSk, degenerate, dtype):
    r""" For states in a non-orthogonal basis """

    # All matrix elements along the 3 directions
    n = state.shape[0]
    v = np.empty([n, n, 3], dtype=dtype)

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
        v[s, :, 0] = conj(state).dot((dHk[0] - e * dSk[0]).dot(state[s]))
        v[s, :, 1] = conj(state).dot((dHk[1] - e * dSk[1]).dot(state[s]))
        v[s, :, 2] = conj(state).dot((dHk[2] - e * dSk[2]).dot(state[s]))

    return v * _velocity_const


def _velocity_matrix_ortho(state, dHk, degenerate, dtype):
    r""" For states in an orthogonal basis """

    # All matrix elements along the 3 directions
    n = state.shape[0]
    v = np.empty([n, n, 3], dtype=dtype)

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

    for s in range(n):
        v[s, :, 0] = conj(state).dot(dHk[0].dot(state[s, :]))
        v[s, :, 1] = conj(state).dot(dHk[1].dot(state[s, :]))
        v[s, :, 2] = conj(state).dot(dHk[2].dot(state[s, :]))

    return v * _velocity_const


def inv_eff_mass_tensor(state, ddHk, energy=None, ddSk=None, degenerate=None, as_matrix=False):
    r""" Calculate the effective mass tensor for a set of states (missing off-diagonal terms)

    These are calculated using the analytic expression (:math:`\alpha,\beta` corresponds to Cartesian directions):

    .. math::

        \mathbf M^{-1}_{i\alpha\beta} = \frac1{\hbar^2} \langle \psi_i |
             \frac{\partial}{\partial\mathbf k}_\alpha\frac{\partial}{\partial\mathbf k}_\beta \mathbf H(\mathbf k)
             | \psi_i \rangle

    In case of non-orthogonal basis the equations substitutes :math:`\mathbf H(\mathbf k)` by
    :math:`\mathbf H(\mathbf k) - \epsilon_i\mathbf S(\mathbf k)`.

    The matrix :math:`\mathbf M` is known as the effective mass tensor, remark that this function returns the inverse
    of :math:`\mathbf M`.

    Currently this routine only returns the above quations, however, the inverse effective mass tensor
    also has contributions from some off-diagonal elements, see PRB 75 (195121) 2007.

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
    degenerate : list of array_like, optional
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


def berry_phase(contour, sub=None, eigvals=False, closed=True, method='berry'):
    r""" Calculate the Berry-phase on a loop using a predefined path

    The Berry phase for a single Bloch state is calculated using the discretized formula:

    .. math::
       \phi = - \Im\ln \mathrm{det} \prod_i^{N-1} \langle \psi_{k_i} | \psi_{k_{i+1}} \rangle

    where :math:`\langle \psi_{k_i} | \psi_{k_{i+1}} \rangle` may be exchanged with an overlap matrix
    of the investigated bands.

    Parameters
    ----------
    contour : BrillouinZone
       containing the closed contour and has the ``contour.parent`` as an instance of Hamiltonian. The
       first and last k-point must not be the same.
    sub : None or list of int, optional
       selected bands to calculate the Berry phase of
    eigvals : bool, optional
       return the eigenvalues of the product of the overlap matrices
    closed : bool, optional
       whether or not to include the connection of the last and first points in the loop
    method : {'berry', 'zak'}
       'berry' will return the usual integral of the Berry connection over the specified contour
       'zak' will compute the Zak phase for 1D systems by performing a closed loop integration but
       taking into account the Bloch factor :math:`e^{-i2\pi/a x}` accumulated over a Brillouin zone,
       see J. Zak, "Berry's phase for energy bands in solids" PRL 62, 2747 (1989).

    Notes
    -----
    The Brillouin zone object *need* not contain a closed discretized contour by doubling the first point.

    The implementation is very similar to PythTB and refer to the details outlined in PythTB for
    additional details.

    This implementation does not work for band-crossings or degenerate states. It is thus important that
    eigenstates are corresponding to the same states for the loop contained in `bz`.

    Examples
    --------

    Calculate the multi-band Berry-phase

    >>> N = 30
    >>> kR = 0.01
    >>> normal = [0, 0, 1]
    >>> origo = [1/3, 2/3, 0]
    >>> bz = BrillouinZone.param_circle(H, N, kR, normal, origo)
    >>> phase = berry_phase(bz)

    Calculate Berry-phase for first band

    >>> N = 30
    >>> kR = 0.01
    >>> normal = [0, 0, 1]
    >>> origo = [1/3, 2/3, 0]
    >>> bz = BrillouinZone.param_circle(H, N, kR, normal, origo)
    >>> phase = berry_phase(bz, sub=0)
    """
    from .hamiltonian import Hamiltonian
    # Currently we require the Berry phase calculation to *only* accept Hamiltonians
    if not isinstance(contour.parent, Hamiltonian):
        raise SislError('berry_phase: requires the Brillouin zone object to contain a Hamiltonian!')

    if not contour.parent.orthogonal:
        raise SislError('berry_phase: requires the Hamiltonian to use an orthogonal basis!')

    if np.allclose(contour.k[0, :], contour.k[-1, :]):
        # When the user has the contour points closed, we don't need to do this in the below loop
        closed = False

    method = method.lower()
    if method == 'berry':
        pass
    elif method == 'zak':
        closed = True
    else:
        raise SislError('berry_phase: requires the method to be [berry, zak]')

    # Whether we should calculate the eigenvalues of the overlap matrix
    if eigvals:
        # We calculate the final eigenvalues
        def _process(prd, ovr):
            U, _, V = svd_destroy(ovr)
            return dot(prd, dot(U, V))
    else:
        # We calculate the final angle from the determinant
        _process = dot

    if sub is None:
        def _berry(eigenstates):
            # Grab the first one to be able to form a loop
            first = next(eigenstates)
            first.change_gauge('r')
            # Create a variable to keep track of the previous state
            prev = first

            # Initialize the consecutive product
            # Starting with the identity matrix!
            prd = 1

            # Loop remaining eigenstates
            for second in eigenstates:
                second.change_gauge('r')
                prd = _process(prd, prev.inner(second, diagonal=False))
                prev = second

            # Complete the loop
            if closed:
                # Insert Bloch phase for 1D integral?
                if method == 'zak':
                    g = contour.parent.geometry
                    axis = contour.k[1] - contour.k[0]
                    axis /= (axis ** 2).sum() ** 0.5
                    phase = dot(g.xyz[g.o2a(_a.arangei(g.no)), :], dot(axis, g.rcell)).reshape(1, -1)
                    prev.state *= np.exp(-1j * phase)

                # Include last-to-first segment
                prd = _process(prd, prev.inner(first, diagonal=False))
            return prd

    else:
        def _berry(eigenstates):
            first = next(eigenstates).sub(sub)
            first.change_gauge('r')
            prev = first
            prd = 1
            for second in eigenstates:
                second = second.sub(sub)
                second.change_gauge('r')
                prd = _process(prd, prev.inner(second, diagonal=False))
                prev = second
            if closed:
                if method == 'zak':
                    g = contour.parent.geometry
                    axis = contour.k[1] - contour.k[0]
                    axis /= (axis ** 2).sum() ** 0.5
                    phase = dot(g.xyz[g.o2a(_a.arangei(g.no)), :], dot(axis, g.rcell)).reshape(1, -1)
                    prev.state *= np.exp(-1j * phase)
                prd = _process(prd, prev.inner(first, diagonal=False))
            return prd

    # Do the actual calculation of the final matrix
    d = _berry(contour.asyield().eigenstate())

    # Correct return values
    if eigvals:
        ret = -angle(eigvals_destroy(d))
        ret = sort(ret)
    else:
        ret = -angle(det_destroy(d))

    return ret


def wavefunction(v, grid, geometry=None, k=None, spinor=0, spin=None, eta=False):
    r""" Add the wave-function (`Orbital.psi`) component of each orbital to the grid

    This routine calculates the real-space wave-function components in the
    specified grid.

    This is an *in-place* operation that *adds* to the current values in the grid.

    It may be instructive to check that an eigenstate is normalized:

    >>> grid = Grid(...)
    >>> psi(state, grid)
    >>> (np.abs(grid.grid) ** 2).sum() * grid.dvolume == 1.

    Note: To calculate :math:`\psi(\mathbf r)` in a unit-cell different from the
    originating geometry, simply pass a grid with a unit-cell smaller than the originating
    supercell.

    The wavefunctions are calculated in real-space via:

    .. math::
       \psi(\mathbf r) = \sum_i\phi_i(\mathbf r) |\psi\rangle_i \exp(-i\mathbf k \mathbf R)

    While for non-colinear/spin-orbit calculations the wavefunctions are determined from the
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
       has not been created via :meth:`~.Hamiltonian.eigenstate`).
    spinor : int, optional
       the spinor for non-colinear/spin-orbit calculations. This is only used if the
       eigenstate object has been created from a parent object with a `Spin` object
       contained, *and* if the spin-configuration is non-colinear or spin-orbit coupling.
       Default to the first spinor component.
    spin : Spin, optional
       specification of the spin configuration of the orbital coefficients. This only has
       influence for non-colinear wavefunctions where `spinor` choice is important.
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
        if v.shape[0] > 1:
            info('wavefunction: summing {} different state coefficients, will continue silently!'.format(v.shape[0]))
        v = v.sum(0)

    if spin is None:
        if len(v) // 2 == geometry.no:
            # We can see from the input that the vector *must* be a non-colinear calculation
            v = v.reshape(-1, 2)[:, spinor]
            info('wavefunction assumes the input wavefunction coefficients to originate from a non-colinear calculation!')

    elif spin.kind > Spin.POLARIZED:
        # For non-colinear cases the user selects the spinor component.
        v = v.reshape(-1, 2)[:, spinor]

    if len(v) != geometry.no:
        raise ValueError("wavefunction: require wavefunction coefficients corresponding to number of orbitals in the geometry.")

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
        raise SislError("wavefunction: input coefficients are complex, while grid only contains real.")

    if is_complex:
        psi_init = _a.zerosz
    else:
        psi_init = _a.zerosd

    # Extract sub variables used throughout the loop
    shape = _a.asarrayi(grid.shape)
    dcell = grid.dcell
    ic_shape = grid.sc.icell * shape.reshape(3, 1)

    # Convert the geometry (hosting the wavefunction coefficients) coordinates into
    # grid-fractionals X grid-shape to get index-offsets in the grid for the geometry
    # supercell.
    geom_shape = dot(geometry.cell, ic_shape.T)

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
    rxyz = _a.emptyd(nrxyz)
    rxyz[..., 0] = ctheta_sphi
    rxyz[..., 1] = stheta_sphi
    rxyz[..., 2] = cphi
    # Reshape
    rxyz.shape = (-1, 3)
    idx = dot(rxyz, ic_shape.T)
    idxm = idx.min(0).reshape(1, 3)
    idxM = idx.max(0).reshape(1, 3)
    del ctheta_sphi, stheta_sphi, cphi, idx, rxyz, nrxyz

    # Fast loop (only per specie)
    origo = grid.sc.origo.reshape(1, 3)
    idx_mm = _a.emptyd([geometry.na, 2, 3])
    all_negative_R = True
    for atom, ia in geometry.atom.iter(True):
        if len(ia) == 0:
            continue
        R = atom.maxR()
        all_negative_R = all_negative_R and R < 0.

        # Now do it for all the atoms to get indices of the middle of
        # the atoms
        # The coordinates are relative to origo, so we need to shift (when writing a grid
        # it is with respect to origo)
        idx = dot(geometry.xyz[ia, :] - origo, ic_shape.T)

        # Get min-max for all atoms
        idx_mm[ia, 0, :] = idxm * R + idx
        idx_mm[ia, 1, :] = idxM * R + idx

    if all_negative_R:
        raise SislError("wavefunction: Cannot create wavefunction since no atoms have an associated basis-orbital on a real-space grid")

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
            grid.set_geometry(Geometry(xyz, geometry.atoms[ia], sc=sc))

    # Instead of looping all atoms in the supercell we find the exact atoms
    # and their supercell indices.
    add_R = _a.fulld(3, geometry.maxR())
    # Calculate the required additional vectors required to increase the fictitious
    # supercell by add_R in each direction.
    # For extremely skewed lattices this will be way too much, hence we make
    # them square.

    o = sc.toCuboid(True)
    sc = SuperCell(o._v + np.diag(2 * add_R), origo=o.origo - add_R)

    # Retrieve all atoms within the grid supercell
    # (and the neighbours that connect into the cell)
    # Note that we cannot pass the "moved" origo because then ISC would be wrong
    IA, XYZ, ISC = geometry.within_inf(sc, periodic=pbc)
    # We need to revert the grid supercell origo as that is not subtracted in the `within_inf` returned
    # coordinates (and the below loop expects positions with respect to the origo of the plotting
    # grid).
    XYZ -= grid.sc.origo.reshape(1, 3)

    phk = k * 2 * np.pi
    phase = 1

    # Retrieve progressbar
    eta = tqdm_eta(len(IA), 'wavefunction', 'atom', eta)

    # Loop over all atoms in the grid-cell
    for ia, xyz, isc in zip(IA, XYZ, ISC):
        # Get current atom
        atom = geometry.atoms[ia]

        # Extract maximum R
        R = atom.maxR()
        if R <= 0.:
            warn("wavefunction: Atom '{}' does not have a wave-function, skipping atom.".format(atom))
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
            phase = np.exp(-1j * (phk * isc).sum())

        # Allocate a temporary array where we add the psi elements
        psi = psi_init(n)

        # Loop on orbitals on this atom, grouped by radius
        for os in atom.iter(True):

            # Get the radius of orbitals (os)
            oR = os[0].R

            if oR <= 0.:
                warn("wavefunction: Orbital(s) '{}' does not have a wave-function, skipping orbital!".format(os))
                # Skip these orbitals
                io += len(os)
                continue

            # Downsize to the correct indices
            if R - oR < 1e-6:
                idx1 = idx
                r1 = r
                theta1 = theta
                phi1 = phi
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
        """ Internal routine to check whether this is a non-colinear calculation """
        try:
            return self.parent.spin.has_noncolinear
        except:
            return False

    def Sk(self, format='csr', spin=None):
        r""" Retrieve the overlap matrix corresponding to the originating parent structure.

        When ``self.parent`` is a Hamiltonian this will return :math:`\mathbf S(k)` for the
        :math:`k`-point these eigenstates originate from

        Parameters
        ----------
        format : str, optional
           the returned format of the overlap matrix. This only takes effect for
           non-orthogonal parents.
        spin : Spin, optional
           for non-colinear spin configurations the *fake* overlap matrix returned
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

        if self.__is_nc():
            n = self.shape[1] // 2
        else:
            n = self.shape[1]

        class __FakeSk(object):
            """ Replacement object which superseedes a matrix """
            __slots__ = []
            shape = (n, n)
            @staticmethod
            def dot(v):
                return v
            @property
            def T(self):
                return self

        return __FakeSk

    def norm2(self, sum=True):
        r""" Return a vector with the norm of each state :math:`\langle\psi|\psi\rangle`

        This will take into account any overlap matrix if present.

        Parameters
        ----------
        sum : bool, optional
           for true only a single number per state will be returned, otherwise the norm
           per basis element will be returned.

        Returns
        -------
        numpy.ndarray
            the squared norm for each state
        """
        if sum:
            return self.inner()

        # Retrieve the overlap matrix (FULL S is required for NC)
        S = self.Sk()
        return conj(self.state) * S.dot(self.state.T).T

    def inner(self, right=None, diagonal=True, align=False):
        r""" Return the inner product by :math:`\mathbf M_{ij} = \langle\psi_i|\psi'_j\rangle`

        Parameters
        ----------
        right : State, optional
           the right object to calculate the inner product with, if not passed it will do the inner
           product with itself. This object will always be the left :math:`\langle\psi_i|`.
        diagonal : bool, optional
           only return the diagonal matrix :math:`\mathbf M_{ii}`.
        align : bool, optional
           first align `right` with the angles for this state (see `align`)

        Raises
        ------
        ValueError : in case where `right` is not None and `self` and `right` has differing overlap matrix.

        Returns
        -------
        numpy.ndarray
            a matrix with the sum of inner state products
        """
        # Retrieve the overlap matrix (FULL S is required for NC)
        S = self.Sk()

        # TODO, perhaps check that it is correct... and fix multiple transposes
        if right is None:
            if diagonal:
                return (conj(self.state) * S.dot(self.state.T).T).sum(1).real
            return dot(conj(self.state), S.dot(self.state.T))

        else:
            if 'FakeSk' in S.__class__.__name__:
                raise NotImplementedError(self.__class__.__name__ + '.inner does not implement the inner product between two different overlap matrices.')

            # Same as State.inner
            # In the current implementation we require no overlap matrix!
            if align:
                if self.shape[0] != right.shape[0]:
                    raise ValueError(self.__class__.__name__ + '.inner with align=True requires exactly the same shape!')
                # Align the states
                right = self.align_phase(right, copy=False)

            if diagonal:
                if self.shape[0] != right.shape[0]:
                    return np.diag(dot(conj(self.state), S.dot(right.state.T)))
                return (conj(self.state) * S.dot(right.state.T).T).sum(1)
            return dot(conj(self.state), S.dot(right.state.T))

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

    def spin_orbital_moment(self):
        r""" Calculate spin moment per orbital from the states

        This routine calls `~sisl.physics.electron.spin_orbital_moment` with appropriate arguments
        and returns the spin moment for each orbital on the states.

        See `~sisl.physics.electron.spin_orbital_moment` for details.
        """
        try:
            spin = self.parent.spin
        except:
            spin = None
        return spin_orbital_moment(self.state, self.Sk(spin=spin))

    def expectation(self, A, diag=True):
        r""" Calculate the expectation value of matrix `A`

        The expectation matrix is calculated as:

        .. math::
            A_{ij} = \langle \psi_i | \mathbf A | \psi_j \rangle

        If `diag` is true, only the diagonal elements are returned.

        Parameters
        ----------
        A : array_like
           a vector or matrix that expresses the operator `A`
        diag : bool, optional
           whether only the diagonal elements are calculated or if the full expectation
           matrix is calculated

        Returns
        -------
        numpy.ndarray
            a vector if `diag` is true, otherwise a matrix with expectation values
        """
        ndim = A.ndim
        n = len(self)
        if diag:
            a = _a.emptyz(n)
        else:
            a = _a.emptyz([n, n])

        s = self.state
        if ndim == 1 and diag:
            for i in range(n):
                a[i] = (s[i, :].conj() * A * s[i, :]).sum()
        elif ndim == 2 and diag:
            for i in range(n):
                a[i] = (s[i, :].conj() * A.dot(s[i, :])).sum()
        elif ndim == 1:
            for i in range(n):
                a[i, :] = s.conj().dot((A * s[i, :]).T)
        elif ndim == 2:
            for i in range(n):
                a[i, :] = s.conj().dot(A.dot(s[i, :]))
        return a

    def wavefunction(self, grid, spinor=0, eta=False):
        r""" Expand the coefficients as the wavefunction on `grid` *as-is*

        See `~sisl.physics.electron.wavefunction` for argument details, the arguments not present
        in this method are automatically passed from this object.
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

        wavefunction(self.state, grid, geometry=geometry, k=k, spinor=spinor, spin=spin, eta=eta)

    def change_gauge(self, gauge):
        r""" In-place change of the gauge of the state coefficients

        The two gauges are related through:

        .. math::

            \tilde C_j = e^{i\mathbf k\mathbf r_j} C_j

        where :math:`C_j` belongs to the gauge ``R`` and :math:`\tilde C_j` is in the gauge
        ``r``.

        Parameters
        ----------
        gauge : {'R', 'r'}
            specify the new gauge for the state coefficients
        """
        # These calls will fail if the gauge is not specified.
        # In that case it will not do anything
        if self.info.get('gauge', gauge) == gauge:
            # Quick return
            return

        # Update gauge value
        self.info['gauge'] = gauge

        # Check that we can do a gauge transformation
        k = _a.asarrayd(self.info.get('k'))
        if (k ** 2).sum() <= 0.000000001:
            return

        g = self.parent.geometry
        phase = dot(g.xyz[g.o2a(_a.arangei(g.no)), :], dot(k, g.rcell))

        if gauge == 'r':
            self.state *= np.exp(1j * phase).reshape(1, -1)
        elif gauge == 'R':
            self.state *= np.exp(-1j * phase).reshape(1, -1)


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

    def velocity_matrix(self, eps=1e-4):
        r""" Calculate velocity matrix for the states

        This routine calls `~sisl.physics.electron.velocity_matrix` with appropriate arguments
        and returns the velocity for the states. I.e. for non-orthogonal basis the overlap
        matrix and energy values are also passed.

        Note that the coefficients associated with the `StateCElectron` *must* correspond
        to the energies of the states.

        See `~sisl.physics.electron.velocity_matrix` for details.

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
            raise SislError(self.__class__.__name__ + '.velocity_matrix requires the parent to have a spin associated.')
        return velocity_matrix(self.state, self.parent.dHk(**opt), self.c, dSk, degenerate=deg)

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
            # Ensure we are dealing with the r gauge
            self.change_gauge('r')

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
        """ Eigenvalues """
        return self.c

    def occupation(self, distribution='fermi_dirac'):
        """ Calculate the occupations for the states according to a distribution function

        Parameters
        ----------
        distribution : str or func, optional
           distribution used to find occupations

        Returns
        -------
        numpy.ndarray
             ``len(self)`` with occupation values
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
        """ Eigenvalues for each state """
        return self.c

    def occupation(self, distribution='fermi_dirac'):
        """ Calculate the occupations for the states according to a distribution function

        Parameters
        ----------
        distribution : str or func, optional
           distribution used to find occupations

        Returns
        -------
        numpy.ndarray
             ``len(self)`` with occupation values
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
