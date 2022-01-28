# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
r"""Electron related functions and classes
==========================================

In sisl electronic structure calculations are relying on routines
specific for electrons. For instance density of states calculations from
electronic eigenvalues and other quantities.

This module implements the necessary tools required for calculating
DOS, PDOS, band-velocities and spin moments of non-colinear calculations.
One may also plot real-space wavefunctions.

   DOS
   PDOS
   COP
   velocity
   velocity_matrix
   berry_phase
   berry_curvature
   conductivity
   wavefunction
   spin_moment
   spin_squared


Supporting classes
------------------

Certain classes aid in the usage of the above methods by implementing them
using automatic arguments.

For instance, the PDOS method requires the overlap matrix in non-orthogonal
basis sets at the :math:`k`-point corresponding to the eigenstates. Hence, the
argument ``S`` must be :math:`\mathbf S(\mathbf k)`. The `EigenstateElectron` class
automatically passes the correct ``S`` because it knows the states :math:`k`-point.

   CoefficientElectron
   StateElectron
   StateCElectron
   EigenvalueElectron
   EigenvectorElectron
   EigenstateElectron

"""

from functools import reduce
import numpy as np
from numpy import find_common_type
from numpy import zeros, empty
from numpy import floor, ceil
from numpy import conj, dot, ogrid, einsum
from numpy import cos, sin, log, exp, pi
from numpy import int32, complex128
from numpy import add, argsort, sort
from scipy.sparse import isspmatrix

from sisl._internal import set_module
from sisl import units, constant
from sisl.supercell import SuperCell
from sisl.geometry import Geometry
from sisl._indices import indices_le
from sisl.oplist import oplist
from sisl._math_small import xyz_to_spherical_cos_phi
import sisl._array as _a
from sisl.linalg import svd_destroy, eigvals_destroy
from sisl.linalg import eigh, det_destroy, sqrth
from sisl.messages import info, warn, SislError, progressbar, deprecate_method
from sisl._help import dtype_complex_to_real, dtype_real_to_complex
from .distribution import get_distribution
from .spin import Spin
from .sparse import SparseOrbitalBZSpin
from .state import degenerate_decouple, Coefficient, State, StateC, _FakeMatrix


__all__ = ['DOS', 'PDOS', 'COP']
__all__ += ['velocity', 'velocity_matrix']
__all__ += ['spin_moment', 'spin_squared']
__all__ += ['berry_phase', 'berry_curvature']
__all__ += ['conductivity']
__all__ += ['wavefunction']
__all__ += ['CoefficientElectron', 'StateElectron', 'StateCElectron']
__all__ += ['EigenvalueElectron', 'EigenvectorElectron', 'EigenstateElectron']


@set_module("sisl.physics.electron")
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
    COP : calculate COOP or COHP curves
    PDOS : projected DOS (same as this, but projected onto each orbital)
    spin_moment : spin moment

    Returns
    -------
    numpy.ndarray
        DOS calculated at energies, has same length as `E`
    """
    if isinstance(distribution, str):
        distribution = get_distribution(distribution)

    return reduce(lambda DOS, eig: DOS + distribution(E - eig), eig, 0.)


@set_module("sisl.physics.electron")
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

       \mathrm{PDOS}_\nu^\sigma(E) &= \sum_i \psi^*_{i,\nu} \boldsymbol\sigma_z \boldsymbol\sigma_z [\mathbf S | \psi_{i}\rangle]_\nu D(E-\epsilon_i)
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
    COP : calculate COOP or COHP curves
    spin_moment : spin moment

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
        class S:
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
        cs = conj(state[0]).reshape(-1, 2)
        v = S.dot(state[0].reshape(-1, 2))
        D1 = (cs * v).real # uu,dd PDOS
        PDOS[0, :, :] = D1.sum(1).reshape(-1, 1) * d # total DOS
        PDOS[3, :, :] = (D1[:, 0] - D1[:, 1]).reshape(-1, 1) * d # z-dos
        D1 = (cs[:, 1] * v[:, 0]).reshape(-1, 1)
        D2 = (cs[:, 0] * v[:, 1]).reshape(-1, 1)
        PDOS[1, :, :] = (D1.real + D2.real) * d # x-dos
        PDOS[2, :, :] = (D1.imag - D2.imag) * d # y-dos
        for i in range(1, len(eig)):
            d = distribution(E - eig[i]).reshape(1, -1)
            cs = conj(state[i]).reshape(-1, 2)
            v = S.dot(state[i].reshape(-1, 2))
            D1 = (cs * v).real
            PDOS[0, :, :] += D1.sum(1).reshape(-1, 1) * d
            PDOS[3, :, :] += (D1[:, 0] - D1[:, 1]).reshape(-1, 1) * d
            D1 = (cs[:, 1] * v[:, 0]).reshape(-1, 1)
            D2 = (cs[:, 0] * v[:, 1]).reshape(-1, 1)
            PDOS[1, :, :] += (D1.real + D2.real) * d
            PDOS[2, :, :] += (D1.imag - D2.imag) * d

    else:
        PDOS = (conj(state[0]) * S.dot(state[0])).real.reshape(-1, 1) \
               * distribution(E - eig[0]).reshape(1, -1)
        for i in range(1, len(eig)):
            PDOS[:, :] += (conj(state[i]) * S.dot(state[i])).real.reshape(-1, 1) \
                          * distribution(E - eig[i]).reshape(1, -1)

    return PDOS


@set_module("sisl.physics.electron")
def COP(E, eig, state, M, distribution='gaussian'):
    r""" Calculate the Crystal Orbital Population for a set of energies, `E`, with a distribution function

    The :math:`\mathrm{COP}(E)` is calculated as:

    .. math::
       \mathrm{COP}_{\nu,\mu}(E) = \sum_i \psi^*_{i,\nu}\psi_{i,\mu} \mathbf M e^{i\mathbf k\cdot \mathbf R} D(E-\epsilon_i)

    where :math:`D(\Delta E)` is the distribution function used. Note that the distribution function
    used may be a user-defined function. Alternatively a distribution function may
    be aquired from `~sisl.physics.distribution`.

    The COP curves generally refers to COOP or COHP curves.
    COOP is the Crystal Orbital Overlap Population with `M` being the overlap matrix.
    COHP is the Crystal Orbital Hamiltonian Population with `M` being the Hamiltonian.

    Parameters
    ----------
    E : array_like
       energies to calculate the COP from
    eig : array_like
       eigenvalues
    state : array_like
       eigenvectors
    M : array_like
       matrix used in the COP curve.
    distribution : func or str, optional
       a function that accepts :math:`E-\epsilon` as argument and calculates the
       distribution function.

    Notes
    -----
    This is not tested for non-collinear states.
    This requires substantial amounts of memory for big systems with lots of energy points.

    This method is considered experimental and implementation may change in the future.

    See Also
    --------
    sisl.physics.distribution : a selected set of implemented distribution functions
    DOS : total DOS
    PDOS : projected DOS over all orbitals
    spin_moment : spin moment

    Returns
    -------
    ~sisl.oplist.oplist
        COP calculated at energies, has dimension ``(len(E), *M.shape)``.
    """
    if isinstance(distribution, str):
        distribution = get_distribution(distribution)

    assert len(eig) == len(state), "COP: number of eigenvalues and states are not consistent"

    n_s = M.shape[1] // M.shape[0]

    def calc_cop(M, state, n_s):
        state = np.tile(np.outer(state.conj(), state), n_s)
        return M.multiply(state).real

    # now calculate the COP curves for the different energies
    cop = oplist([0.] * len(E))
    if isspmatrix(M):
        for e, s in zip(eig, state):
            # calculate contribution from this state
            we = distribution(E - e)
            tmp = calc_cop(M, s, n_s)
            cop += [tmp.multiply(w) for w in we]
    else:
        for e, s in zip(eig, state):
            we = distribution(E - e)
            cop += we.reshape(-1, 1, 1) * calc_cop(M, s, n_s)

    return cop


@set_module("sisl.physics.electron")
def spin_moment(state, S=None, project=False):
    r""" Spin magnetic moment (spin texture) and optionally orbitally resolved moments

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

    If `project` is true, the above will be the orbitally resolved quantities.

    Parameters
    ----------
    state : array_like
       vectors describing the electronic states, 2nd dimension contains the states
    S : array_like, optional
       overlap matrix used in the :math:`\langle\psi|\mathbf S|\psi\rangle` calculation. If `None` the identity
       matrix is assumed. The overlap matrix should correspond to the system and :math:`k` point the eigenvectors
       has been evaluated at.
    project: bool, optional
       whether the spin-moments will be orbitally resolved or not

    Notes
    -----
    This routine cannot check whether the input eigenvectors originate from a non-colinear calculation.
    If a non-polarized eigenvector is passed to this routine, the output will have no physical meaning.

    See Also
    --------
    DOS : total DOS
    PDOS : projected DOS
    COP : calculate COOP or COHP curves

    Returns
    -------
    numpy.ndarray
        spin moments per state with final dimension ``(state.shape[0], 3)``, or ``(state.shape[0], state.shape[1]//2, 3)`` if project is true
    """
    if state.ndim == 1:
        return spin_moment(state.reshape(1, -1), S, project)[0]

    if S is None:
        class S:
            __slots__ = []
            shape = (state.shape[1] // 2, state.shape[1] // 2)
            @staticmethod
            def dot(v):
                return v

    if S.shape[1] == state.shape[1]:
        S = S[::2, ::2]

    if project:
        s = np.empty([state.shape[0], state.shape[1] // 2, 3], dtype=dtype_complex_to_real(state.dtype))

        for i in range(len(state)):
            cs = conj(state[i]).reshape(-1, 2)
            Sstate = S.dot(state[i].reshape(-1, 2))
            D1 = (cs * Sstate).real
            s[i, :, 2] = D1[:, 0] - D1[:, 1]
            D1 = cs[:, 1] * Sstate[:, 0]
            D2 = cs[:, 0] * Sstate[:, 1]
            s[i, :, 0] = D1.real + D2.real
            s[i, :, 1] = D1.imag - D2.imag

    else:
        s = np.empty([state.shape[0], 3], dtype=dtype_complex_to_real(state.dtype))

        # TODO consider doing this all in a few lines
        # TODO Since there are no energy dependencies here we can actually do all
        # TODO dot products in one go and then use b-casting rules. Should be much faster
        # TODO but also way more memory demanding!
        for i in range(len(state)):
            cs = conj(state[i]).reshape(-1, 2)
            Sstate = S.dot(state[i].reshape(-1, 2))
            D = cs.T @ Sstate
            s[i, 2] = (D[0, 0] - D[1, 1]).real
            s[i, 0] = (D[1, 0] + D[0, 1]).real
            s[i, 1] = (D[1, 0] - D[0, 1]).imag

    return s


@set_module("sisl.physics.electron")
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
        raise ValueError("spin_squared requires alpha and beta states to have same number of orbitals")

    if S is None:
        class S:
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


@set_module("sisl.physics.electron")
def velocity(state, dHk, energy=None, dSk=None, degenerate=None, degenerate_dir=(1, 1, 1), project=False):
    r""" Calculate the velocity of a set of states

    These are calculated using the analytic expression (:math:`\alpha` corresponding to the Cartesian directions):

    .. math::

       \mathbf{v}^\alpha_{i} = \frac1\hbar \langle \psi_i |
                \frac{\partial}{\partial\mathbf k_\alpha} \mathbf H(\mathbf k) | \psi_i \rangle

    In case of non-orthogonal basis the equations substitutes :math:`\mathbf H(\mathbf k)` by
    :math:`\mathbf H(\mathbf k) - \epsilon_i\mathbf S(\mathbf k)`.

    In case the user requests to project the velocities (`project` is True) the equation follows that
    of `PDOS` with the same changes.
    In case of non-colinear spin the velocities will be returned also for each spin-direction.

    Notes
    -----
    The velocities are calculated without the Berry curvature contribution see Eq. (2) in [1]_.
    The missing contribution may be added in later editions, for completeness sake, it is:

    .. math::
        \delta \mathbf v = - \mathbf k\times \Omega_i(\mathbf k)

    where :math:`\Omega_i` is the Berry curvature for state :math:`i`.

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
       is required to decouple them. See `degenerate_dir` for the sum of directions.
    degenerate_dir : (3,), optional
       a direction used for degenerate decoupling. The decoupling based on the velocity along this direction
    project : bool, optional
       whether the velocities will be returned projected per orbital

    See Also
    --------
    Hamiltonian.dHk : function for generating the Hamiltonian derivatives (`dHk` argument)
    Hamiltonian.dSk : function for generating the Hamiltonian derivatives (`dSk` argument)

    References
    ----------
    .. [1] :doi:`X. Wang, J. R. Yates, I. Souza, D. Vanderbilt, "Ab initio calculation of the anomalous Hall conductivity by Wannier interpolation", PRB **74**, 195118 (2006) <10.1103/PhysRevB.74.195118>`

    Returns
    -------
    numpy.ndarray
        if `project` is false, velocities per state with final dimension ``(state.shape[0], 3)``, the velocity unit is Ang/ps. Units *may* change in future releases.
    numpy.ndarray
        if `project` is true, velocities per state with final dimension ``(state.shape[0], state.shape[1], 3)``, the velocity unit is Ang/ps. Units *may* change in future releases.
    """
    if state.ndim == 1:
        return velocity(state.reshape(1, -1), dHk, energy, dSk, degenerate, degenerate_dir, project)[0]

    if dSk is None:
        return _velocity_ortho(state, dHk, degenerate, degenerate_dir, project)
    return _velocity_non_ortho(state, dHk, energy, dSk, degenerate, degenerate_dir, project)


# dHk is in [Ang eV]
# velocity units in [Ang/ps]
_velocity_const = 1 / constant.hbar('eV ps')


def _velocity_non_ortho(state, dHk, energy, dSk, degenerate, degenerate_dir, project):
    r""" For states in a non-orthogonal basis """
    # Decouple the degenerate states
    if not degenerate is None:
        degenerate_dir = _a.asarrayd(degenerate_dir)
        degenerate_dir /= (degenerate_dir ** 2).sum() ** 0.5
        deg_dHk = sum(d*dh for d, dh in zip(degenerate_dir, dHk))
        for deg in degenerate:
            # Set the average energy
            e = np.average(energy[deg])
            energy[deg] = e

            # Now diagonalize to find the contributions from individual states
            # then re-construct the seperated degenerate states
            # Since we do this for all directions we should decouple them all
            state[deg] = degenerate_decouple(state[deg], deg_dHk - sum(d * e * ds for d, ds in zip(degenerate_dir, dSk)))
        del deg_dHk

    if project:
        v = np.empty([state.shape[0], state.shape[1], 3], dtype=dtype_complex_to_real(state.dtype))
        # Since they depend on the state energies and dSk we have to loop them individually.
        for s, e in enumerate(energy):
            cs = conj(state[s])
            # Since dHk *may* be a csr_matrix or sparse, we have to do it like
            # this. A sparse matrix cannot be re-shaped with an extra dimension.
            v[s, :, 0] = (cs * (dHk[0] - e * dSk[0]).dot(state[s])).real
            v[s, :, 1] = (cs * (dHk[1] - e * dSk[1]).dot(state[s])).real
            v[s, :, 2] = (cs * (dHk[2] - e * dSk[2]).dot(state[s])).real

    else:
        v = np.empty([state.shape[0], 3], dtype=dtype_complex_to_real(state.dtype))
        for s, e in enumerate(energy):
            cs = conj(state[s])
            v[s, 0] = cs.dot((dHk[0] - e * dSk[0]).dot(state[s])).real
            v[s, 1] = cs.dot((dHk[1] - e * dSk[1]).dot(state[s])).real
            v[s, 2] = cs.dot((dHk[2] - e * dSk[2]).dot(state[s])).real

    return v * _velocity_const


def _velocity_ortho(state, dHk, degenerate, degenerate_dir, project):
    r""" For states in an orthogonal basis """
    # Decouple the degenerate states
    if not degenerate is None:
        degenerate_dir = _a.asarrayd(degenerate_dir)
        degenerate_dir /= (degenerate_dir ** 2).sum() ** 0.5
        deg_dHk = sum(d*dh for d, dh in zip(degenerate_dir, dHk))
        for deg in degenerate:
            # Now diagonalize to find the contributions from individual states
            # then re-construct the seperated degenerate states
            # Since we do this for all directions we should decouple them all
            state[deg] = degenerate_decouple(state[deg], deg_dHk)
        del deg_dHk

    cs = conj(state)
    if project:
        v = np.empty([state.shape[0], state.shape[1], 3], dtype=dtype_complex_to_real(state.dtype))

        v[:, :, 0] = (cs * dHk[0].dot(state.T).T).real
        v[:, :, 1] = (cs * dHk[1].dot(state.T).T).real
        v[:, :, 2] = (cs * dHk[2].dot(state.T).T).real

    else:
        v = np.empty([state.shape[0], 3], dtype=dtype_complex_to_real(state.dtype))

        v[:, 0] = einsum('ij,ji->i', cs, dHk[0].dot(state.T)).real
        v[:, 1] = einsum('ij,ji->i', cs, dHk[1].dot(state.T)).real
        v[:, 2] = einsum('ij,ji->i', cs, dHk[2].dot(state.T)).real

    return v * _velocity_const


@set_module("sisl.physics.electron")
def velocity_matrix(state, dHk, energy=None, dSk=None, degenerate=None, degenerate_dir=(1, 1, 1)):
    r""" Calculate the velocity matrix of a set of states

    These are calculated using the analytic expression (:math:`\alpha` corresponding to the Cartesian directions):

    .. math::

       \mathbf{v}^\alpha_{ij} = \frac1\hbar \langle \psi_j |
                \frac{\partial}{\partial\mathbf k_\alpha} \mathbf H(\mathbf k) | \psi_i \rangle

    In case of non-orthogonal basis the equations substitutes :math:`\mathbf H(\mathbf k)` by
    :math:`\mathbf H(\mathbf k) - \epsilon_i\mathbf S(\mathbf k)`.

    Although the matrix :math:`\mathbf v` should be Hermitian it is not checked, and we explicitly calculate
    all elements.

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
       is required to decouple them. See `degenerate_dir` for details.
    degenerate_dir : (3,), optional
       a direction used for degenerate decoupling. The decoupling based on the velocity along this direction


    See Also
    --------
    velocity : only calculate the diagonal components of this matrix
    Hamiltonian.dHk : function for generating the Hamiltonian derivatives (`dHk` argument)
    Hamiltonian.dSk : function for generating the Hamiltonian derivatives (`dSk` argument)

    Notes
    -----
    The velocities are calculated without the Berry curvature contribution see Eq. (2) in [1]_.
    The missing contribution may be added in later editions, for completeness sake, it is:

    .. math::
        \delta \mathbf v = - \mathbf k\times \Omega_i(\mathbf k)

    where :math:`\Omega_i` is the Berry curvature for state :math:`i`.

    References
    ----------
    .. [1] :doi:`X. Wang, J. R. Yates, I. Souza, D. Vanderbilt, "Ab initio calculation of the anomalous Hall conductivity by Wannier interpolation", PRB **74**, 195118 (2006) <10.1103/PhysRevB.74.195118>`

    Returns
    -------
    numpy.ndarray
        velocity matrix state with final dimension ``(state.shape[0], state.shape[0], 3)``, the velocity unit is Ang/ps. Units *may* change in future releases.
    """
    if state.ndim == 1:
        return velocity_matrix(state.reshape(1, -1), dHk, energy, dSk, degenerate, degenerate_dir)

    dtype = find_common_type([state.dtype, dHk[0].dtype, dtype_real_to_complex(state.dtype)], [])
    if dSk is None:
        return _velocity_matrix_ortho(state, dHk, degenerate, degenerate_dir, dtype)
    return _velocity_matrix_non_ortho(state, dHk, energy, dSk, degenerate, degenerate_dir, dtype)


def _velocity_matrix_non_ortho(state, dHk, energy, dSk, degenerate, degenerate_dir, dtype):
    r""" For states in a non-orthogonal basis """

    # All matrix elements along the 3 directions
    n = state.shape[0]
    v = np.empty([n, n, 3], dtype=dtype)

    # Decouple the degenerate states
    if not degenerate is None:
        degenerate_dir = _a.asarrayd(degenerate_dir)
        degenerate_dir /= (degenerate_dir ** 2).sum() ** 0.5
        deg_dHk = sum(d*dh for d, dh in zip(degenerate_dir, dHk))
        for deg in degenerate:
            # Set the average energy
            e = np.average(energy[deg])
            energy[deg] = e

            # Now diagonalize to find the contributions from individual states
            # then re-construct the seperated degenerate states
            # Since we do this for all directions we should decouple them all
            state[deg] = degenerate_decouple(state[deg], deg_dHk - sum(d * e * ds for d, ds in zip(degenerate_dir, dSk)))
        del deg_dHk

    # Since they depend on the state energies and dSk we have to loop them individually.
    cs = conj(state)
    for s, e in enumerate(energy):

        # Since dHk *may* be a csr_matrix or sparse, we have to do it like
        # this. A sparse matrix cannot be re-shaped with an extra dimension.
        v[s, :, 0] = cs.dot((dHk[0] - e * dSk[0]).dot(state[s]))
        v[s, :, 1] = cs.dot((dHk[1] - e * dSk[1]).dot(state[s]))
        v[s, :, 2] = cs.dot((dHk[2] - e * dSk[2]).dot(state[s]))

    return v * _velocity_const


def _velocity_matrix_ortho(state, dHk, degenerate, degenerate_dir, dtype):
    r""" For states in an orthogonal basis """

    # All matrix elements along the 3 directions
    n = state.shape[0]
    v = np.empty([n, n, 3], dtype=dtype)

    # Decouple the degenerate states
    if not degenerate is None:
        degenerate_dir = _a.asarrayd(degenerate_dir)
        degenerate_dir /= (degenerate_dir ** 2).sum() ** 0.5
        deg_dHk = sum(d*dh for d, dh in zip(degenerate_dir, dHk))
        for deg in degenerate:
            # Now diagonalize to find the contributions from individual states
            # then re-construct the seperated degenerate states
            # Since we do this for all directions we should decouple them all
            state[deg] = degenerate_decouple(state[deg], deg_dHk)
        del deg_dHk

    cs = conj(state)
    for s in range(n):
        v[s, :, 0] = cs.dot(dHk[0].dot(state[s, :]))
        v[s, :, 1] = cs.dot(dHk[1].dot(state[s, :]))
        v[s, :, 2] = cs.dot(dHk[2].dot(state[s, :]))

    return v * _velocity_const


@set_module("sisl.physics.electron")
def berry_curvature(state, energy, dHk, dSk=None, degenerate=None, degenerate_dir=(1, 1, 1)):
    r""" Calculate the Berry curvature matrix for a set of states (using Kubo)

    The Berry curvature is calculated using the following expression
    (:math:`\alpha`, :math:`\beta` corresponding to Cartesian directions):

    .. math::

       \boldsymbol\Omega_{i,\alpha\beta} = - \frac2{\hbar^2}\Im\sum_{j\neq i}
                \frac{v^\alpha_{ij} v^\beta_{ji}}
                     {[\epsilon_j - \epsilon_i]^2}

    For details see Eq. (11) in [1]_ or Eq. (2.59) in [2]_.

    Parameters
    ----------
    state : array_like
       vectors describing the electronic states, 2nd dimension contains the states. In case of degenerate
       states the vectors *may* be rotated upon return.
    energy : array_like, optional
       energies of the states. In case of degenerate
       states the eigenvalues of the states will be averaged in the degenerate sub-space.
    dHk : list of array_like
       Hamiltonian derivative with respect to :math:`\mathbf k`. This needs to be a tuple or
       list of the Hamiltonian derivative along the 3 Cartesian directions.
    dSk : list of array_like, optional
       :math:`\delta \mathbf S_k` matrix required for non-orthogonal basis.
       Same derivative as `dHk`.
       NOTE: Using non-orthogonal basis sets are not tested.
    degenerate : list of array_like, optional
       a list containing the indices of degenerate states. In that case a prior diagonalization
       is required to decouple them.
    degenerate_dir : (3,), optional
       along which direction degenerate states are decoupled.

    See Also
    --------
    velocity : calculate state velocities
    velocity_matrix : calculate state velocities between all states
    Hamiltonian.dHk : function for generating the Hamiltonian derivatives (`dHk` argument)
    Hamiltonian.dSk : function for generating the Hamiltonian derivatives (`dSk` argument)

    References
    ----------
    .. [1] :doi:`X. Wang, J. R. Yates, I. Souza, D. Vanderbilt, "Ab initio calculation of the anomalous Hall conductivity by Wannier interpolation", PRB **74**, 195118 (2006) <10.1103/PhysRevB.74.195118>`
    .. [2] :doi:`J. K. Asboth, L. Oroslany, A. Palyi, "A Short Course on Topological Insulators", arXiv *1509.02295* (2015) <10.1007/978-3-319-25607-8>`

    Returns
    -------
    numpy.ndarray
        Berry flux with final dimension ``(state.shape[0], 3, 3)``
    """
    if state.ndim == 1:
        return berry_curvature(state.reshape(1, -1), energy, dHk, dSk, degenerate, degenerate_dir)[0]

    dtype = find_common_type([state.dtype, dHk[0].dtype, dtype_real_to_complex(state.dtype)], [])
    if dSk is None:
        v_matrix = _velocity_matrix_ortho(state, dHk, degenerate, degenerate_dir, dtype)
    else:
        v_matrix = _velocity_matrix_non_ortho(state, dHk, energy, dSk, degenerate, degenerate_dir, dtype)
        warn("berry_curvature calculation for non-orthogonal basis sets are not tested! Do not expect this to be correct!")
    return _berry_curvature(v_matrix, energy)


# This reverses the velocity unit (squared since Berry curvature is v.v)
_berry_curvature_const = 1 / _velocity_const ** 2


def _berry_curvature(v_M, energy):
    r""" Calculate Berry curvature for a given velocity matrix """

    # All matrix elements along the 3 directions
    N = v_M.shape[0]
    # For cases where all states are degenerate then we would not be able
    # to calculate anything. Hence we need to initialize as zero
    # This is a vector of matrices
    #   \Omega_{n, \alpha \beta}
    sigma = np.zeros([N, 3, 3], dtype=dtype_complex_to_real(v_M.dtype))

    for s, e in enumerate(energy):
        de = (energy - e) ** 2
        # add factor 2 here, but omit the minus sign until later
        # where we are forced to use the constant upon return anyways
        np.divide(2, de, where=(de != 0), out=de)

        # Calculate the berry-curvature
        sigma[s] = ((de.reshape(-1, 1) * v_M[s]).T @ v_M[:, s]).imag

    # negative here
    return sigma * (- _berry_curvature_const)


@set_module("sisl.physics.electron")
def conductivity(bz, distribution='fermi-dirac', method='ahc', degenerate=1.e-5, degenerate_dir=(1, 1, 1)):
    r""" Electronic conductivity for a given `BrillouinZone` integral

    Currently the *only* implemented method is the anomalous Hall conductivity (AHC[1]_)
    which may be calculated as:

    .. math::
       \sigma_{\alpha\beta} = \frac{-e^2}{\hbar}\int\,\mathrm d\mathbf k\sum_i f_i\Omega_{i,\alpha\beta}(\mathbf k)

    where :math:`\Omega_{i,\alpha\beta}` and :math:`f_i` is the Berry curvature and occupation
    for state :math:`i`.

    The conductivity will be averaged by the Brillouin zone volume of the parent. See `BrillouinZone.volume` for details.
    Hence for 1D the returned unit will be S/Ang, 2D it will be S/Ang^2 and 3D it will be S/Ang^3.

    Parameters
    ----------
    bz : BrillouinZone
        containing the integration grid and has the ``bz.parent`` as an instance of Hamiltonian.
    distribution : str or func, optional
        distribution used to find occupations
    method : {'ahc'}
       'ahc' calculates the dc anomalous Hall conductivity
    degenerate : float, optional
       de-couple degenerate states within the given tolerance (in eV)
    degenerate_dir : (3,), optional
       along which direction degenerate states are decoupled.

    References
    ----------
    .. [1] :doi:`X. Wang, J. R. Yates, I. Souza, D. Vanderbilt, "Ab initio calculation of the anomalous Hall conductivity by Wannier interpolation", PRB **74**, 195118 (2006) <10.1103/PhysRevB.74.195118>`

    Returns
    -------
    cond : float
        conductivity in units [S/cm^D]. The D is the dimensionality of the system.

    See Also
    --------
    berry_curvature: method used to calculate the Berry-flux for calculating the conductivity
    BrillouinZone.volume: volume calculation of the Brillouin zone
    """
    from .hamiltonian import Hamiltonian
    # Currently we require the conductivity calculation to *only* accept Hamiltonians
    if not isinstance(bz.parent, Hamiltonian):
        raise SislError("conductivity: requires the Brillouin zone object to contain a Hamiltonian!")

    if isinstance(distribution, str):
        distribution = get_distribution(distribution)

    method = method.lower()
    if method == 'ahc':
        def _ahc(es):
            occ = distribution(es.eig)
            bc = es.berry_curvature(degenerate=degenerate, degenerate_dir=degenerate_dir)
            return (bc.T @ distribution(es.eig)).T

        vol, dim = bz.volume(ret_dim=True)

        if dim == 0:
            raise SislError(f"conductivity: found a dimensionality of 0 which is non-physical")

        cond = bz.apply.average.eigenstate(wrap=_ahc) * (-constant.G0 / (4*np.pi))

        # Convert the dimensions from S/m^D to S/cm^D
        cond /= vol * units(f"Ang^{dim}", f"cm^{dim}")
        warn("conductivity: be aware that the units are currently not tested, please provide feedback!")

    else:
        raise SislError("conductivity: requires the method to be [ahc]")

    return cond


@set_module("sisl.physics.electron")
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
       Forced true for Zak-phase calculations.
    method : {'berry', 'zak'}
       'berry' will return the usual integral of the Berry connection over the specified contour
       'zak' will compute the Zak phase for 1D systems by performing
       a closed loop integration, see [1]_.
       Additionally, one may do the Berry-phase calculation using the SVD method of the
       overlap matrices. Simply append ":svd" to the chosen method, e.g. "berry:svd".

    Notes
    -----
    The Brillouin zone object *need* not contain a closed discretized contour by doubling the first point.

    The implementation is very similar to PythTB, except we are here using the :math:`\mathbf R` gauge
    (convention II according to PythTB), see discussion in :pull:`131`.

    For systems with band-crossings or degenerate states there is an arbitrariness to the definition
    of the Berry phase for *individual* bands. However, the total phase (i.e., sum over filled bands) is
    invariant and unaffected by this arbitrariness as long as the filled and empty bands do not intersect,
    see [2]_.

    For non-orthogonal basis sets it is not fully known how important the :math:`\delta\mathbf k` spacing is since
    it relies on the Lowdin transformation of the states. However, one should be careful about choosing
    the correct bands for examination.

    The returned angles are _not_ placed in the interval :math:`]-\pi;\pi]` as what `numpy.angle` would do.
    This is to allow users to examine the quantities as is.

    Examples
    --------

    Calculate Berry-phase for first band but using the SVD me

    >>> N = 30
    >>> kR = 0.01
    >>> normal = [0, 0, 1]
    >>> origin = [1/3, 2/3, 0]
    >>> bz = BrillouinZone.param_circle(H, N, kR, normal, origin)
    >>> phase = berry_phase(bz, sub=0)

    Calculate the multi-band Berry-phase using the SVD method, thus
    ensuring singular vectors are removed.

    >>> N = 30
    >>> kR = 0.01
    >>> normal = [0, 0, 1]
    >>> origin = [1/3, 2/3, 0]
    >>> bz = BrillouinZone.param_circle(H, N, kR, normal, origin)
    >>> phase = berry_phase(bz, method="berry:svd")

    References
    ----------
    .. [1] :doi:`J. Zak, "Berry's phase for energy bands in solids", PRL **62**, 2747 (1989) <10.1103/PhysRevLett.62.2747>`
    .. [2] :doi:`R. Resta, "Manifestations of Berry's phase in molecules and condensed matter", JPCM **12**, R107 (2000) <10.1088/0953-8984/12/9/201>`
    .. [3] :doi:`Tutorial: Computing Topological Invariants in 2D Photonic Crystals <10.1002/qute.201900117>`
    """
    from .hamiltonian import Hamiltonian
    # Currently we require the Berry phase calculation to *only* accept Hamiltonians
    if not isinstance(contour.parent, Hamiltonian):
        raise SislError("berry_phase: requires the Brillouin zone object to contain a Hamiltonian!")
    spin = contour.parent.spin

    if contour.parent.orthogonal:
        def _lowdin(state):
            pass
    else:
        def _lowdin(state):
            """ change state to the lowdin state, assuming everything is in R gauge
            So needs to be done before changing gauge """
            S12 = sqrth(state.parent.Sk(state.info["k"], format='array'),
                        overwrite_a=True)
            state.state[:, :] = (S12 @ state.state.T).T

    method, *opts = method.lower().split(':')
    if method == "berry":
        pass
    elif method == "zak":
        closed = True
    else:
        raise ValueError("berry_phase: requires the method to be [berry, zak]")

    if np.allclose(contour.k[0, :], contour.k[-1, :]):
        # When the user has the contour points closed, we don't need to do this in the below loop
        closed = False

    # We calculate the final angle from the determinant
    _process = dot

    if "svd" in opts:
        def _process(prd, overlap):
            U, _, V = svd_destroy(overlap)
            return dot(prd, U @ V)

    if sub is None:
        def _berry(eigenstates):
            # Grab the first one to be able to form a loop
            first = next(eigenstates)
            _lowdin(first)
            # Create a variable to keep track of the previous state
            prev = first

            # Initialize the consecutive product
            # Starting with the identity matrix!
            prd = 1

            # Loop remaining eigenstates
            for second in eigenstates:
                _lowdin(second)
                prd = _process(prd, prev.inner(second, diag=False))
                prev = second

            # Complete the loop
            if closed:
                # Include last-to-first segment
                prd = _process(prd, prev.inner(first, diag=False))
            return prd

    else:
        def _berry(eigenstates):
            first = next(eigenstates)
            _lowdin(first)
            first.sub(sub, inplace=True)
            prev = first
            prd = 1
            for second in eigenstates:
                _lowdin(second)
                second.sub(sub, inplace=True)
                prd = _process(prd, prev.inner(second, diag=False))
                prev = second
            if closed:
                prd = _process(prd, prev.inner(first, diag=False))
            return prd

    # Do the actual calculation of the final matrix
    d = _berry(contour.apply.iter.eigenstate())

    # Get the angle of the berry-phase
    # When using np.angle the returned value is in ]-pi; pi]
    # However, small numerical differences makes wrap-arounds annoying.
    # We'll always return the full angle. Then users can them-selves control
    # how to convert them.
    if eigvals:
        ret = -log(eigvals_destroy(d)).imag
        ret = sort(ret)
    else:
        ret = -log(det_destroy(d)).imag

    return ret


@set_module("sisl.physics.electron")
def wavefunction(v, grid, geometry=None, k=None, spinor=0, spin=None, eta=None):
    r""" Add the wave-function (`Orbital.psi`) component of each orbital to the grid

    This routine calculates the real-space wave-function components in the
    specified grid.

    This is an *in-place* operation that *adds* to the current values in the grid.

    It may be instructive to check that an eigenstate is normalized:

    >>> grid = Grid(...)
    >>> wavefunction(state, grid)
    >>> (np.absolute(grid.grid) ** 2).sum() * grid.dvolume == 1.

    Note: To calculate :math:`\psi(\mathbf r)` in a unit-cell different from the
    originating geometry, simply pass a grid with a unit-cell smaller than the originating
    supercell.

    The wavefunctions are calculated in real-space via:

    .. math::
       \psi(\mathbf r) = \sum_i\phi_i(\mathbf r) |\psi\rangle_i \exp(i\mathbf k \mathbf R)

    While for non-colinear/spin-orbit calculations the wavefunctions are determined from the
    spinor component (`spinor`)

    .. math::
       \psi_{\alpha/\beta}(\mathbf r) = \sum_i\phi_i(\mathbf r) |\psi_{\alpha/\beta}\rangle_i \exp(i\mathbf k \mathbf R)

    where ``spinor in [0, 1]`` determines :math:`\alpha` or :math:`\beta`, respectively.

    Notes
    -----
    Currently this method only works for `v` being coefficients of the gauge='R' method. In case
    you are passing a `v` with the incorrect gauge you will find a phase-shift according to:

    .. math::
        \tilde v_j = e^{i\mathbf k\mathbf r_j} v_j

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
    # Decipher v from State type
    if isinstance(v, State):
        if geometry is None:
            geometry = v.parent.geometry
        if k is None:
            k = v.info.get("k", k)
        elif not np.allclose(k, v.info.get("k", k)):
            raise ValueError(f"wavefunction: k passed and k in info does not match: {k} and {v.info.get('k')}")
        v = v.state
    if geometry is None:
        geometry = grid.geometry
    if geometry is None:
        raise SislError("wavefunction: did not find a usable Geometry through keywords or the Grid!")
    # Ensure coordinates are in the primary unit-cell, regardless of origin etc.
    geometry = geometry.copy()
    geometry.xyz = (geometry.fxyz % 1) @ geometry.sc.cell

    # In case the user has passed several vectors we sum them to plot the summed state
    if v.ndim == 2:
        if v.shape[0] > 1:
            info(f"wavefunction: summing {v.shape[0]} different state coefficients, will continue silently!")
        v = v.sum(0)

    if spin is None:
        if len(v) // 2 == geometry.no:
            # We can see from the input that the vector *must* be a non-colinear calculation
            v = v.reshape(-1, 2)[:, spinor]
            info("wavefunction: assumes the input wavefunction coefficients to originate from a non-colinear calculation!")

    elif spin.kind > Spin.POLARIZED:
        # For non-colinear cases the user selects the spinor component.
        v = v.reshape(-1, 2)[:, spinor]

    if len(v) != geometry.no:
        raise ValueError("wavefunction: require wavefunction coefficients corresponding to number of orbitals in the geometry.")

    # Check for k-points
    k = _a.asarrayd(k)
    kl = k.dot(k) ** 0.5
    has_k = kl > 0.000001
    if has_k:
        info('wavefunction: k != Gamma is currently untested!')

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
    origin = grid.sc.origin.reshape(1, 3)
    idx_mm = _a.emptyd([geometry.na, 2, 3])
    all_negative_R = True
    for atom, ia in geometry.atoms.iter(True):
        if len(ia) == 0:
            continue
        R = atom.maxR()
        all_negative_R = all_negative_R and R < 0.

        # Now do it for all the atoms to get indices of the middle of
        # the atoms
        # The coordinates are relative to origin, so we need to shift (when writing a grid
        # it is with respect to origin)
        idx = dot(geometry.xyz[ia, :] - origin, ic_shape.T)

        # Get min-max for all atoms
        idx_mm[ia, 0, :] = idxm * R + idx
        idx_mm[ia, 1, :] = idxM * R + idx

    if all_negative_R:
        raise SislError("wavefunction: Cannot create wavefunction since no atoms have an associated basis-orbital on a real-space grid")

    # Now we have min-max for all atoms
    # When we run the below loop all indices can be retrieved by looking
    # up in the above table.
    # Before continuing, we can easily clean up the temporary arrays
    del origin, idx

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
    sc = SuperCell(o._v + np.diag(2 * add_R), origin=o.origin - add_R)

    # Retrieve all atoms within the grid supercell
    # (and the neighbours that connect into the cell)
    # Note that we cannot pass the "moved" origin because then ISC would be wrong
    IA, XYZ, ISC = geometry.within_inf(sc, periodic=pbc)
    # We need to revert the grid supercell origin as that is not subtracted in the `within_inf` returned
    # coordinates (and the below loop expects positions with respect to the origin of the plotting
    # grid).
    XYZ -= grid.sc.origin.reshape(1, 3)

    phk = k * 2 * np.pi
    phase = 1

    # Retrieve progressbar
    eta = progressbar(len(IA), "wavefunction", "atom", eta)

    # Loop over all atoms in the grid-cell
    for ia, xyz, isc in zip(IA, XYZ, ISC):
        # Get current atom
        atom = geometry.atoms[ia]

        # Extract maximum R
        R = atom.maxR()
        if R <= 0.:
            warn(f"wavefunction: Atom '{atom}' does not have a wave-function, skipping atom.")
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
            phase = exp(1j * phk.dot(isc))

        # Allocate a temporary array where we add the psi elements
        psi = psi_init(n)

        # Loop on orbitals on this atom, grouped by radius
        for os in atom.iter(True):

            # Get the radius of orbitals (os)
            oR = os[0].R

            if oR <= 0.:
                warn(f"wavefunction: Orbital(s) '{os}' does not have a wave-function, skipping orbital!")
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


class _electron_State:
    __slots__ = []

    def __is_nc(self):
        """ Internal routine to check whether this is a non-colinear calculation """
        try:
            return not self.parent.spin.is_diagonal
        except:
            return False

    def Sk(self, format=None, spin=None):
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
        if format is None:
            format = self.info.get("format", "csr")
        if isinstance(self.parent, SparseOrbitalBZSpin):
            # Calculate the overlap matrix
            if not self.parent.orthogonal:
                opt = {'k': self.info.get('k', (0, 0, 0)),
                       "dtype": self.dtype,
                       "format": format}
                for key in ("gauge",):
                    val = self.info.get(key, None)
                    if not val is None:
                        opt[key] = val
                return self.parent.Sk(**opt)

        if self.__is_nc():
            n = self.shape[1] // 2
        else:
            n = self.shape[1]
        if 'sc:' in format:
            m = n * self.parent.n_s
        else:
            m = n

        return _FakeMatrix(n, m)

    def norm2(self, sum=True):
        r""" Return a vector with the norm of each state :math:`\langle\psi|\mathbf S|\psi\rangle`

        :math:`\mathbf S` is the overlap matrix (or basis), for orthogonal basis
        :math:`\mathbf S \equiv \mathbf I`.

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
        # Retrieve the overlap matrix (FULL S is required for NC)
        S = self.Sk()

        if sum:
            return self.inner(matrix=S)
        return conj(self.state) * S.dot(self.state.T).T

    def spin_moment(self, project=False):
        r""" Calculate spin moment from the states

        This routine calls `~sisl.physics.electron.spin_moment` with appropriate arguments
        and returns the spin moment for the states.

        See `~sisl.physics.electron.spin_moment` for details.

        Parameters
        ----------
        project : bool, optional
           whether the moments are orbitally resolved or not
        """
        return spin_moment(self.state, self.Sk(), project=project)

    def wavefunction(self, grid, spinor=0, eta=None):
        r""" Expand the coefficients as the wavefunction on `grid` *as-is*

        See `~sisl.physics.electron.wavefunction` for argument details, the arguments not present
        in this method are automatically passed from this object.
        """
        spin = getattr(self.parent, "spin", None)

        if isinstance(self.parent, Geometry):
            geometry = self.parent
        else:
            geometry = getattr(self.parent, "geometry", None)

        # Ensure we are dealing with the R gauge
        self.change_gauge('R')

        # Retrieve k
        k = self.info.get('k', _a.zerosd(3))

        wavefunction(self.state, grid, geometry=geometry, k=k, spinor=spinor, spin=spin, eta=eta)


@set_module("sisl.physics.electron")
class CoefficientElectron(Coefficient):
    r""" Coefficients describing some physical quantity related to electrons """
    __slots__ = []


@set_module("sisl.physics.electron")
class StateElectron(_electron_State, State):
    r""" A state describing a physical quantity related to electrons """
    __slots__ = []


@set_module("sisl.physics.electron")
class StateCElectron(_electron_State, StateC):
    r""" A state describing a physical quantity related to electrons, with associated coefficients of the state """
    __slots__ = []

    def velocity(self, *args, **kwargs):
        r""" Calculate velocity for the states

        This routine calls ``derivative(1, *args, **kwargs)`` and returns the velocity for the states.

        Note that the coefficients associated with the `StateCElectron` *must* correspond
        to the energies of the states.

        Notes
        -----
        The states and energies for the states *may* have changed after calling this routine.
        This is because of the velocity un-folding for degenerate modes. I.e. calling
        `displacement` and/or `PDOS` after this method *may* change the result.

        See Also
        --------
        derivative : for details of the implementation
        """
        return self.derivative(1, *args, **kwargs).real * _velocity_const

    def berry_curvature(self, *args, **kwargs):
        r""" Calculate Berry curvature for the states

        This routine calls ``derivative(1, *args, **kwargs, matrix=True)`` and
        returns the Berry curvature for the states.

        Note that the coefficients associated with the `StateCElectron` *must* correspond
        to the energies of the states.

        See Also
        --------
        derivative : for details of the velocity matrix calculation implementation
        sisl.physics.electron.berry_curvature : for details of the Berry curvature implementation
        """
        v = self.derivative(1, *args, **kwargs, matrix=True)
        return _berry_curvature(v, self.c)

    def effective_mass(self, *args, **kwargs):
        r""" Calculate effective mass tensor for the states

        This routine calls ``derivative(2, *args, **kwargs)`` and
        returns the effective mass for all states.

        Note that the coefficients associated with the `StateCElectron` *must* correspond
        to the energies of the states.

        Notes
        -----
        Since some directions may not be periodic there will be zeros. This routine will
        invert elements where the values are different from 0.

        It is not advisable to use a `sub` before calculating the effective mass
        since the 1st order perturbation uses the energy differences and the 1st derivative
        matrix for correcting the curvature.

        The returned effective mass is given in the Voigt notation.

        For :math:`\Gamma` point calculations it may be beneficial to pass `dtype=np.complex128`
        to the `eigenstate` argument to ensure their complex values. This is necessary for the
        degeneracy decoupling.

        See Also
        --------
        derivative: for details of the implementation
        """
        ieff = self.derivative(2, *args, **kwargs)[1].real
        np.divide(_velocity_const ** 2, ieff, where=(ieff != 0), out=ieff)
        return ieff


@set_module("sisl.physics.electron")
class EigenvalueElectron(CoefficientElectron):
    r""" Eigenvalues of electronic states, no eigenvectors retained

    This holds routines that enable the calculation of density of states.
    """
    __slots__ = []

    @property
    def eig(self):
        """ Eigenvalues """
        return self.c

    def occupation(self, distribution="fermi_dirac"):
        r""" Calculate the occupations for the states according to a distribution function

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

    def DOS(self, E, distribution="gaussian"):
        r""" Calculate DOS for provided energies, `E`.

        This routine calls `sisl.physics.electron.DOS` with appropriate arguments
        and returns the DOS.

        See `~sisl.physics.electron.DOS` for argument details.
        """
        return DOS(E, self.eig, distribution)


@set_module("sisl.physics.electron")
class EigenvectorElectron(StateElectron):
    r""" Eigenvectors of electronic states, no eigenvalues retained

    This holds routines that enable the calculation of spin moments.
    """
    __slots__ = []


@set_module("sisl.physics.electron")
class EigenstateElectron(StateCElectron):
    r""" Eigen states of electrons with eigenvectors and eigenvalues.

    This holds routines that enable the calculation of (projected) density of states,
    spin moments (spin texture).
    """
    __slots__ = []

    @property
    def eig(self):
        r""" Eigenvalues for each state """
        return self.c

    def occupation(self, distribution="fermi_dirac"):
        r""" Calculate the occupations for the states according to a distribution function

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

    def DOS(self, E, distribution="gaussian"):
        r""" Calculate DOS for provided energies, `E`.

        This routine calls `sisl.physics.electron.DOS` with appropriate arguments
        and returns the DOS.

        See `~sisl.physics.electron.DOS` for argument details.
        """
        return DOS(E, self.c, distribution)

    def PDOS(self, E, distribution="gaussian"):
        r""" Calculate PDOS for provided energies, `E`.

        This routine calls `~sisl.physics.electron.PDOS` with appropriate arguments
        and returns the PDOS.

        See `~sisl.physics.electron.PDOS` for argument details.
        """
        return PDOS(E, self.c, self.state, self.Sk(), distribution, getattr(self.parent, "spin", None))

    def COP(self, M, E, distribution="gaussian"):
        r""" Calculate COP for provided energies, `E` using matrix `M`

        This routine calls `~sisl.physics.electron.COP` with appropriate arguments.
        """
        return COP(E, self.c, self.state, M, distribution)

    def COOP(self, E, distribution="gaussian"):
        r""" Calculate COOP for provided energies, `E`.

        This routine calls `~sisl.physics.electron.COP` with appropriate arguments.
        """
        # Get Sk in full format
        Sk = self.Sk(format='sc:csr')
        return COP(E, self.c, self.state, Sk, distribution)

    def COHP(self, E, distribution="gaussian"):
        r""" Calculate COHP for provided energies, `E`.

        This routine calls `~sisl.physics.electron.COP` with appropriate arguments.
        """
        # Get Hk in full format
        Hk = self.parent.Hk(format='sc:csr')
        return COP(E, self.c, self.state, Hk, distribution)
