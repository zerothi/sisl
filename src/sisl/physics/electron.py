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
   berry_phase
   ahc
   shc
   wavefunction
   spin_moment
   spin_contamination


Supporting classes
------------------

Certain classes aid in the usage of the above methods by implementing them
using automatic arguments.

For instance, the PDOS method requires the overlap matrix in non-orthogonal
basis sets at the :math:`\mathbf k`-point corresponding to the eigenstates. Hence, the
argument ``S`` must be :math:`\mathbf S(\mathbf k)`. The `EigenstateElectron` class
automatically passes the correct ``S`` because it knows the states :math:`\mathbf k`-point.

   CoefficientElectron
   StateElectron
   StateCElectron
   EigenvalueElectron
   EigenvectorElectron
   EigenstateElectron

"""
from __future__ import annotations

from collections.abc import Callable
from functools import reduce
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
import scipy.sparse as scs
from numpy import (
    add,
    ceil,
    conj,
    cos,
    dot,
    empty,
    exp,
    floor,
    int32,
    log,
    ogrid,
    pi,
    sin,
    sort,
    zeros,
)
from scipy.sparse import csr_matrix, hstack, issparse

import sisl._array as _a
from sisl import BoundaryCondition as BC
from sisl import C, Geometry, Grid, Lattice
from sisl._core.oplist import oplist
from sisl._indices import indices_le
from sisl._internal import set_module
from sisl._math_small import xyz_to_spherical_cos_phi
from sisl.linalg import det
from sisl.linalg import eigvals as la_eigvals
from sisl.linalg import sqrth, svd_destroy
from sisl.messages import (
    SislError,
    deprecate_argument,
    deprecation,
    info,
    progressbar,
    warn,
)
from sisl.physics._common import comply_projection
from sisl.typing import (
    CartesianAxisStrLiteral,
    DistributionType,
    ProjectionType,
    ProjectionTypeDiag,
    ProjectionTypeHadamard,
    ProjectionTypeHadamardAtoms,
)
from sisl.utils.misc import direction

if TYPE_CHECKING:
    from .brillouinzone import BrillouinZone

from .distribution import get_distribution
from .sparse import SparseOrbitalBZSpin
from .spin import Spin
from .state import Coefficient, State, StateC, _FakeMatrix

__all__ = ["DOS", "PDOS", "COP"]
__all__ += ["spin_moment", "spin_contamination"]
__all__ += ["berry_phase"]
__all__ += ["ahc", "shc", "conductivity"]
__all__ += ["wavefunction"]
__all__ += ["CoefficientElectron", "StateElectron", "StateCElectron"]
__all__ += ["EigenvalueElectron", "EigenvectorElectron", "EigenstateElectron"]


@set_module("sisl.physics.electron")
def DOS(E, eig, distribution: DistributionType = "gaussian"):
    r"""Calculate the density of states (DOS) for a set of energies, `E`, with a distribution function

    The :math:`\mathrm{DOS}(E)` is calculated as:

    .. math::
       \mathrm{DOS}(E) = \sum_i D(E-\epsilon_i) \approx\delta(E-\epsilon_i)

    where :math:`D(\Delta E)` is the distribution function used. Note that the distribution function
    used may be a user-defined function. Alternatively a distribution function may
    be retrieved from :ref:`physics.distribution`.

    Parameters
    ----------
    E : array_like
       energies to calculate the DOS at
    eig : array_like
       electronic eigenvalues
    distribution :
       a function that accepts :math:`\Delta E` as argument and calculates the
       distribution function.

    See Also
    --------
    :ref:`physics.distribution` : a selected set of implemented distribution functions
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

    return reduce(lambda DOS, eig: DOS + distribution(E - eig), eig, 0.0)


@set_module("sisl.physics.electron")
def PDOS(E, eig, state, S=None, distribution: DistributionType = "gaussian", spin=None):
    r""" Calculate the projected density of states (PDOS) for a set of energies, `E`, with a distribution function

    The :math:`\mathrm{PDOS}(E)` is calculated as:

    .. math::
       \mathrm{PDOS}_i(E) = \sum_\alpha \psi^*_{\alpha,i} [\mathbf S | \psi_{\alpha}\rangle]_i D(E-\epsilon_\alpha)

    where :math:`D(\Delta E)` is the distribution function used. Note that the distribution function
    used may be a user-defined function. Alternatively a distribution function may
    be acquired from :ref:`physics.distribution`.

    In case of an orthogonal basis set :math:`\mathbf S` is equal to the identity matrix.
    Note that `DOS` is the sum of the orbital projected DOS:

    .. math::
       \mathrm{DOS}(E) = \sum_i\mathrm{PDOS}_i(E)

    For non-colinear calculations (this includes spin-orbit calculations) the PDOS is additionally
    separated into 4 components (in this order):

    - Total projected DOS
    - Projected spin magnetic moment along :math:`x` direction
    - Projected spin magnetic moment along :math:`y` direction
    - Projected spin magnetic moment along :math:`z` direction

    These are calculated using the Pauli matrices :math:`\boldsymbol\sigma_x`, :math:`\boldsymbol\sigma_y` and :math:`\boldsymbol\sigma_z`:

    .. math::

       \mathrm{PDOS}_i^\sigma(E) &= \sum_\alpha \psi^*_{\alpha,i} \boldsymbol\sigma_z \boldsymbol\sigma_z [\mathbf S | \psi_\alpha\rangle]_i D(E-\epsilon_\alpha)
       \\
               \mathrm{PDOS}_i^x(E) &= \sum_\alpha \psi^*_{\alpha,i} \boldsymbol\sigma_x [\mathbf S | \psi_\alpha\rangle]_i D(E-\epsilon_\alpha)
       \\
               \mathrm{PDOS}_i^y(E) &= \sum_\alpha \psi^*_{\alpha,i} \boldsymbol\sigma_y [\mathbf S | \psi_\alpha\rangle]_i D(E-\epsilon_\alpha)
       \\
               \mathrm{PDOS}_i^z(E) &= \sum_\alpha\psi^*_{\alpha,i} \boldsymbol\sigma_z [\mathbf S | \psi_\alpha\rangle]_i D(E-\epsilon_\alpha)

    Note that the total PDOS may be calculated using :math:`\boldsymbol\sigma_\gamma\boldsymbol\sigma_\gamma` where :math:`\gamma` may be either of :math:`x`,
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
    distribution :
       a function that accepts :math:`E-\epsilon` as argument and calculates the
       distribution function.
    spin : str or Spin, optional
       the spin configuration. This is generally only needed when the eigenvectors correspond to a non-colinear
       calculation.

    See Also
    --------
    :ref:`physics.distribution` : a selected set of implemented distribution functions
    DOS : total DOS (same as summing over orbitals)
    COP : calculate COOP or COHP curves
    spin_moment : spin moment
    Geometry.apply : allows one to convert orbital data, to atomic data

    Returns
    -------
    numpy.ndarray
        projected DOS calculated at energies, has dimension ``(1, state.shape[1], len(E))``.
        For non-colinear calculations it will be ``(4, state.shape[1] // 2, len(E))``, ordered as
        indicated in the above list.
        For Nambu calculations it will be ``(8, state.shape[1] // 4, len(E))``.
    """
    if isinstance(distribution, str):
        distribution = get_distribution(distribution)

    # Figure out whether we are dealing with a non-colinear calculation
    if S is None:
        S = _FakeMatrix(state.shape[1])

    if spin is None:
        if S.shape[1] == state.shape[1] // 2:
            spin = Spin("nc")
            S = S[::2, ::2]
        elif S.shape[1] == state.shape[1] // 4:
            spin = Spin("nambu")
            S = S[::4, ::4]
        else:
            spin = Spin()

    # check for non-colinear (or SO)
    if spin.kind > Spin.SPINORBIT:
        # Non colinear eigenvectors
        if S.shape[1] == state.shape[1]:
            # Since we are going to reshape the eigen-vectors
            # to more easily get the mixed states, we can reduce the overlap matrix
            S = S[::4, ::4]

        # Initialize data
        PDOS = empty([8, state.shape[1] // 4, len(E)], dtype=state.real.dtype)

        # Do spin-box calculations:
        #  PDOS[:4] = electron
        #  PDOS[0] = total DOS (diagonal)
        #  PDOS[1] = x == < psi | \sigma_x S | psi >
        #  PDOS[2] = y == < psi | \sigma_y S | psi >
        #  PDOS[3] = z == < psi | \sigma_z S | psi >
        #  PDOS[4:] = hole

        d = distribution(E - eig[0]).reshape(1, -1)
        cs = conj(state[0]).reshape(-1, 4)
        v = S @ state[0].reshape(-1, 4)
        D1 = (cs * v).real  # uu,dd PDOS
        PDOS[0, :, :] = D1[..., [0, 1]].sum(1).reshape(-1, 1) * d  # total DOS
        PDOS[3, :, :] = (D1[:, 0] - D1[:, 1]).reshape(-1, 1) * d  # z-dos
        PDOS[4, :, :] = D1[..., [2, 3]].sum(1).reshape(-1, 1) * d  # total DOS
        PDOS[7, :, :] = (D1[:, 2] - D1[:, 3]).reshape(-1, 1) * d  # z-dos
        D1 = (cs[:, 1] * v[:, 0]).reshape(-1, 1)  # d,u
        D2 = (cs[:, 0] * v[:, 1]).reshape(-1, 1)  # u,d
        PDOS[1, :, :] = (D1.real + D2.real) * d  # x-dos
        PDOS[2, :, :] = (D2.imag - D1.imag) * d  # y-dos
        D1 = (cs[:, 3] * v[:, 2]).reshape(-1, 1)  # d,u
        D2 = (cs[:, 2] * v[:, 3]).reshape(-1, 1)  # u,d
        PDOS[5, :, :] = (D1.real + D2.real) * d  # x-dos
        PDOS[6, :, :] = (D2.imag - D1.imag) * d  # y-dos
        for i in range(1, len(eig)):
            d = distribution(E - eig[i]).reshape(1, -1)
            cs = conj(state[i]).reshape(-1, 4)
            v = S @ state[i].reshape(-1, 4)
            D1 = (cs * v).real
            PDOS[0, :, :] += D1[..., [0, 1]].sum(1).reshape(-1, 1) * d  # total DOS
            PDOS[3, :, :] += (D1[:, 0] - D1[:, 1]).reshape(-1, 1) * d  # z-dos
            PDOS[4, :, :] += D1[..., [2, 3]].sum(1).reshape(-1, 1) * d  # total DOS
            PDOS[7, :, :] += (D1[:, 2] - D1[:, 3]).reshape(-1, 1) * d  # z-dos
            D1 = (cs[:, 1] * v[:, 0]).reshape(-1, 1)  # d,u
            D2 = (cs[:, 0] * v[:, 1]).reshape(-1, 1)  # u,d
            PDOS[1, :, :] += (D1.real + D2.real) * d  # x-dos
            PDOS[2, :, :] += (D2.imag - D1.imag) * d  # y-dos
            D1 = (cs[:, 3] * v[:, 2]).reshape(-1, 1)  # d,u
            D2 = (cs[:, 2] * v[:, 3]).reshape(-1, 1)  # u,d
            PDOS[5, :, :] += (D1.real + D2.real) * d  # x-dos
            PDOS[6, :, :] += (D2.imag - D1.imag) * d  # y-dos

    elif spin.kind > Spin.POLARIZED:
        # check for non-colinear (or SO)
        # Non colinear eigenvectors
        if S.shape[1] == state.shape[1]:
            # Since we are going to reshape the eigen-vectors
            # to more easily get the mixed states, we can reduce the overlap matrix
            S = S[::2, ::2]

        # Initialize data
        PDOS = empty([4, state.shape[1] // 2, len(E)], dtype=state.real.dtype)

        # Do spin-box calculations:
        #  PDOS[0] = total DOS (diagonal)
        #  PDOS[1] = x == < psi | \sigma_x S | psi >
        #  PDOS[2] = y == < psi | \sigma_y S | psi >
        #  PDOS[3] = z == < psi | \sigma_z S | psi >

        d = distribution(E - eig[0]).reshape(1, -1)
        cs = conj(state[0]).reshape(-1, 2)
        v = S @ state[0].reshape(-1, 2)
        D1 = (cs * v).real  # uu,dd PDOS
        PDOS[0, :, :] = D1.sum(1).reshape(-1, 1) * d  # total DOS
        PDOS[3, :, :] = (D1[:, 0] - D1[:, 1]).reshape(-1, 1) * d  # z-dos
        D1 = (cs[:, 1] * v[:, 0]).reshape(-1, 1)  # d,u
        D2 = (cs[:, 0] * v[:, 1]).reshape(-1, 1)  # u,d
        PDOS[1, :, :] = (D1.real + D2.real) * d  # x-dos
        PDOS[2, :, :] = (D2.imag - D1.imag) * d  # y-dos
        for i in range(1, len(eig)):
            d = distribution(E - eig[i]).reshape(1, -1)
            cs = conj(state[i]).reshape(-1, 2)
            v = S @ state[i].reshape(-1, 2)
            D1 = (cs * v).real
            PDOS[0, :, :] += D1.sum(1).reshape(-1, 1) * d
            PDOS[3, :, :] += (D1[:, 0] - D1[:, 1]).reshape(-1, 1) * d
            D1 = (cs[:, 1] * v[:, 0]).reshape(-1, 1)
            D2 = (cs[:, 0] * v[:, 1]).reshape(-1, 1)
            PDOS[1, :, :] += (D1.real + D2.real) * d
            PDOS[2, :, :] += (D2.imag - D1.imag) * d

    else:
        PDOS = (conj(state[0]) * (S @ state[0])).real.reshape(-1, 1) * distribution(
            E - eig[0]
        ).reshape(1, -1)

        for i in range(1, len(eig)):
            PDOS += (conj(state[i]) * (S @ state[i])).real.reshape(
                -1, 1
            ) * distribution(E - eig[i]).reshape(1, -1)
        PDOS.shape = (1, *PDOS.shape)

    return PDOS


@set_module("sisl.physics.electron")
@deprecate_argument(
    "tol",
    "atol",
    "argument tol has been deprecated in favor of atol, please update your code.",
    "0.15",
    "0.17",
)
def COP(
    E, eig, state, M, distribution: DistributionType = "gaussian", atol: float = 1e-10
):
    r"""Calculate the Crystal Orbital Population for a set of energies, `E`, with a distribution function

    The :math:`\mathrm{COP}(E)` is calculated as:

    .. math::
       \mathrm{COP}_{i,j}(E) = \sum_\alpha \psi^*_{\alpha,i}\psi_{\alpha,j} \mathbf M e^{i\mathbf k\cdot \mathbf R} D(E-\epsilon_\alpha)

    where :math:`D(\Delta E)` is the distribution function used. Note that the distribution function
    used may be a user-defined function. Alternatively a distribution function may
    be acquired from :ref:`physics.distribution`.

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
    distribution :
       a function that accepts :math:`E-\epsilon` as argument and calculates the
       distribution function.
    atol :
       tolerance value where the distribution should be above before
       considering an eigenstate to contribute to an energy point,
       a higher value means that more energy points are discarded and so the calculation
       is faster.

    Notes
    -----
    This is not tested for non-collinear states.
    This requires substantial amounts of memory for big systems with lots of energy points.

    This method is considered experimental and implementation may change in the future.

    See Also
    --------
    :ref:`physics.distribution` : a selected set of implemented distribution functions
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

    assert len(eig) == len(
        state
    ), "COP: number of eigenvalues and states are not consistent"

    # get default dtype
    dtype = state.real.dtype

    # initialize the COP values
    no = M.shape[0]
    n_s = M.shape[1] // M.shape[0]

    if isinstance(M, _FakeMatrix):
        # A fake matrix equals the identity matrix.
        # Hence we can do all calculations only on the diagonal,
        # then finally we recreate the full matrix dimensions.
        cop = oplist(zeros(no, dtype=dtype) for _ in range(len(E)))

        def new_list(bools, tmp, we):
            for bl, w in zip(bools, we):
                if bl:
                    yield tmp * w
                else:
                    yield 0.0

        for e, s in zip(eig, state):
            # calculate contribution from this state
            we = distribution(E - e)
            bools = we >= atol
            if np.any(bools):
                tmp = (s.conj() * s).real
                cop += new_list(bools, tmp, we)

        # Now recreate the full size (in sparse form)
        idx = np.arange(no)

        def tosize(diag, idx):
            return csr_matrix((diag, (idx, idx)), shape=M.shape)

        cop = oplist(tosize(d, idx) for d in cop)

    elif issparse(M):
        # create the new list
        cop0 = M.multiply(0.0).real
        cop = oplist(cop0.copy() for _ in range(len(E)))
        del cop0

        # split M, then we will rejoin afterwards
        Ms = [M[:, i * no : (i + 1) * no] for i in range(n_s)]

        def new_list(bools, tmp, we):
            for bl, w in zip(bools, we):
                if bl:
                    yield tmp.multiply(w)
                else:
                    yield 0.0

        for e, s in zip(eig, state):
            # calculate contribution from this state
            we = distribution(E - e)
            bools = we >= atol
            if np.any(bools):
                s = np.outer(s.conj(), s)
                tmp = hstack([m.multiply(s).real for m in Ms])
                cop += new_list(bools, tmp, we)

    else:
        old_shape = M.shape
        Ms = M.reshape(old_shape[0], n_s, old_shape[0])
        cop = oplist(zeros(old_shape, dtype=dtype) for _ in range(len(E)))

        for e, s in zip(eig, state):
            we = distribution(E - e)
            # expand the state and do multiplication
            s = np.outer(s.conj(), s)[:, None, :]
            cop += we.reshape(-1, 1, 1) * (Ms * s).real.reshape(old_shape)

    return cop


@set_module("sisl.physics.electron")
@deprecate_argument(
    "project",
    "projection",
    "argument project has been deprecated in favor of projection",
    "0.15",
    "0.17",
)
def spin_moment(
    state,
    S=None,
    projection: Union[
        ProjectionTypeTrace, ProjectionTypeDiag, ProjectionTypeHadamard, True, False
    ] = "diagonal",
):
    r""" Spin magnetic moment (spin texture) and optionally orbitally resolved moments

    This calculation only makes sense for non-colinear calculations.

    The returned quantities are given in this order:

    - Spin magnetic moment along :math:`x` direction
    - Spin magnetic moment along :math:`y` direction
    - Spin magnetic moment along :math:`z` direction

    These are calculated using the Pauli matrices :math:`\boldsymbol\sigma_x`, :math:`\boldsymbol\sigma_y` and :math:`\boldsymbol\sigma_z`:

    .. math::

       \mathbf{S}_\alpha^x &= \langle \psi_\alpha | \boldsymbol\sigma_x \mathbf S | \psi_\alpha \rangle
       \\
       \mathbf{S}_\alpha^y &= \langle \psi_\alpha | \boldsymbol\sigma_y \mathbf S | \psi_\alpha \rangle
       \\
       \mathbf{S}_\alpha^z &= \langle \psi_\alpha | \boldsymbol\sigma_z \mathbf S | \psi_\alpha \rangle

    If `projection` is orbitals/basis/true, the above will be the orbitally resolved quantities.

    Parameters
    ----------
    state : array_like
       vectors describing the electronic states, 2nd dimension contains the states
    S : array_like, optional
       overlap matrix used in the :math:`\langle\psi|\mathbf S|\psi\rangle` calculation. If `None` the identity
       matrix is assumed. The overlap matrix should correspond to the system and :math:`\mathbf k` point the eigenvectors
       has been evaluated at.
    projection:
       how the projection should be done

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
        spin moments per state with final dimension ``(3, state.shape[0])``, or ``(3,
        state.shape[0], state.shape[1]//2)`` if projection is orbitals/basis/true
    """
    if state.ndim == 1:
        return spin_moment(state.reshape(1, -1), S, projection)[0]

    if isinstance(projection, bool):
        projection = "hadamard" if projection else "diagonal"
    projection = comply_projection(projection)

    if S is None:
        S = _FakeMatrix(state.shape[1] // 2, state.shape[1] // 2)

    if S.shape[1] == state.shape[1]:
        S = S[::2, ::2]

    # see PDOS for details related to the spin-box calculations

    if projection == "hadamard":
        s = empty(
            [3, state.shape[0], state.shape[1] // 2],
            dtype=state.real.dtype,
        )

        for i in range(len(state)):
            cs = conj(state[i]).reshape(-1, 2)
            Sstate = S @ state[i].reshape(-1, 2)
            D1 = (cs * Sstate).real
            s[2, i] = D1[:, 0] - D1[:, 1]
            D1 = cs[:, 1] * Sstate[:, 0]
            D2 = cs[:, 0] * Sstate[:, 1]
            s[0, i] = D1.real + D2.real
            s[1, i] = D2.imag - D1.imag

    elif projection == "diagonal":
        s = empty([3, state.shape[0]], dtype=state.real.dtype)

        # TODO consider doing this all in a few lines
        # TODO Since there are no energy dependencies here we can actually do all
        # TODO dot products in one go and then use b-casting rules. Should be much faster
        # TODO but also way more memory demanding!
        for i in range(len(state)):
            cs = conj(state[i]).reshape(-1, 2)
            Sstate = S @ state[i].reshape(-1, 2)
            D = cs.T @ Sstate
            s[2, i] = D[0, 0].real - D[1, 1].real
            s[0, i] = D[1, 0].real + D[0, 1].real
            s[1, i] = D[0, 1].imag - D[1, 0].imag

    elif projection == "trace":
        s = empty([3], dtype=state.real.dtype)

        for i in range(len(state)):
            cs = conj(state[i]).reshape(-1, 2)
            Sstate = S @ state[i].reshape(-1, 2)
            D = cs.T @ Sstate
            s[2] = (D[0, 0].real - D[1, 1].real).sum()
            s[0] = (D[1, 0].real + D[0, 1].real).sum()
            s[1] = (D[0, 1].imag - D[1, 0].imag).sum()

    else:
        raise ValueError(f"spin_moment got wrong 'projection' argument: {projection}.")

    return s


@set_module("sisl.physics.electron")
def spin_contamination(state_alpha, state_beta, S=None, sum: bool = True) -> oplist:
    r""" Calculate the spin contamination value between two spin states

    This calculation only makes sense for spin-polarized calculations.

    The contamination value is calculated using the following formula:

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
       matrix is assumed. The overlap matrix should correspond to the system and :math:`\mathbf k` point the eigenvectors
       have been evaluated at.
    sum :
        whether the spin-contamination should be summed for all states (a single number returned).
        If sum, a spin-contamination per state per spin-channel will be returned.

    Notes
    -----
    `state_alpha` and `state_beta` need not have the same number of states.

    Returns
    -------
    ~sisl.oplist :
         spin squared expectation value per spin channel :math:`\alpha` and :math:`\beta`.
         If `sum` is true, only a single number is returned (not a `~sisl.oplist`, otherwise a list for each
         state.
    """
    if state_alpha.ndim == 1:
        if state_beta.ndim == 1:
            ret = spin_contamination(
                state_alpha.reshape(1, -1), state_beta.reshape(1, -1), S, sum
            )
            if sum:
                return ret
            return oplist((ret[0][0], ret[1][0]))
        return spin_contamination(state_alpha.reshape(1, -1), state_beta, S, sum)
    elif state_beta.ndim == 1:
        return spin_contamination(state_alpha, state_beta.reshape(1, -1), S, sum)

    if state_alpha.shape[1] != state_beta.shape[1]:
        raise ValueError(
            "spin_contamination requires alpha and beta states to have same number of orbitals"
        )

    n_alpha = state_alpha.shape[0]
    n_beta = state_beta.shape[0]
    if S is None:
        S_state_beta = state_beta.T
    else:
        S_state_beta = S @ state_beta.T

    if sum:

        Ss = 0
        for s in state_alpha:
            D = conj(s) @ S_state_beta
            Ss += (D @ D.conj()).real

        return Ss

    else:

        Sa = empty([n_alpha], dtype=state_alpha.real.dtype)
        Sb = zeros([n_beta], dtype=Sa.dtype)

        # Loop alpha...
        for i, s in enumerate(state_alpha):
            D = conj(s) @ S_state_beta
            D = (D * conj(D)).real
            Sb += D
            Sa[i] = D.sum()

        return oplist((Sa, Sb))


# dHk is in [Ang eV]
# velocity units in [Ang/ps]
_velocity_const = 1 / C.hbar("eV ps")

# With G0 = 2e^2 / h = e^2 / (\hbar \pi)
# AHC is
#   \propto e^2/\hbar = G0 \pi
# This converts \sigma into S
_ahc_const = C.G0 * np.pi


@set_module("sisl.physics.electron")
def ahc(
    bz: BrillouinZone,
    k_average: bool = True,
    *,
    distribution: DistributionType = "step",
    eigenstate_kwargs={},
    apply_kwargs={},
    **berry_kwargs,
) -> np.ndarray:
    r"""Electronic anomalous Hall conductivity for a given `BrillouinZone` integral

    .. math::
       \sigma_{\alpha\beta} = \frac{-e^2}{\hbar}\int\,\mathrm d\mathbf k\sum_i f_i\Omega_{i,\alpha\beta}(\mathbf k)

    where :math:`\Omega_{i,\alpha\beta}` and :math:`f_i` is the Berry curvature and occupation
    for state :math:`i`.

    The conductivity will be averaged by volume of the periodic unit cell.
    Hence the unit of `ahc` depends on the periodic unit cell.
    See `~sisl.Lattice.volumef` for details.

    See :cite:`Wang2006` for details on the implementation.

    Parameters
    ----------
    bz :
        containing the integration grid and has the ``bz.parent`` as an instance of Hamiltonian.
    k_average :
        if `True`, the returned quantity is averaged over `bz`, else all k-point
        contributions will be collected (in the 1st dimension).
        Note, for large `bz` integrations this may explode the memory usage.
    distribution :
        An optional distribution enabling one to automatically sum states
        across occupied/unoccupied states.
    eigenstate_kwargs :
       keyword arguments passed directly to the ``contour.eigenstate`` method.
       One should *not* pass a ``k`` or a ``wrap`` keyword argument as they are
       already used.
    apply_kwargs :
       keyword arguments passed directly to ``bz.apply.renew(**apply_kwargs)``.
    **berry_kwargs :
        arguments passed directly to the `berry_curvature` method.

        Here one can pass `derivative_kwargs` to pass flags to the
        `derivative` method. In particular ``axes`` can be used
        to speedup the calculation (by omitting certain directions).

    Examples
    --------

    To calculate the AHC for a range of energy-points.
    First create ``E`` which is the energy grid.
    In order for the internal algorithm to be able
    to broadcast arrays correctly, we have to allow the eigenvalue
    spectrum to be appended by reshaping.

    >>> E = np.linspace(-2, 2, 51)
    >>> dist = get_distribution("step", x0=E.reshape(-1, 1))
    >>> ahc_cond = ahc(bz, dist)
    >>> assert ahc_cond.shape == (3, 3, len(E))

    Sometimes one wishes to see the k-resolved AHC.
    Be aware that AHC requires a dense k-grid, and hence it might
    require a lot of memory.
    Here it is calculated at :math:`E=0` (default energy reference).

    >>> ahc_cond = ahc(bz, k_average=False)
    >>> assert ahc_cond.shape == (len(bz), 3, 3)

    See Also
    --------
    ~sisl.physics.derivative: method for calculating the exact derivatives
    ~sisl.physics.berry_curvature: method used to calculate the Berry curvature for calculating the conductivity
    ~sisl.Lattice.volumef: volume calculation of the lattice
    shc: spin Hall conductivity

    Returns
    -------
    ahc:
        Anomalous Hall conductivity returned in certain dimensions ``ahc[:, :]``.
        If `sum` is False, it will be at least a 3D array with the 3rd dimension
        having the contribution from state `i`.
        If `k_average` is False, it will have a dimension prepended with
        k-point resolved AHC.
        If one passes `axes` to the `derivative_kwargs` argument one will get
        dimensions according to the number of axes requested, by default all
        axes will be used (even if they are non-periodic).
        The dtype will be imaginary.
        When :math:`D` is the dimensionality of the system we find the unit to be
        :math:`\mathrm S/\mathrm{Ang}^{D-2}`.
    """
    from .hamiltonian import Hamiltonian

    H = bz.parent

    # Currently we require the conductivity calculation to *only* accept Hamiltonians
    if not isinstance(H, Hamiltonian):
        raise SislError(
            "ahc: requires the Brillouin zone object to contain a Hamiltonian!"
        )

    if isinstance(distribution, str):
        distribution = get_distribution(distribution)

    def _ahc(es, k, weight, parent):
        # the latter arguments are merely for speeding up the procedure
        nonlocal berry_kwargs, distribution
        return es.berry_curvature(**berry_kwargs, distribution=distribution)

    apply = bz.apply.renew(**apply_kwargs)
    if k_average:
        apply = apply.average
    else:
        apply = apply.ndarray
    cond = apply.eigenstate(**eigenstate_kwargs, wrap=_ahc)

    lat = H.geometry.lattice
    per_axes = lat.pbc.nonzero()[0]
    vol = lat.volumef(per_axes)

    # Convert to S / Ang
    cond *= -_ahc_const / vol

    return cond


def _create_sigma(n, sigma, dtype, format):
    r"""This will return the Pauli matrix filled in a diagonal of the matrix

    It will not return the spin operator, which has the pre-factor \hbar/2

    """
    if isinstance(sigma, str):
        sigma = getattr(Spin, sigma.upper()) / 2
    else:
        # it must be an ndarray
        sigma = np.asarray(sigma)
        assert sigma.ndim == 2
        if len(sigma) == 2:
            # only the spin-box
            sigma = sigma / 2
        elif len(sigma) == n * 2:
            # full sigma
            sigma = sigma / 2
            return sigma

    if format in ("array", "matrix"):
        m = np.zeros([n, 2, n, 2], dtype=dtype)
        idx = np.arange(n)
        m[idx, 0, idx, 0] = sigma[0, 0]
        m[idx, 0, idx, 1] = sigma[0, 1]
        m[idx, 1, idx, 0] = sigma[1, 0]
        m[idx, 1, idx, 1] = sigma[1, 1]
        m.shape = (n * 2, n * 2)
    else:
        m = scs.kron(scs.eye(n, dtype=dtype), sigma).tocsr()
    return m


@set_module("sisl.physics.electron")
def shc(
    bz: BrillouinZone,
    k_average: bool = True,
    sigma: Union[CartesianAxisStrLiteral, npt.ArrayLike] = "z",
    *,
    J_axes: Union[CartesianAxisStrLiteral, Sequence[CartesianAxisStrLiteral]] = "xyz",
    distribution: DistributionType = "step",
    eigenstate_kwargs={},
    apply_kwargs={},
    **berry_kwargs,
) -> np.ndarray:
    r"""Electronic spin Hall conductivity for a given `BrillouinZone` integral

    .. math::
       \sigma^\gamma_{\alpha\beta} = \frac{-e^2}{\hbar}\int\,\mathrm d\mathbf k
       \sum_i f_i\boldsymbol\Omega^\gamma_{i,\alpha\beta}(\mathbf k)

    where :math:`\boldsymbol\Omega^\gamma_{i,\alpha\beta}` and :math:`f_i` are the
    spin Berry curvature and occupation for state :math:`i`.

    The conductivity will be averaged by volume of the periodic unit cell.
    See `~sisl.Lattice.volumef` for details.

    See :cite:`PhysRevB.98.214402` and :cite:`Ji2022` for details on the implementation.

    Parameters
    ----------
    bz :
        containing the integration grid and has the ``bz.parent`` as an instance of Hamiltonian.
    k_average:
        if `True`, the returned quantity is averaged over `bz`, else all k-point
        contributions will be collected.
        Note, for large `bz` integrations this may explode the memory usage.
    sigma:
        which Pauli matrix is used, alternatively one can pass a custom spin matrix,
        or the full sigma.
    J_axes:
        the direction(s) where the :math:`J` operator will be applied (defaults to all).
    distribution :
        An optional distribution enabling one to automatically sum states
        across occupied/unoccupied states.
        Defaults to the step function.
    eigenstate_kwargs :
       keyword arguments passed directly to the ``bz.eigenstate`` method.
       One should *not* pass a ``k`` or a ``wrap`` keyword argument as they are
       already used.
    apply_kwargs :
       keyword arguments passed directly to ``bz.apply.renew(**apply_kwargs)``.
    **berry_kwargs : dict, optional
        arguments passed directly to the `berry_curvature` method.

        Here one can pass `derivative_kwargs` to pass flags to the
        `derivative` method. In particular ``axes`` can be used
        to speedup the calculation (by omitting certain directions).

    Examples
    --------
    For instance, ``sigma = 'x', J_axes = 'y'`` will result in
    :math:`J^{\sigma^x}_y=\dfrac12\{\hat{\sigma}^x, \hat{v}_y\}`, and the rest will
    be the AHC.

    >>> cond = shc(bz, J_axes="y")
    >>> shc_y_xyz = cond[1]
    >>> ahc_xz_xyz = cond[[0, 2]]

    Passing an explicit :math:`\sigma` matrix is also allowed:

    >>> cond = shc(bz)
    >>> assert np.allclose(cond, shc(bz, sigma=Spin.Z))

    For further examples, please see `ahc` which is equivalent to this
    method.

    Notes
    -----
    Original implementation by Armando Pezo.

    See Also
    --------
    ~sisl.physics.derivative: method for calculating the exact derivatives
    berry_curvature: the actual method used internally
    spin_berry_curvature: method used to calculate the Berry-flux for calculating the spin conductivity
    ~sisl.Lattice.volumef: volume calculation of the primary unit cell.
    ahc: anomalous Hall conductivity, this is the equivalent method for the SHC.

    Returns
    -------
    shc: numpy.ndarray
        Spin Hall conductivity returned in certain dimensions ``shc[J_axes, :]``.
        Anomalous Hall conductivity returned in the remaining dimensions ``shc[!J_axes, :]``.
        If `sum` is False, it will be at least a 3D array with the 3rd dimension
        having the contribution from state `i`.
        If `k_average` is False, it will have a dimension prepended with
        k-point resolved AHC/SHC.
        If one passes `axes` to the `derivative_kwargs` argument one will get
        dimensions according to the number of axes requested, by default all
        axes will be used (even if they are non-periodic).
        The dtype will be imaginary.
        When :math:`D` is the dimensionality we find the unit to be

        * AHC: ``shc[!J_axes, :]`` :math:`S/\mathrm{Ang}^{D-2}`.
        * SHC: ``shc[J_axes, :]`` :math:`\hbar/e S/\mathrm{Ang}^{D-2}`.

    """
    from .hamiltonian import Hamiltonian

    if isinstance(J_axes, (tuple, list)):
        J_axes = "".join(J_axes)
    J_axes = J_axes.lower()

    H = bz.parent

    # Currently we require the conductivity calculation to *only* accept Hamiltonians
    if not isinstance(H, Hamiltonian):
        raise SislError(
            "shc: requires the Brillouin zone object to contain a Hamiltonian!"
        )
    # A spin-berry-curvature requires the objects parent
    # to have a spin associated
    if H.spin.is_diagonal:
        raise ValueError(
            f"spin_berry_curvature requires 'state' to be a non-colinear matrix."
        )

    dtype = eigenstate_kwargs.get("dtype", np.complex128)

    if H.spin.is_nambu:
        no = H.no * 2
    else:
        no = H.no
    m = _create_sigma(no, sigma, dtype, eigenstate_kwargs.get("format", "csr"))

    # To reduce (heavily) the computational load, we pre-setup the
    # operators here.
    def J(M, d):
        nonlocal m, J_axes
        if d in J_axes:
            return M @ m + m @ M

        return M

    def noop(M, d):
        return M

    axes = berry_kwargs.get("derivative_kwargs", {}).get("axes", "xyz")
    axes = [direction(axis) for axis in sorted(axes)]

    # At this point we have the AHC (in terms of units)
    cond = ahc(
        bz,
        k_average,
        distribution=distribution,
        eigenstate_kwargs=eigenstate_kwargs,
        apply_kwargs=apply_kwargs,
        **berry_kwargs,
        operator=(J, noop),
    )

    # The SHC misses a factor -2e/hbar to correct the operator change:
    #  j_x = -e v_x, 1/2 {s_z, v_x}
    # The s_z = \hbar / 2 \sigma_z
    # and v = 1/\hbar \delta_k
    #
    # AHC:
    #   j_x = -e / \hbar
    # SHC:
    #   j_x = 1/2 { \hbar/2 \sigma_z, 1/\hbar v_x } = 1/2
    # The 1/\hbar is contained in `berry_curvature`, and hence we
    # are left with:
    # AHC:
    #   j_x = -e
    # SHC:
    #   j_x = 1/2 \hbar
    # Since we never use \hbar or e, it is the same as though
    # the units are implicit. Hence at this point, the unit is:
    #    -\hbar / (2e) S / Ang
    # To convert to \hbar / e S / Ang
    # simply multiply by: -1/2
    shc_idx = [i for i in map(direction, J_axes) if i in axes]
    if k_average:
        cond[shc_idx] *= -0.5
    else:
        cond[:, shc_idx] *= -0.5

    return cond


@set_module("sisl.physics.electron")
@deprecation("conductivity is deprecated, please use 'ahc' instead.")
def conductivity(
    bz,
    distribution: DistributionType = "fermi-dirac",
    method: Literal["ahc"] = "ahc",
    *,
    eigenstate_kwargs={},
    apply_kwargs={},
    **berry_kwargs,
):
    r"""Deprecated, use `ahc` instead"""

    if method != "ahc":
        raise NotImplementedError("conductivity with method != ahc is not implemented")
    return ahc(
        bz,
        eigenstate_kwargs=eigenstate_kwargs,
        apply_kwargs=apply_kwargs,
        distribution=distribution,
        **kwargs,
    )


@set_module("sisl.physics.electron")
def berry_phase(
    contour: BrillouinZone,
    sub=None,
    eigvals: bool = False,
    closed: bool = True,
    method: Literal["berry", "zak", "berry:svd", "zak:svd"] = "berry",
    *,
    ret_overlap: bool = False,
    eigenstate_kwargs: Optional[dict[str, Any]] = None,
    apply_kwargs: Optional[dict[str, Any]] = None,
):
    r""" Calculate the Berry-phase on a loop path

    The Berry phase for a single Bloch state is calculated using the discretized formula:

    .. math::
       \mathbf S = \prod_\alpha^{N-1} \langle \psi_{\mathbf k_\alpha} | \psi_{\mathbf k_{\alpha+1}} \rangle
       \\
       \phi = - \Im\ln \mathrm{det} \mathbf S

    where :math:`\langle \psi_{\mathbf k_\alpha} | \psi_{\mathbf k_{\alpha+1}} \rangle` may be exchanged with an overlap matrix
    of the investigated bands. I.e. :math:`\psi` is a manifold set of wavefunctions.
    The overlap matrix :math:`\mathbf S` is also known as the global unitary
    rotation matrix corresponding to the maximally localized Wannier centers.

    If `closed` is true the overlap matrix will also include the circular inner product:

    .. math::
       \mathbf S^{\mathcal C} = \mathbf S \langle \psi_{\mathbf k_N} | \psi_{\mathbf k_1} \rangle


    Parameters
    ----------
    contour :
       containing the closed contour and has the ``contour.parent`` as an instance of Hamiltonian. The
       first and last k-point must not be the same.
    sub : None or list of int, optional
       selected bands to calculate the Berry phase of
    eigvals :
       return the eigenvalues of the product of the overlap matrices
    closed :
       whether or not to include the connection of the last and first points in the loop
       Forced true for Zak-phase calculations.
    method :
       "berry" will return the usual integral of the Berry connection over the specified contour
       "zak" will compute the Zak phase for 1D systems by performing
       a closed loop integration, see :cite:`Zak1989`.
       Additionally, one may do the Berry-phase calculation using the SVD method of the
       overlap matrices. Simply append ":svd" to the chosen method, e.g. "berry:svd".
    ret_overlap:
       optionally return the overlap matrix :math:`\mathbf S`
    eigenstate_kwargs : dict, optional
       keyword arguments passed directly to the ``contour.eigenstate`` method.
       One should *not* pass ``k`` as that is already used.
    eigenstate_kwargs :
       keyword arguments passed directly to the ``contour.eigenstate`` method.
       One should *not* pass ``k`` as that is already used.
    apply_kwargs :
       keyword arguments passed directly to ``contour.apply.renew(**apply_kwargs)``.

    Notes
    -----
    The Brillouin zone object *need* not contain a closed discretized contour by doubling the first point.

    The implementation is very similar to PythTB, except we are here using the :math:`\mathbf R` gauge
    (convention II according to PythTB), see discussion in :pull:`131`.

    For systems with band-crossings or degenerate states there is an arbitrariness to the definition
    of the Berry phase for *individual* bands. However, the total phase (i.e., sum over filled bands) is
    invariant and unaffected by this arbitrariness as long as the filled and empty bands do not intersect,
    see :cite:`Resta2000`.

    For non-orthogonal basis sets it is not fully known how important the :math:`\delta\mathbf k` spacing is since
    it relies on the Lowdin transformation of the states. However, one should be careful about choosing
    the correct bands for examination.

    The returned angles are _not_ placed in the interval :math:`]-\pi;\pi]` as what `numpy.angle` would do.
    This is to allow users to examine the quantities as is.

    For more understanding of the Berry-phase and its calculation :cite:`TopInvTut` is a good reference.

    Examples
    --------

    Calculate Berry-phase for first band but using the SVD method

    >>> N = 30
    >>> kR = 0.01
    >>> normal = [0, 0, 1]
    >>> origin = [1/3, 2/3, 0]
    >>> contour = BrillouinZone.param_circle(H, N, kR, normal, origin)
    >>> phase = berry_phase(contour, sub=0)

    Calculate the multi-band Berry-phase using the SVD method, thus
    ensuring removal of singular vectors.

    >>> phase = berry_phase(contour, method="berry:svd")
    """
    from .hamiltonian import Hamiltonian

    # Currently we require the Berry phase calculation to *only* accept Hamiltonians
    if not isinstance(contour.parent, Hamiltonian):
        raise SislError(
            "berry_phase: requires the Brillouin zone object to contain a Hamiltonian!"
        )

    if eigenstate_kwargs is None:
        eigenstate_kwargs = {}
    if apply_kwargs is None:
        apply_kwargs = {}

    if contour.parent.orthogonal:

        def _lowdin(state):
            pass

    else:
        gauge = eigenstate_kwargs.get("gauge", "lattice")

        def _lowdin(state):
            """change state to the lowdin state, assuming everything is in lattice gauge
            So needs to be done before changing gauge"""
            S12 = sqrth(
                state.parent.Sk(state.info["k"], gauge=gauge, format="array"),
                overwrite_a=True,
            )
            state.state[:, :] = (S12 @ state.state.T).T

    method, *opts = method.lower().split(":")
    if method == "berry":
        pass
    elif method == "zak":
        closed = True
    else:
        raise ValueError("berry_phase: requires the method to be [berry, zak]")

    # We calculate the final angle from the determinant
    _process = dot

    if "svd" in opts:

        def _process(prd, overlap):
            U, _, V = svd_destroy(overlap)
            # We have to use dot, since @ does not allow scalars
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
                prd = _process(prd, prev.inner(second, projection="matrix"))
                prev = second

            # Complete the loop
            if closed:
                # Include last-to-first segment
                prd = _process(prd, prev.inner(first, projection="matrix"))
            return prd

    else:

        def _berry(eigenstates):
            nonlocal sub
            first = next(eigenstates)
            first.sub(sub, inplace=True)
            _lowdin(first)
            prev = first
            prd = 1
            for second in eigenstates:
                second.sub(sub, inplace=True)
                _lowdin(second)
                prd = _process(prd, prev.inner(second, projection="matrix"))
                prev = second
            if closed:
                prd = _process(prd, prev.inner(first, projection="matrix"))
            return prd

    S = _berry(contour.apply.renew(**apply_kwargs).iter.eigenstate(**eigenstate_kwargs))

    # Get the angle of the berry-phase
    # When using np.angle the returned value is in ]-pi; pi]
    # However, small numerical differences makes wrap-arounds annoying.
    # We'll always return the full angle. Then users can them-selves control
    # how to convert them.
    if eigvals:
        ret = -log(la_eigvals(S, overwrite_a=not ret_overlap)).imag
        ret = sort(ret)
    else:
        ret = -log(det(S, overwrite_a=not ret_overlap)).imag

    if ret_overlap:
        return ret, S
    return ret


@set_module("sisl.physics.electron")
def wavefunction(
    v, grid, geometry=None, k=None, spinor=0, spin: Optional[Spin] = None, eta=None
):
    r"""Add the wave-function (`Orbital.psi`) component of each orbital to the grid

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
       \psi(\mathbf r) = \sum_i\phi_i(\mathbf r) |\psi\rangle_i \exp(i\mathbf k\cdot\mathbf R)

    While for non-colinear/spin-orbit calculations the wavefunctions are determined from the
    spinor component (`spinor`)

    .. math::
       \psi_{\alpha/\beta}(\mathbf r) = \sum_i\phi_i(\mathbf r) |\psi_{\alpha/\beta}\rangle_i \exp(i\mathbf k\cdot \mathbf R)

    where ``spinor in [0, 1]`` determines :math:`\alpha` or :math:`\beta`, respectively.

    Notes
    -----
    Currently this method only works for `v` being coefficients of the ``gauge="lattice"`` method. In case
    you are passing a `v` with the incorrect gauge you will find a phase-shift according to:

    .. math::
        \tilde v_j = e^{i\mathbf k\cdot\mathbf r_j} v_j

    where :math:`j` is the orbital index and :math:`\mathbf r_j` is the orbital position.


    Parameters
    ----------
    v : array_like
       coefficients for the orbital expansion on the real-space grid.
       If `v` is a complex array then the `grid` *must* be complex as well. The coefficients
       must be using the *lattice* gauge.
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
    spin :
       specification of the spin configuration of the orbital coefficients. This only has
       influence for non-colinear wavefunctions where `spinor` choice is important.
    eta : bool, optional
       Display a console progressbar.
    """
    # Decipher v from State type
    if isinstance(v, State):
        if geometry is None:
            geometry = v._geometry()
        if k is None:
            k = v.info.get("k", k)
        elif not np.allclose(k, v.info.get("k", k)):
            raise ValueError(
                f"wavefunction: k passed and k in info does not match: {k} and {v.info.get('k')}"
            )
        v = v.state
    if geometry is None:
        geometry = grid.geometry
    if geometry is None:
        raise SislError(
            "wavefunction: did not find a usable Geometry through keywords or the Grid!"
        )

    # We cannot move stuff since outside stuff may rely on exact coordinates.
    # If people have out-liers, they should do it them-selves.
    # We'll do this and warn if they are dissimilar.
    dxyz = geometry.lattice.cell2length(1e-6).sum(0)
    dxyz = (
        geometry.move(dxyz).translate2uc(axes=(0, 1, 2)).move(-dxyz).xyz - geometry.xyz
    )
    if not np.allclose(dxyz, 0):
        info(
            f"wavefunction: coordinates may be outside your primary unit-cell. "
            "Translating all into the primary unit cell could disable this information"
        )

    # In case the user has passed several vectors we sum them to plot the summed state
    if v.ndim == 2:
        if v.shape[0] > 1:
            info(
                f"wavefunction: summing {v.shape[0]} different state coefficients, will continue silently!"
            )
        v = v.sum(0)

    if spin is None:
        if len(v) // 2 == geometry.no:
            # the input corresponds to a non-collinear calculation
            v = v.reshape(-1, 2)[:, spinor]
            info(
                "wavefunction: assumes the input wavefunction coefficients to originate from a non-colinear calculation!"
            )
        elif len(v) // 4 == geometry.no:
            # the input corresponds to a NAMBU calculation
            v = v.reshape(-1, 4)[:, spinor]
            info(
                "wavefunction: assumes the input wavefunction coefficients to originatefrom a nambu calculation!"
            )

    elif spin.kind > Spin.POLARIZED:
        # For non-colinear+nambu cases the user selects the spinor component.
        v = v.reshape(-1, spin.spinor)[:, spinor]

    if len(v) != geometry.no:
        raise ValueError(
            "wavefunction: require wavefunction coefficients corresponding to number of orbitals in the geometry."
        )

    # Check for k-points
    k = _a.asarrayd(k)
    kl = k.dot(k) ** 0.5
    has_k = kl > 0.000001
    if has_k:
        info("wavefunction: k != Gamma is currently untested!")

    # Check that input/grid makes sense.
    # If the coefficients are complex valued, then the grid *has* to be
    # complex valued.
    # Likewise if a k-point has been passed.
    is_complex = np.iscomplexobj(v) or has_k
    if is_complex and not np.iscomplexobj(grid.grid):
        raise SislError(
            "wavefunction: input coefficients are complex, while grid only contains real."
        )

    if is_complex:
        psi_init = _a.zerosz
    else:
        psi_init = _a.zerosd

    # Extract sub variables used throughout the loop
    shape = _a.asarrayi(grid.shape)
    dcell = grid.dcell
    ic_shape = grid.lattice.icell * shape.reshape(3, 1)

    # Convert the geometry (hosting the wavefunction coefficients) coordinates into
    # grid-fractionals X grid-shape to get index-offsets in the grid for the geometry
    # supercell.
    geom_shape = geometry.cell @ ic_shape.T

    # In the following we don't care about division
    # So 1) save error state, 2) turn off divide by 0, 3) calculate, 4) turn on old error state
    old_err = np.seterr(divide="ignore", invalid="ignore")

    addouter = add.outer

    def idx2spherical(ix, iy, iz, offset, dc, R):
        """Calculate the spherical coordinates from indices"""
        rx = addouter(
            addouter(ix * dc[0, 0], iy * dc[1, 0]), iz * dc[2, 0] - offset[0]
        ).ravel()
        ry = addouter(
            addouter(ix * dc[0, 1], iy * dc[1, 1]), iz * dc[2, 1] - offset[1]
        ).ravel()
        rz = addouter(
            addouter(ix * dc[0, 2], iy * dc[1, 2]), iz * dc[2, 2] - offset[2]
        ).ravel()

        # Total size of the indices
        n = rx.shape[0]
        # Reduce our arrays to where the radius is "fine"
        idx = indices_le(rx**2 + ry**2 + rz**2, R**2)
        rx = rx[idx]
        ry = ry[idx]
        rz = rz[idx]
        xyz_to_spherical_cos_phi(rx, ry, rz)
        return n, idx, rx, ry, rz

    # Figure out the max-min indices with a spacing of 1 radian
    # calculate based on the minimum length of the grid-spacing
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
    idx = rxyz @ ic_shape.T
    idxm = idx.min(0)
    idxM = idx.max(0)
    del ctheta_sphi, stheta_sphi, cphi, idx, rxyz, nrxyz

    # Fast loop (only per specie)
    origin = grid.lattice.origin
    idx_mm = _a.emptyd([geometry.na, 2, 3])
    all_negative_R = True
    for atom, ia in geometry.atoms.iter(True):
        if len(ia) == 0:
            continue
        R = atom.maxR()
        all_negative_R = all_negative_R and R < 0.0

        # Now do it for all the atoms to get indices of the middle of
        # the atoms
        # The coordinates are relative to origin, so we need to shift (when writing a grid
        # it is with respect to origin)
        idx = (geometry.xyz[ia, :] - origin) @ ic_shape.T

        # Get min-max for all atoms
        idx_mm[ia, 0, :] = idxm * R + idx
        idx_mm[ia, 1, :] = idxM * R + idx

    if all_negative_R:
        raise SislError(
            "wavefunction: Cannot create wavefunction since no atoms have an associated basis-orbital on a real-space grid"
        )

    # Now we have min-max for all atoms
    # When we run the below loop all indices can be retrieved by looking
    # up in the above table.
    # Before continuing, we can easily clean up the temporary arrays
    del origin, idx

    arangei = _a.arangei

    # In case this grid does not have a Geometry associated
    # We can *perhaps* easily attach a geometry with the given
    # atoms in the unit-cell
    lattice = grid.lattice.copy()

    # Find the periodic directions
    pbc = [
        bc == BC.PERIODIC or geometry.nsc[i] > 1
        for i, bc in enumerate(grid.lattice.boundary_condition[:, 0])
    ]
    if grid.geometry is None:
        # Create the actual geometry that encompass the grid
        ia, xyz, _ = geometry.within_inf(lattice, periodic=pbc)
        if len(ia) > 0:
            grid.set_geometry(Geometry(xyz, geometry.atoms[ia], lattice=lattice))

    # Instead of looping all atoms in the supercell we find the exact atoms
    # and their supercell indices.
    # plus some tolerance
    add_R = _a.fulld(3, geometry.maxR()) + 1.0e-6
    # Calculate the required additional vectors required to increase the fictitious
    # supercell by add_R in each direction.
    # For extremely skewed lattices this will be way too much, hence we make
    # them square.

    o = lattice.to.Cuboid(orthogonal=True)
    lattice = Lattice(o._v + np.diag(2 * add_R), origin=o.origin - add_R)

    # Retrieve all atoms within the grid supercell
    # (and the neighbors that connect into the cell)
    # Note that we cannot pass the "moved" origin because then ISC would be wrong
    IA, XYZ, ISC = geometry.within_inf(lattice, periodic=pbc)
    # We need to revert the grid supercell origin as that is not subtracted in the `within_inf` returned
    # coordinates (and the below loop expects positions with respect to the origin of the plotting
    # grid).
    XYZ -= grid.lattice.origin

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
        if R <= 0.0:
            warn(
                f"wavefunction: Atom '{atom}' does not have a wave-function, skipping atom."
            )
            eta.update()
            continue

        # Get indices in the supercell grid
        idx = (isc.reshape(3, 1) * geom_shape).sum(0)
        idxm = floor(idx_mm[ia, 0, :] + idx).astype(int32)
        idxM = ceil(idx_mm[ia, 1, :] + idx).astype(int32) + 1

        # Fast check whether we can skip this point
        if (
            idxm[0] >= shape[0]
            or idxm[1] >= shape[1]
            or idxm[2] >= shape[2]
            or idxM[0] <= 0
            or idxM[1] <= 0
            or idxM[2] <= 0
        ):
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
        n, idx, r, theta, phi = idx2spherical(
            arangei(idxm[0], idxM[0]),
            arangei(idxm[1], idxM[1]),
            arangei(idxm[2], idxM[2]),
            xyz,
            dcell,
            R,
        )

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

            if oR <= 0.0:
                warn(
                    f"wavefunction: Orbital(s) '{os}' does not have a wave-function, skipping orbital!"
                )
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
                psi[idx1] += o.psi_spher(r1, theta1, phi1, cos_phi=True) * (
                    v[io] * phase
                )
                io += 1

        # Clean-up
        del idx1, r1, theta1, phi1, idx, r, theta, phi

        # Convert to correct shape and add the current atom contribution to the wavefunction
        psi.shape = idxM - idxm
        grid.grid[idxm[0] : idxM[0], idxm[1] : idxM[1], idxm[2] : idxM[2]] += psi

        # Clean-up
        del psi

        # Step progressbar
        eta.update()

    eta.close()

    # Reset the error code for division
    np.seterr(**old_err)


class _electron_State:
    # pylint: disable=E1101
    __slots__ = []

    def __is_nc(self, spin: Optional[Spin] = None):
        """Internal routine to check whether this is a non-colinear calculation"""
        if spin is not None:
            return not spin.is_diagonal
        try:
            return not self.parent.spin.is_diagonal
        except Exception:
            return False

    def Sk(self, format=None):
        r"""Retrieve the overlap matrix corresponding to the originating parent structure.

        When ``self.parent`` is a Hamiltonian this will return :math:`\mathbf S(\mathbf k)` for the
        :math:`\mathbf k`-point these eigenstates originate from.

        Parameters
        ----------
        format : str, optional
           the returned format of the overlap matrix. This only takes effect for
           non-orthogonal parents.
        """
        if format is None:
            format = self.info.get("format", "csr")
        if isinstance(self.parent, SparseOrbitalBZSpin):
            # Calculate the overlap matrix
            if not self.parent.orthogonal:
                opt = {
                    "k": self.info.get("k", (0, 0, 0)),
                    "dtype": self.dtype,
                    "format": format,
                }
                for key in ("gauge",):
                    val = self.info.get(key, None)
                    if not val is None:
                        opt[key] = val
                return self.parent.Sk(**opt)

        n = m = self.shape[1]
        if "sc:" in format:
            m = n * self.parent.n_s

        return _FakeMatrix(n, m)

    @deprecate_argument(
        "sum",
        "projection",
        "argument sum has been deprecated in favor of projection",
        "0.15",
        "0.17",
    )
    def norm2(
        self,
        projection: Union[
            ProjectionType, ProjectionTypeHadamard, ProjectionTypeHadamardAtoms
        ] = "diagonal",
    ):
        r"""Return a vector with the norm of each state :math:`\langle\psi|\mathbf S|\psi\rangle`

        :math:`\mathbf S` is the overlap matrix (or basis), for orthogonal basis
        :math:`\mathbf S \equiv \mathbf I`.

        Parameters
        ----------
        projection :
           whether to compute the norm per state as a single number or as orbital-/atom-resolved quantity

        See Also
        --------
        inner: used method for calculating the squared norm.

        Returns
        -------
        numpy.ndarray
            the squared norm for each state
        """
        return self.inner(matrix=self.Sk(), projection=projection)

    @deprecate_argument(
        "project",
        "projection",
        "argument project has been deprecated in favor of projection",
        "0.15",
        "0.17",
    )
    def spin_moment(self, projection="diagonal"):
        r"""Calculate spin moment from the states

        This routine calls `~sisl.physics.electron.spin_moment` with appropriate arguments
        and returns the spin moment for the states.

        See `~sisl.physics.electron.spin_moment` for details.

        Parameters
        ----------
        projection:
           whether the moments are orbitally resolved or not
        """
        return spin_moment(self.state, self.Sk(), projection=projection)

    def wavefunction(self, grid, spinor=0, eta=None):
        r"""Expand the coefficients as the wavefunction on `grid` *as-is*

        See `~sisl.physics.electron.wavefunction` for argument details, the arguments not present
        in this method are automatically passed from this object.
        """
        spin = getattr(self.parent, "spin", None)

        if isinstance(self.parent, Geometry):
            geometry = self.parent
        else:
            geometry = getattr(self.parent, "geometry", None)

        if not isinstance(grid, Grid):
            # probably the grid is a Real, or a tuple that denotes the shape
            # at least this makes it easier to parse
            grid = Grid(grid, geometry=geometry, dtype=self.dtype)

        # Ensure we are dealing with the lattice gauge
        self.change_gauge("lattice")

        # Retrieve k
        k = self.info.get("k", _a.zerosd(3))

        wavefunction(
            self.state, grid, geometry=geometry, k=k, spinor=spinor, spin=spin, eta=eta
        )


@set_module("sisl.physics.electron")
class CoefficientElectron(Coefficient):
    r"""Coefficients describing some physical quantity related to electrons"""

    __slots__ = []


@set_module("sisl.physics.electron")
class StateElectron(_electron_State, State):
    r"""A state describing a physical quantity related to electrons"""

    __slots__ = []


@set_module("sisl.physics.electron")
class StateCElectron(_electron_State, StateC):
    r"""A state describing a physical quantity related to electrons, with associated coefficients of the state"""

    __slots__ = []

    def effective_mass(self, *args, **kwargs):
        r"""Calculate effective mass tensor for the states, units are (ps/Ang)^2

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
        np.divide(_velocity_const**2, ieff, where=(ieff != 0), out=ieff)
        return ieff


@set_module("sisl.physics.electron")
class EigenvalueElectron(CoefficientElectron):
    r"""Eigenvalues of electronic states, no eigenvectors retained

    This holds routines that enable the calculation of density of states.
    """

    __slots__ = []

    @property
    def eig(self):
        """Eigenvalues"""
        return self.c

    def occupation(self, distribution: DistributionType = "fermi_dirac"):
        r"""Calculate the occupations for the states according to a distribution function

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
        return distribution(self.eig)

    def DOS(self, E, distribution: DistributionType = "gaussian"):
        r"""Calculate DOS for provided energies, `E`.

        This routine calls `sisl.physics.electron.DOS` with appropriate arguments
        and returns the DOS.

        See `~sisl.physics.electron.DOS` for argument details.
        """
        return DOS(E, self.eig, distribution)


@set_module("sisl.physics.electron")
class EigenvectorElectron(StateElectron):
    r"""Eigenvectors of electronic states, no eigenvalues retained

    This holds routines that enable the calculation of spin moments.
    """

    __slots__ = []


@set_module("sisl.physics.electron")
class EigenstateElectron(StateCElectron):
    r"""Eigen states of electrons with eigenvectors and eigenvalues.

    This holds routines that enable the calculation of (projected) density of states,
    spin moments (spin texture).
    """

    __slots__ = []

    @property
    def eig(self):
        r"""Eigenvalues for each state"""
        return self.c

    def occupation(self, distribution: DistributionType = "fermi_dirac"):
        r"""Calculate the occupations for the states according to a distribution function

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
        return distribution(self.eig)

    def DOS(self, E, distribution: DistributionType = "gaussian"):
        r"""Calculate DOS for provided energies, `E`.

        This routine calls `sisl.physics.electron.DOS` with appropriate arguments
        and returns the DOS.

        See `~sisl.physics.electron.DOS` for argument details.
        """
        return DOS(E, self.c, distribution)

    def PDOS(self, E, distribution: DistributionType = "gaussian"):
        r"""Calculate PDOS for provided energies, `E`.

        This routine calls `~sisl.physics.electron.PDOS` with appropriate arguments
        and returns the PDOS.

        See `~sisl.physics.electron.PDOS` for argument details.
        """
        return PDOS(
            E,
            self.c,
            self.state,
            self.Sk(),
            distribution,
            getattr(self.parent, "spin", None),
        )

    def COP(self, E, M, *args, **kwargs):
        r"""Calculate COP for provided energies, `E` using matrix `M`

        This routine calls `~sisl.physics.electron.COP` with appropriate arguments.
        """
        return COP(E, self.c, self.state, M, *args, **kwargs)

    def COOP(self, E, *args, **kwargs):
        r"""Calculate COOP for provided energies, `E`.

        This routine calls `~sisl.physics.electron.COP` with appropriate arguments.
        """
        format = self.info.get("format", "csr")
        Sk = self.Sk(format=f"sc:{format}")
        return COP(E, self.c, self.state, Sk, *args, **kwargs)

    def COHP(self, E, *args, **kwargs):
        r"""Calculate COHP for provided energies, `E`.

        This routine calls `~sisl.physics.electron.COP` with appropriate arguments.
        """
        format = self.info.get("format", "csr")
        Hk = self.parent.Hk(format=f"sc:{format}")
        return COP(E, self.c, self.state, Hk, *args, **kwargs)
