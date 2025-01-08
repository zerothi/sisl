# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from sisl._ufuncs import register_sisl_dispatch
from sisl.typing import DistributionType

from .distribution import get_distribution
from .electron import StateCElectron, _create_sigma, _velocity_const
from .state import _dM_Operator

# Nothing gets exposed here
__all__ = []


@register_sisl_dispatch(StateCElectron, module="sisl.physics")
def velocity(state: StateCElectron, *args, **kwargs):
    r"""Calculate velocity for the states

    This routine calls ``derivative(1, *args, **kwargs)`` and returns the velocity for the states.

    Note that the coefficients associated with the `StateCElectron` *must* correspond
    to the energies of the states.

    The unit is Ang/ps.

    Notes
    -----
    The velocities are calculated without the Berry curvature contribution see Eq. (2) in :cite:`Wang2006`.
    It is thus typically denoted as the *effective velocity operater* (see Ref. 21 in :cite:`Wang2006`.
    The missing contribution may be added in later editions, for completeness sake, it is:

    .. math::
       \delta \mathbf v = - \mathbf k\times \Omega_i(\mathbf k)

    where :math:`\Omega_i` is the Berry curvature for state :math:`i`.

    Parameters
    ----------
    *args, **kwargs :
        arguments passed directly to `derivative`, see that method
        for argument details.

    See Also
    --------
    derivative : for details of the implementation
    """
    v = state.derivative(1, *args, **kwargs)
    v *= _velocity_const
    return v


@register_sisl_dispatch(StateCElectron, module="sisl.physics")
def berry_curvature(
    state: StateCElectron,
    sum: bool = True,
    *,
    distribution: Optional[DistributionType] = None,
    derivative_kwargs: dict = {},
    operator: Union[_dM_Operator, tuple[_dM_Operator, _dM_Operator]] = lambda M, d: M,
    eta: float = 0.0,
) -> np.ndarray:
    r"""Calculate the Berry curvature matrix for a set of states (Kubo)

    The Berry curvature is calculated using the following expression
    (:math:`\alpha`, :math:`\beta` corresponding to Cartesian directions):

    .. math::

       \boldsymbol\Omega_{i,\alpha\beta} = 2i\hbar^2\sum_{j\neq i}
                \frac{v^{\alpha}_{ij} v^\beta_{ji}}
                     {[\epsilon_i - \epsilon_j]^2 + i\eta^2}

    For details on the Berry curvature, see Eq. (11) in :cite:`Wang2006`
    or Eq. (2.59) in :cite:`TopInvCourse`.

    The `operator` argument can be used to define the Berry curvature in other
    quantities. E.g. the spin Berry curvature is defined by replacing :math:`v^\alpha`
    by the spin current operator. see `spin_berry_curvature` for details.

    For additional details on the spin Berry curvature, see Eq. (1) in
    :cite:`PhysRevB.98.214402` and Eq. (2) in :cite:`Ji2022`.

    Notes
    -----
    There exists reports on some terms missing in the above formula, for details
    see :cite:`gan2021calculation`.

    Parameters
    ----------
    state :
        the state describing the electronic states we wish to calculate the Berry curvature
        of.
    sum :
        only return the summed Berry curvature (over all states).
    distribution :
        An optional distribution enabling one to automatically sum states
        across occupied/unoccupied states. This is useful when calculating AHC/SHC
        contributions since it can improve numerical accuracy.
        If this is None, it will do the above equation exactly.
    derivative_kwargs :
        arguments passed to `derivative`. Since `operator` is defined here,
        one cannot have `operator` in `derivative_kwargs`.
    operator :
        the operator to use for changing the `dPk` matrices.
        Note, that this may change the resulting units, and it will be up
        to the user to adapt the units accordingly.
    eta :
        direct imaginary part broadening of the Lorentzian.

    See Also
    --------
    derivative: method for calculating the exact derivatives
    spin_berry_curvature : calculate the spin Berry curvature
    ahc: anomalous Hall conductivity
    shc: spin Hall conductivity

    Returns
    -------
    bc :
        If `sum` is False, it will be at least a 3D array with the 3rd dimension
        having the contribution from state `i`.
        If one passes `axes` to the `derivative_kwargs` argument one will get
        dimensions according to the number of axes requested, by default all
        axes will be used (even if they are non-periodic).
        The dtype will be imaginary.
        The unit is :math:`\mathrm{Ang}^2`.
    """
    # cast dtypes to *any* complex valued data-type that can be expressed
    # minimally by a complex64 object

    if isinstance(distribution, str):
        distribution = get_distribution(distribution)

    if isinstance(operator, (tuple, list)):
        opA = operator[0]
        opB = operator[1]
    else:
        opA = operator
        opB = operator

    if opA is opB:

        # same operator
        dA = dB = state.derivative(
            order=1, operator=opA, matrix=True, **derivative_kwargs
        )

    else:
        # different operators
        dA = state.derivative(order=1, operator=opA, matrix=True, **derivative_kwargs)
        dB = state.derivative(order=1, operator=opB, matrix=True, **derivative_kwargs)

    ieta2 = 1j * eta**2
    energy = state.c

    ndA = len(dA)
    ndB = len(dB)

    if distribution is None:
        if sum:
            sigma = np.zeros((ndA, ndB), dtype=dA.dtype)

            for si, ei in enumerate(energy):
                de = (ei - energy) ** 2 + ieta2
                np.divide(2, de, where=(de != 0), out=de)

                for iA in range(ndA):
                    for iB in range(ndB):
                        sigma[iA, iB] += 1j * de @ (dA[iA, si] * dB[iB, :, si])

        else:
            sigma = np.zeros((ndA, ndB, len(energy)), dtype=dA.dtype)

            for si, ei in enumerate(energy):
                de = (ei - energy) ** 2 + ieta2
                np.divide(2, de, where=(de != 0), out=de)

                for iA in range(ndA):
                    for iB in range(ndB):
                        sigma[iA, iB, si] += 1j * de @ (dA[iA, si] * dB[iB, :, si])

    else:
        # when calculating the distribution, one should always
        # find the energy dimension along the last axis, so x0
        # must have a shape (-1, 1) to allow b-casting!
        # Hence the last dimension of distribution(energy) corresponds
        # to the states.
        # Hence we transpose it for direct usage below.
        dist_e = distribution(energy)
        # Since we are using the double counting dist_e[si] - dist_e
        # we have to omit the factor 2 here.

        if sum:
            dsigma_shape = (ndA, ndB) + (1,) * (dist_e.ndim - 1)

            # then it will be: [3, 3[, dist.shape]]
            shape = np.broadcast_shapes(dist_e.shape[:-1], dsigma_shape)
            sigma = np.zeros(shape, dtype=dA.dtype)

            for si, ei in enumerate(energy):
                de = (ei - energy) ** 2 + ieta2
                np.divide(1, de, where=(de != 0), out=de)
                # the order of this term can be found in:
                # 10.21468/SciPostPhysCore.6.1.002 Eq. 29
                dd = dist_e[..., [si]] - dist_e

                for iA in range(ndA):
                    for iB in range(ndB):
                        dsigma = 1j * (de * dA[iA, si] * dB[iB, :, si])

                        sigma[iA, iB] += dd @ dsigma

        else:
            dsigma_shape = (ndA, ndB, len(energy)) + (1,) * (dist_e.ndim - 1)

            # then it will be: [3, 3, nstates[, dist.shape]]
            shape = np.broadcast_shapes(dist_e.shape[:-1], dsigma_shape)
            sigma = np.zeros(shape, dtype=dA.dtype)

            for si, ei in enumerate(energy):
                de = (ei - energy) ** 2 + ieta2
                np.divide(1, de, where=(de != 0), out=de)
                dd = dist_e[..., [si]] - dist_e

                for iA in range(ndA):
                    for iB in range(ndB):
                        dsigma = 1j * (de * dA[iA, si] * dB[iB, :, si])

                        sigma[iA, iB, si] += dd @ dsigma

    # When the operators are the simple velocity operators, then
    # we don't need to do anything for the units.
    # The velocity operator is 1/hbar \hat v
    # and the pre-factor of hbar ^2 means they cancel out.
    return sigma


@register_sisl_dispatch(StateCElectron, module="sisl.physics")
def spin_berry_curvature(
    state: StateCElectron,
    sigma: Union[CartesianAxisStrLiteral, npt.ArrayLike] = "z",
    sum: bool = True,
    *,
    distribution: Optional[DistributionType] = None,
    J_axes: Union[CartesianAxisStrLiteral, Sequence[CartesianAxisStrLiteral]] = "xyz",
    **berry_kwargs,
) -> np.ndarray:
    r"""Calculate the spin Berry curvature

    This is equivalent to calling `berry_curvature`
    with the spin current operator and the regular velocity
    operator instead of :math:`v^\alpha`:

    .. code-block:: python

        def noop(M, d): return M
        def Jz(M, d):
            if d in J_axes:
                return (M @ sigma_z + sigma_z @ M) * 0.5
            return M

        state.berry_curvature(..., operator=(Jz, noop))

    I.e. the *left* velocity operator being swapped with the
    spin current operator:

    .. math::

        J^{\gamma\alpha} = \frac12 \{ v^\alpha, \hat{\sigma}^\gamma \}

    where :math:`\{\}` means the anticommutator.

    When calling it like this, the spin Berry curvature is found in the
    index corresponding to the axes the spin operator is acting on. The regular
    spin Berry curvature is found in all indices (`J_axes`).

    E.g. if ``J_axes = 'xy', sigma = 'z'``, then ``shc[[0, 1]]`` will be the
    spin Berry curvature using the Pauli matrix
    :math:`\hat{\sigma}^z` (not the spin-operator
    :math:`\hat{s}^z = \dfrac\hbar2\hat{\sigma}^z`),
    and ``shc[2]`` will be the *normal* Berry curvature (since only
    the left velocity operator will be changed for `J_axes`).

    Notes
    -----
    For performance reasons, it can be very benificial to extract the
    above methods and call `berry_curvature` directly.
    This is because the :math:`\sigma` operator gets created on every
    call of this method.

    This, repeated matrix creation, might change in the future.

    Parameters
    ----------
    sigma :
        which Pauli matrix is used, alternatively one can pass a custom spin matrix,
        or the full sigma.
    J_axes :
        the direction(s) where the :math:`J^\sigma` operator will be applied, defaults
        to all.
    **kwargs :
        see `berry_curvature` for the remaining arguments.

    See Also
    --------
    berry_curvature : calculate the Berry curvature (internally called)
    derivative: method for calculating the exact derivatives
    ahc: anomalous Hall conductivity
    shc: spin Hall conductivity

    Returns
    -------
    bc :
        Spin Berry curvature + (possibly Berry curvature) returned in certain dimensions.
        If one passes `axes` to the `derivative_kwargs` argument one will get
        dimensions according to the number of axes requested, by default all
        axes will be used (even if they are non-periodic).
        The dtype will be imaginary.
        The unit is :math:`\mathrm{Ang}^2`.
    """
    H = state.parent

    if isinstance(J_axes, (tuple, list)):
        J_axes = "".join(J_axes)
    J_axes = J_axes.lower()

    # A spin-berry-curvature requires the objects parent
    # to have a spin associated
    if H.spin.is_diagonal:
        raise ValueError(
            f"spin_berry_curvature requires 'state' to be a non-colinear matrix."
        )

    dtype = np.result_type(state.dtype, state.info.get("dtype", np.complex128))

    # no is not including the spin-dimension
    if H.spin.is_nambu:
        no = H.no * 2
    else:
        no = H.no
    m = _create_sigma(no, sigma, dtype, state.info.get("format", "csr"))

    def J(M, d):
        nonlocal m, J_axes
        if d in J_axes:
            return M @ m + m @ M
        return M

    def noop(M, d):
        return M

    return berry_curvature(
        state,
        sum=sum,
        distribution=distribution,
        **berry_kwargs,
        operator=(J, noop),
    )
