# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Optional

import numpy as np

import sisl._array as _a
from sisl._ufuncs import register_sisl_dispatch
from sisl.typing import SimpleIndex

from .state import Coefficient, State, StateC

# Nothing gets exposed here
__all__ = []


@register_sisl_dispatch(Coefficient, module="sisl.physics")
def copy(coefficient: Coefficient) -> Coefficient:
    """Return a copy (only the coefficients are copied). ``parent`` and ``info`` are passed by reference"""
    out = coefficient.__class__(coefficient.c.copy(), coefficient.parent)
    out.info = coefficient.info
    return out


@register_sisl_dispatch(State, module="sisl.physics")
def copy(state: State) -> State:
    """Return a copy (only the state is copied). ``parent`` and ``info`` are passed by reference"""
    out = state.__class__(state.state.copy(), state.parent)
    out.info = state.info
    return out


@register_sisl_dispatch(StateC, module="sisl.physics")
def copy(statec: StateC) -> StateC:
    """Return a copy (only the coefficients and states are copied), ``parent`` and ``info`` are passed by reference"""
    out = statec.__class__(statec.state.copy(), statec.c.copy(), statec.parent)
    out.info = statec.info
    return out


@register_sisl_dispatch(Coefficient, module="sisl.physics")
def sub(
    coefficient: Coefficient, index: SimpleIndex, inplace: bool = False
) -> Optional[Coefficient]:
    """Return a new coefficient with only the specified coefficients

    Parameters
    ----------
    index :
        indices that are retained in the returned object
    inplace :
        whether the values will be retained inplace

    Returns
    -------
    Coefficient
        a new coefficient only containing the requested elements, only if `inplace` is false
    """
    index = coefficient._sanitize_index(index)
    if inplace:
        coefficient.c = coefficient.c[index]
    else:
        out = coefficient.__class__(coefficient.c[index], coefficient.parent)
        out.info = coefficient.info
        return out


@register_sisl_dispatch(Coefficient, module="sisl.physics")
def remove(
    coefficient: Coefficient, index: SimpleIndex, inplace: bool = False
) -> Optional[Coefficient]:
    """Return a new coefficient without the specified coefficients

    Parameters
    ----------
    index :
        indices that are removed in the returned object
    inplace :
        whether the values will be removed inplace

    Returns
    -------
    Coefficient
        a new coefficient without containing the requested elements
    """
    index = np.delete(np.arange(len(coefficient)), coefficient._sanitize_index(index))
    return coefficient.sub(index, inplace)


@register_sisl_dispatch(State, module="sisl.physics")
def sub(state: State, index: SimpleIndex, inplace: bool = False) -> Optional[State]:
    """Return a new state with only the specified states

    Parameters
    ----------
    index :
        indices that are retained in the returned object
    inplace :
        whether the values will be retained inplace

    Returns
    -------
    State
       a new state only containing the requested elements, only if `inplace` is false
    """
    index = state._sanitize_index(index)
    if inplace:
        state.state = state.state[index]
    else:
        out = state.__class__(state.state[index], state.parent)
        out.info = state.info
        return out


@register_sisl_dispatch(State, module="sisl.physics")
def remove(state: State, index: SimpleIndex, inplace: bool = False) -> Optional[State]:
    """Return a new state without the specified vectors

    Parameters
    ----------
    index :
        indices that are removed in the returned object
    inplace :
        whether the values will be removed inplace

    Returns
    -------
    State
        a new state without containing the requested elements, only if `inplace` is false
    """
    index = np.delete(np.arange(len(state)), state._sanitize_index(index))
    return state.sub(index, inplace)


@register_sisl_dispatch(StateC, module="sisl.physics")
def sub(statec: StateC, index: SimpleIndex, inplace: bool = False) -> Optional[StateC]:
    """Return a new state with only the specified states

    Parameters
    ----------
    index :
        indices that are retained in the returned object
    inplace :
        whether the values will be retained inplace

    Returns
    -------
    StateC
        a new object with a subset of the states, only if `inplace` is false
    """
    index = statec._sanitize_index(index).ravel()
    if inplace:
        statec.state = statec.state[index]
        statec.c = statec.c[index]
    else:
        out = statec.__class__(statec.state[index, ...], statec.c[index], statec.parent)
        out.info = statec.info
        return out


@register_sisl_dispatch(StateC, module="sisl.physics")
def remove(
    statec: StateC, index: SimpleIndex, inplace: bool = False
) -> Optional[StateC]:
    """Return a new state without the specified indices

    Parameters
    ----------
    index :
        indices that are removed in the returned object
    inplace :
        whether the values will be removed inplace

    Returns
    -------
    StateC
        a new state without containing the requested elements, only if `inplace` is false
    """
    index = np.delete(np.arange(len(statec)), statec._sanitize_index(index))
    return statec.sub(index, inplace)


@register_sisl_dispatch(State, module="sisl.physics")
def rotate(
    state: State, phi: float = 0.0, individual: bool = False, inplace: bool = False
) -> Optional[State]:
    r"""Rotate all states to rotate the largest component to be along the angle `phi`

    The states will be rotated according to:

    .. math::

        \mathbf S' = \mathbf S / \mathbf S^\dagger_{\phi-\mathrm{max}} \exp (i \phi),

    where :math:`\mathbf S^\dagger_{\phi-\mathrm{max}}` is the phase of the component with the largest amplitude
    and :math:`\phi` is the angle to align on.

    Parameters
    ----------
    phi : float, optional
       angle to align the state at (in radians), 0 is the positive real axis
    individual : bool, optional
       whether the rotation is per state, or a single maximum component is chosen.
    inplace :
        whether to do the rotation on the object it-self (True), or return a copy
        with the rotated states (False).
    """
    # Convert angle to complex phase
    phi = np.exp(1j * phi)
    if inplace:
        out = state
    else:
        out = state.copy()
    s = out.state.view()
    if individual:
        for i in range(len(s)):
            # Find the maximum amplitude index
            idx = np.argmax(np.absolute(s[i, :]))
            s[i, :] *= phi * np.conj(s[i, idx] / np.absolute(s[i, idx]))
    else:
        # Find the maximum amplitude index among all elements
        idx = np.unravel_index(np.argmax(np.absolute(s)), s.shape)
        s *= phi * np.conj(s[idx] / np.absolute(s[idx]))
    return out


@register_sisl_dispatch(State, module="sisl.physics")
def tile(
    state: State, reps: int, axis: int, normalize: bool = False, offset: float = 0
) -> State:
    r"""Tile the state vectors for a new supercell

    Tiling a state vector makes use of the Bloch factors for a state by utilizing

    .. math::

       \psi_{\mathbf k}(\mathbf r + \mathbf T) \propto e^{i\mathbf k\cdot \mathbf T}

    where :math:`\mathbf T = i\mathbf a_0 + j\mathbf a_1 + l\mathbf a_2`. Note that `axis`
    selects which of the :math:`\mathbf a_i` vectors that are translated and `reps` corresponds
    to the :math:`i`, :math:`j` and :math:`l` variables. The `offset` moves the individual states
    by said amount, i.e. :math:`i\to i+\mathrm{offset}`.

    Parameters
    ----------
    reps :
       number of repetitions along a specific lattice vector
    axis :
       lattice vector to tile along
    normalize:
       whether the states are normalized upon return, may be useful for
       eigenstates, equivalent to ``state.tile().normalize()``
    offset:
       the offset for the phase factors

    See Also
    --------
    Geometry.tile
    Grid.tile
    Lattice.tile
    """
    # the parent gets tiled
    parent = state.parent.tile(reps, axis)
    # the k-point gets reduced
    k = _a.asarrayd(state.info.get("k", [0, 0, 0]))

    # now tile the state vectors
    state_t = np.tile(state.state, (1, reps)).astype(np.complex128, copy=False)
    # re-shape to apply phase-factors
    state_t.shape = (len(state), reps, -1)

    # Tiling stuff is trivial since we simply
    # translate the bloch coefficients with:
    #   exp(i k.T)
    # with T being
    #   i * a_0 + j * a_1 + k * a_2
    # We can leave out the lattice vectors entirely
    phase = np.exp(2j * np.pi * k[axis] * (_a.aranged(reps) + offset))

    state_t *= phase.reshape(1, -1, 1)
    state_t.shape = (len(state), -1)

    # update new k; when we double the system, we halve the periodicity
    # and hence we need to account for this
    k[axis] = k[axis] * reps % 1
    while k[axis] > 0.5:
        k[axis] -= 1
    while k[axis] <= -0.5:
        k[axis] += 1

    # this allows us to make the same usable for StateC classes
    out = state.copy()
    out.parent = parent
    out.state = state_t
    # update the k-point
    out.info = dict(**state.info)
    out.info.update({"k": k})

    if normalize:
        return out.normalize()
    return out
