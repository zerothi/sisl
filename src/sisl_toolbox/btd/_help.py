# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def dagger(M: np.ndarray) -> np.ndarray:
    return np.conj(M.T)


def get_expand(H_len: int, matrix_len: int) -> int:
    """Calculate the expansion coefficient

    This neglects the spin properties.
    """
    return H_len // matrix_len


def expand_blocks(blocks: np.ndarray, expand: int) -> np.ndarray:
    """Convert a BTD segment array to have `expand` bigger chunks

    Examples
    --------

    >>> a = np.array([20, 10, 30])
    >>> expand_blocks(a, 2)
    array([40, 20, 60])
    """
    if expand > 1:
        return blocks * expand

    return blocks


def expand_orbs(orbs: np.ndarray, expand: int) -> np.ndarray:
    """Convert an orbital index array to be `*expand` longer

    Examples
    --------

    >>> a = np.array([1, 0, 2])
    >>> expand_orbs(a, 2)
    array([2, 3, 0, 1, 4, 5])
    """
    if expand > 1:
        orbs = np.repeat(orbs * expand, expand)
        for i in range(1, expand):
            orbs[i::expand] += i

    return orbs


def _split_space(idx, what: str) -> tuple[str, object]:
    """Split ``idx`` into its index space and value

    ``idx`` is a single entry ``{"atoms": value}`` or ``{"orbitals": value}``,
    anything else is atoms.
    """
    if isinstance(idx, Mapping):
        if len(idx) != 1 or not set(idx) <= {"atoms", "orbitals"}:
            raise ValueError(
                f"{what} must be a single entry dictionary with key "
                f"'atoms' or 'orbitals', got keys {list(idx)}"
            )
        ((space, idx),) = idx.items()
        return space, idx
    return "atoms", idx


def _check_atoms(H, atoms: np.ndarray, what: str) -> None:
    """Raise an error for atomic indices out of range (likely orbitals)"""
    if len(atoms) > 0 and (atoms.min() < 0 or atoms.max() >= H.na):
        raise ValueError(
            f"{what} are atoms and must be in [0, {H.na}), "
            "use {'orbitals': ...} for orbital indices."
        )


def _to_orbs(H, idx, what: str) -> np.ndarray:
    """Orbitals of `H` (without spin) from ``idx``

    ``idx`` are atoms, or a single entry ``{"atoms": idx}`` or ``{"orbitals": idx}``.
    """
    space, idx = _split_space(idx, what)
    idx = np.asarray(idx).ravel()
    if space == "atoms":
        _check_atoms(H, idx, what)
        return H.a2o(idx, all=True)
    return idx
