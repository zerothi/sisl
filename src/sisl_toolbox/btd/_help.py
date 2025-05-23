# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np


def dagger(M: np.ndarray) -> np.ndarray:
    return np.conj(M.T)


def get_expand(H_len: int, matrix_len: int) -> int:
    """Calculate the expansion coefficient

    This neglects the spin properties.
    """
    return H_len // matrix_len


def expand_btd(btd: np.ndarray, expand: int) -> np.ndarray:
    """Convert a BTD segment array to have `expand` bigger chunks

    Examples
    --------

    >>> a = [20, 10, 30]
    >>> _expand_btd(a, 2)
    [40, 20, 60]
    """
    if expand > 1:
        return btd * expand

    return btd


def expand_orbs(orbs: np.ndarray, expand: int) -> np.ndarray:
    """Convert an orbital index array to be `*expand` longer

    Examples
    --------

    >>> a = [1, 0, 2]
    >>> _expand_orbs(a, 2)
    [2, 3, 0, 1, 4, 5]
    """
    if expand > 1:
        orbs = np.repeat(orbs * expand, expand)
        for i in range(1, expand):
            orbs[i::expand] += i

    return orbs
