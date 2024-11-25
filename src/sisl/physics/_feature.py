# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

__all__ = ["yield_manifolds"]


def yield_manifolds(values, atol: float = 0.1, axis: int = -1) -> Iterator[list]:
    r"""Yields indices for manifolds along the axis `axis`

    A manifold is found under the criteria that all neighboring
    values along `axis` are separated by at least `atol`.

    Parameters
    ----------
    values : array_like
       the values to be separated into manifolds, all other axis than `axis` will
       be considered a single dimension when determining the manifolds
    atol : float, optional
       the tolerance used for assigning an index with a manifold.
    axis : int, optional
       which axis we use for extracting the manifold, all other axes will be reduced

    Examples
    --------
    >>> H = Hamiltonian(...)
    >>> mp = MonkhorstPack(H, [2, 2, 2])
    >>> eigs = mp.apply.ndarray.eigh()
    >>> for manifold in yield_manifolds(eigs):
    ...    print(manifold, eigs[:, manifold])
    """
    values = np.asarray(values)

    # Figure out min/max
    axes = list(range(values.ndim))
    del axes[axis]
    v_min = values.min(tuple(axes))
    v_max = values.max(tuple(axes))

    # Now calculate the manifold for each of the different directions
    manifold = [0]
    for i in range(1, len(v_min)):
        if np.all(v_max[i - 1] < v_min[i] - atol):
            # we are starting a new manifold
            yield manifold
            manifold = [i]
        else:
            manifold.append(i)
    yield manifold
