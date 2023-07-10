# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from re import compile as re_compile

import numpy as np

__all__ = ["starts_with_list", "header_to_dict",
           "grid_reduce_indices"]


def starts_with_list(l, comments):
    for comment in comments:
        if l.strip().startswith(comment):
            return True
    return False


def header_to_dict(header):
    """ Convert a header line with 'key=val key1=val1' sequences to a single dictionary """
    e = re_compile(r"(\S+)=")

    # 1. Remove *any* entry with 0 length
    # 2. Ensure it is a list
    # 3. Reverse the list order (for popping)
    kv = list(filter(lambda x: len(x.strip()) > 0, e.split(header)))[::-1]

    # Now create the dictionary
    d = {}
    while len(kv) >= 2:
        # We have reversed the list
        key = kv.pop().strip(' =') # remove white-space *and* =
        val = kv.pop().strip() # remove outer whitespace
        d[key] = val

    return d


def grid_reduce_indices(grids, factors, axis=0, out=None):
    """ Reduce `grids` into a single `grid` value along `axis` by summing the `factors`

    If `out` is defined the data will be stored there.
    """
    if len(factors) > grids.shape[axis]:
        raise ValueError(f"Trying to reduce a grid with too many factors: {len(factors)} > {grids.shape[axis]}")

    if out is not None:
        grid = out
        np.take(grids, 0, axis=axis, out=grid)
        grid *= factors[0]
    else:
        grid = np.take(grids, 0, axis=axis) * factors[0]

    for idx, factor in enumerate(factors[1:], start=1):
        grid += np.take(grids, idx, axis=axis) * factor

    return grid
