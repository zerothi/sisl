# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from functools import partial as _partial

import numpy as np
from numpy import (
    asarray,
    complex64,
    complex128,
    cumsum,
    float32,
    float64,
    int32,
    int64,
    ones,
    take,
    zeros,
)

__all__ = ["broadcast_shapes"]


def _append(name, suffix="ilfd"):
    return [name + s for s in suffix]


def broadcast_shapes(*shapes):
    """Calculate the broad-casted shape of a list of shapes

    This should be replaced by np.broadcast_shapes when 1.20 is the default.
    """
    # create all arrays using 0 memory
    arrays = [np.empty(shape, dtype=[]) for shape in shapes]
    return np.broadcast(*arrays).shape


def array_arange(start, end=None, n=None, dtype=int64):
    """Creates a single array from a sequence of `numpy.arange`

    Parameters
    ----------
    start : array_like
       a list of start elements for `numpy.arange`
    end : array_like
       a list of end elements (exclusive) for `numpy.arange`.
       This argument is not used if `n` is passed.
    n : array_like
       a list of counts of elements for `numpy.arange`.
       This is equivalent to ``end=start + n``.
    dtype : numpy.dtype
       the returned lists data-type

    Examples
    --------
    >>> array_arange([1, 5], [3, 6])
    array([1, 2, 5], dtype=int64)
    >>> array_arange([1, 6], [4, 9])
    array([1, 2, 3, 6, 7, 8], dtype=int64)
    >>> array_arange([1, 6], n=[2, 2])
    array([1, 2, 6, 7], dtype=int64)
    """
    # Tests show that the below code is faster than
    # implicit for-loops, or list-comprehensions
    # concatenate(map(..)
    # The below is much faster and does not require _any_ loops
    if n is None:
        # We need n to speed things up
        n = asarray(end) - asarray(start)
    else:
        n = asarray(n)
    # The below algorithm only works for non-zero n
    idx = n.nonzero()[0]

    # Grab corner case
    if len(idx) == 0:
        return zeros(0, dtype=dtype)

    # Reduce size
    start = take(start, idx)
    n = take(n, idx)

    # Create array of 1's.
    # The 1's are important when issuing the cumultative sum
    a = ones(n.sum(), dtype=dtype)

    # set pointers such that we can
    # correct for final cumsum
    ptr = cumsum(n[:-1])
    a[0] = start[0]
    # Define start and correct for previous values
    a[ptr] = start[1:] - start[:-1] - n[:-1] + 1

    return cumsum(a, dtype=dtype)


__all__ += ["array_arange"]

# Create all partial objects for creating arrays
array_arangei = _partial(array_arange, dtype=int32)
array_arangel = _partial(array_arange, dtype=int64)
__all__ += _append("array_arange", "il")

zeros = np.zeros
zerosi = _partial(zeros, dtype=int32)
zerosl = _partial(zeros, dtype=int64)
zerosf = _partial(zeros, dtype=float32)
zerosd = _partial(zeros, dtype=float64)
zerosc = _partial(zeros, dtype=complex64)
zerosz = _partial(zeros, dtype=complex128)
__all__ += _append("zeros", "ilfdcz")

ones = np.ones
onesi = _partial(ones, dtype=int32)
onesl = _partial(ones, dtype=int64)
onesf = _partial(ones, dtype=float32)
onesd = _partial(ones, dtype=float64)
onesc = _partial(ones, dtype=complex64)
onesz = _partial(ones, dtype=complex128)
__all__ += _append("ones", "ilfdcz")

empty = np.empty
emptyi = _partial(empty, dtype=int32)
emptyl = _partial(empty, dtype=int64)
emptyf = _partial(empty, dtype=float32)
emptyd = _partial(empty, dtype=float64)
emptyc = _partial(empty, dtype=complex64)
emptyz = _partial(empty, dtype=complex128)
__all__ += _append("empty", "ilfdcz")

array = np.array
arrayi = _partial(array, dtype=int32)
arrayl = _partial(array, dtype=int64)
arrayf = _partial(array, dtype=float32)
arrayd = _partial(array, dtype=float64)
arrayc = _partial(array, dtype=complex64)
arrayz = _partial(array, dtype=complex128)
__all__ += _append("array", "ilfdcz")

asarray = np.asarray
asarrayi = _partial(asarray, dtype=int32)
asarrayl = _partial(asarray, dtype=int64)
asarrayf = _partial(asarray, dtype=float32)
asarrayd = _partial(asarray, dtype=float64)
asarrayc = _partial(asarray, dtype=complex64)
asarrayz = _partial(asarray, dtype=complex128)
__all__ += _append("asarray", "ilfdcz") + ["asarray"]

fromiter = np.fromiter
fromiteri = _partial(fromiter, dtype=int32)
fromiterl = _partial(fromiter, dtype=int64)
fromiterf = _partial(fromiter, dtype=float32)
fromiterd = _partial(fromiter, dtype=float64)
fromiterc = _partial(fromiter, dtype=complex64)
fromiterz = _partial(fromiter, dtype=complex128)
__all__ += _append("fromiter", "ilfdcz")

sumi = _partial(np.sum, dtype=int32)
suml = _partial(np.sum, dtype=int64)
sumf = _partial(np.sum, dtype=float32)
sumd = _partial(np.sum, dtype=float64)
sumc = _partial(np.sum, dtype=complex64)
sumz = _partial(np.sum, dtype=complex128)
__all__ += _append("sum", "ilfdcz")

cumsum = np.cumsum
cumsumi = _partial(cumsum, dtype=int32)
cumsuml = _partial(cumsum, dtype=int64)
cumsumf = _partial(cumsum, dtype=float32)
cumsumd = _partial(cumsum, dtype=float64)
cumsumc = _partial(cumsum, dtype=complex64)
cumsumz = _partial(cumsum, dtype=complex128)
__all__ += _append("cumsum", "ilfdcz")

arange = np.arange
arangei = _partial(arange, dtype=int32)
arangel = _partial(arange, dtype=int64)
arangef = _partial(arange, dtype=float32)
aranged = _partial(arange, dtype=float64)
arangec = _partial(arange, dtype=complex64)
arangez = _partial(arange, dtype=complex128)
__all__ += _append("arange", "ilfdcz")

prod = np.prod
prodi = _partial(prod, dtype=int32)
prodl = _partial(prod, dtype=int64)
prodf = _partial(prod, dtype=float32)
prodd = _partial(prod, dtype=float64)
prodc = _partial(prod, dtype=complex64)
prodz = _partial(prod, dtype=complex128)
__all__ += _append("prod", "ilfdcz")

# Create all partial objects for creating arrays
full = np.full
fulli = _partial(full, dtype=int32)
fulll = _partial(full, dtype=int64)
fullf = _partial(full, dtype=float32)
fulld = _partial(full, dtype=float64)
fullc = _partial(full, dtype=complex64)
fullz = _partial(full, dtype=complex128)
__all__ += _append("full", "ilfdcz")

linspace = np.linspace
linspacef = _partial(linspace, dtype=float32)
linspaced = _partial(linspace, dtype=float64)
linspacec = _partial(linspace, dtype=complex64)
linspacez = _partial(linspace, dtype=complex128)
__all__ += _append("linspace", "fdcz")

del _append, _partial
