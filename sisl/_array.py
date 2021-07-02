# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from functools import partial as _partial

import numpy as np
from numpy import zeros, ones, cumsum, take, int32, int64
from numpy import asarray


__all__ = []


def _append(name, suffix='ilfd'):
    return [name + s for s in suffix]


def array_arange(start, end=None, n=None, dtype=int64):
    """ Creates a single array from a sequence of `numpy.arange`

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


__all__ += ['array_arange']

# Create all partial objects for creating arrays
array_arangei = _partial(array_arange, dtype=np.int32)
array_arangel = _partial(array_arange, dtype=np.int64)
__all__ += _append('array_arange', 'il')

zerosi = _partial(np.zeros, dtype=np.int32)
zerosl = _partial(np.zeros, dtype=np.int64)
zerosf = _partial(np.zeros, dtype=np.float32)
zerosd = _partial(np.zeros, dtype=np.float64)
zerosc = _partial(np.zeros, dtype=np.complex64)
zerosz = _partial(np.zeros, dtype=np.complex128)
__all__ += _append('zeros', 'ilfdcz')

onesi = _partial(np.ones, dtype=np.int32)
onesl = _partial(np.ones, dtype=np.int64)
onesf = _partial(np.ones, dtype=np.float32)
onesd = _partial(np.ones, dtype=np.float64)
onesc = _partial(np.ones, dtype=np.complex64)
onesz = _partial(np.ones, dtype=np.complex128)
__all__ += _append('ones', 'ilfdcz')

emptyi = _partial(np.empty, dtype=np.int32)
emptyl = _partial(np.empty, dtype=np.int64)
emptyf = _partial(np.empty, dtype=np.float32)
emptyd = _partial(np.empty, dtype=np.float64)
emptyc = _partial(np.empty, dtype=np.complex64)
emptyz = _partial(np.empty, dtype=np.complex128)
__all__ += _append('empty', 'ilfdcz')

arrayi = _partial(np.array, dtype=np.int32)
arrayl = _partial(np.array, dtype=np.int64)
arrayf = _partial(np.array, dtype=np.float32)
arrayd = _partial(np.array, dtype=np.float64)
arrayc = _partial(np.array, dtype=np.complex64)
arrayz = _partial(np.array, dtype=np.complex128)
__all__ += _append('array', 'ilfdcz')

asarray = np.asarray
asarrayi = _partial(np.asarray, dtype=np.int32)
asarrayl = _partial(np.asarray, dtype=np.int64)
asarrayf = _partial(np.asarray, dtype=np.float32)
asarrayd = _partial(np.asarray, dtype=np.float64)
asarrayc = _partial(np.asarray, dtype=np.complex64)
asarrayz = _partial(np.asarray, dtype=np.complex128)
__all__ += _append('asarray', 'ilfdcz') + ['asarray']

fromiteri = _partial(np.fromiter, dtype=np.int32)
fromiterl = _partial(np.fromiter, dtype=np.int64)
fromiterf = _partial(np.fromiter, dtype=np.float32)
fromiterd = _partial(np.fromiter, dtype=np.float64)
fromiterc = _partial(np.fromiter, dtype=np.complex64)
fromiterz = _partial(np.fromiter, dtype=np.complex128)
__all__ += _append('fromiter', 'ilfdcz')

sumi = _partial(np.sum, dtype=np.int32)
suml = _partial(np.sum, dtype=np.int64)
sumf = _partial(np.sum, dtype=np.float32)
sumd = _partial(np.sum, dtype=np.float64)
sumc = _partial(np.sum, dtype=np.complex64)
sumz = _partial(np.sum, dtype=np.complex128)
__all__ += _append('sum', 'ilfdcz')

cumsumi = _partial(np.cumsum, dtype=np.int32)
cumsuml = _partial(np.cumsum, dtype=np.int64)
cumsumf = _partial(np.cumsum, dtype=np.float32)
cumsumd = _partial(np.cumsum, dtype=np.float64)
cumsumc = _partial(np.cumsum, dtype=np.complex64)
cumsumz = _partial(np.cumsum, dtype=np.complex128)
__all__ += _append('cumsum', 'ilfdcz')

arangei = _partial(np.arange, dtype=np.int32)
arangel = _partial(np.arange, dtype=np.int64)
arangef = _partial(np.arange, dtype=np.float32)
aranged = _partial(np.arange, dtype=np.float64)
arangec = _partial(np.arange, dtype=np.complex64)
arangez = _partial(np.arange, dtype=np.complex128)
__all__ += _append('arange', 'ilfdcz')

prodi = _partial(np.prod, dtype=np.int32)
prodl = _partial(np.prod, dtype=np.int64)
prodf = _partial(np.prod, dtype=np.float32)
prodd = _partial(np.prod, dtype=np.float64)
prodc = _partial(np.prod, dtype=np.complex64)
prodz = _partial(np.prod, dtype=np.complex128)
__all__ += _append('prod', 'ilfdcz')

# Create all partial objects for creating arrays
fulli = _partial(np.full, dtype=np.int32)
fulll = _partial(np.full, dtype=np.int64)
fullf = _partial(np.full, dtype=np.float32)
fulld = _partial(np.full, dtype=np.float64)
fullc = _partial(np.full, dtype=np.complex64)
fullz = _partial(np.full, dtype=np.complex128)
__all__ += _append('full', 'ilfdcz')

linspacef = _partial(np.linspace, dtype=np.float32)
linspaced = _partial(np.linspace, dtype=np.float64)
linspacec = _partial(np.linspace, dtype=np.complex64)
linspacez = _partial(np.linspace, dtype=np.complex128)
__all__ += _append('linspace', 'fdcz')

del _append, _partial
