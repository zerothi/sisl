from __future__ import print_function, division

import sys
from numbers import Integral, Real, Complex
import collections

import numpy as np

__all__ = ['array_fill_repeat', '_str', 'isiterable', 'ensure_array']


# Base-class for string object checks
if sys.version_info >= (3, 0):
    _str = str
else:
    _str = basestring


def array_fill_repeat(array, size, cls=None):
    """
    This repeats an array along the zeroth axis until it
    has size `size`. Note that initial size of `array` has
    to be an integer part of `size`.
    """
    try:
        reps = len(array)
    except:
        array = [array]
    reps = size // len(array)
    if size % len(array) != 0:
        # We do not have it correctly formatted (either an integer
        # repeatable part, full, or a single)
        raise ValueError(
            'Repetition of or array is not divisible with actual length. ' +
            'Hence we cannot create a repeated size.')
    if cls is None:
        if reps > 1:
            return np.tile(array, reps)
        return array
    else:
        if reps > 1:
            return np.tile(np.array(array, dtype=cls), reps)
        return np.array(array, dtype=cls)

def isiterable(obj):
    """ Returns whether the object is an iterable or not """
    return isinstance(obj, collections.Iterable)

def ensure_array(arr, dtype=np.int32):
    """ Casts a number, list, tuple to a 1D array

    This will check which kind of argument `arr` is
    and will convert the value to the corresponding
    1D-array type.

    Parameters
    ----------
    arr : number/array_like/iterator
       if this is a number an array of `len() == 1` will
       be returned. Else, the array will be assured
       to be a ``numpy.ndarray``.
    dtype : ``numpy.dtype``
       the data-type of the array
    """
    if np.issubdtype(dtype, np.integer):
        comp = Integral
    elif np.issubdtype(dtype, np.float):
        comp = Real
    elif np.issubdtype(dtype, np.complex):
        comp = Complex
    if isinstance(arr, comp):
        return np.array([arr], dtype)
    elif isiterable(arr):
        return np.fromiter(arr, dtype)
    return np.asarray(arr, dtype)
