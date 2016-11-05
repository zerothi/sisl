from __future__ import print_function, division

import sys
from numbers import Integral, Real, Complex
import collections

import numpy as np

__all__ = ['array_fill_repeat', '_str', 'isiterable', 'ensure_array']
__all__ += ['get_dtype']


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


def get_dtype(var, int=None, other=None):
    """ Returns the `numpy.dtype` equivalent of `var`.

    Parameters
    ----------
    var : object
       the variable that will be tried to be cast into
       a `numpy.dtype`.
    int : `numpy.dtype` of `np.int*`
       whether an integer would be allowed to be cast to 
       the int64 equivalent.
       Because default integers in Python are of infinite
       precision, but `numpy` is limited to long, `numpy` will
       always select `np.int64` when an integer is tried
       to be converted.
       This will prohibit this conversion and will revert to int32.
    other : `numpy.dtype`
       If supplied the returned value will be extracted from:
       >>> numpy.result_type(dtype(var)(1), other(1))
       such that one can select the highest among `var` and
       the input `other`.
       For instance:
       >>> get_dtype(1., other=numpy.complex128) == np.complex128
       >>> get_dtype(1., other=numpy.int32) == np.float64
       >>> get_dtype(1, other=numpy.int32) == np.int32
    """
    if int is None:
        int = np.int32

    dtype = np.result_type(var)
    if dtype == np.int64:
        dtype = int
    try:
        dtype(1)
    except:
        dtype = dtype.type

    if other is not None:
        try:
            other(1)
        except:
            other = other.type
        return np.result_type(dtype(1), other(1))
    
    return dtype
