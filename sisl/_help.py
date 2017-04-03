from __future__ import print_function, division

import sys
from numbers import Integral, Real, Complex
import collections

import numpy as np

__all__ = ['array_fill_repeat', 'ensure_array', 'ensure_dtype']
__all__ += ['isndarray', 'isiterable']
__all__ += ['get_dtype']

# Wrappers typically used
__all__ += ['_str', '_range', '_zip']
__all__ += ['is_python2', 'is_python3']


# Base-class for string object checks
is_python3 = sys.version_info >= (3, 0)
is_python2 = not is_python3
if is_python3:
    _str = str
    _range = range
    _zip = zip
else:
    _str = basestring
    _range = xrange
    from itertools import izip as _zip


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

# To speed up isiterable
_Iterable = collections.Iterable


def isiterable(obj):
    """ Returns whether the object is an iterable or not """
    return isinstance(obj, _Iterable)


def isndarray(arr):
    """ Returns ``True`` if the input object is a ``numpy.ndarray`` object """
    return isinstance(arr, _ndarray)

# Private variables for speeding up ensure_array
_fromiter = np.fromiter
_ndarray = np.ndarray
_array = np.array
_asarray = np.asarray


def ensure_array(arr, dtype=np.int32, force=True):
    """ Casts a number, list, tuple to a 1D array

    This will check which kind of argument `arr` is
    and will convert the value to the corresponding
    1D-array type.

    Parameters
    ----------
    arr : number or array_like or iterator
       if this is a number an array of `len() == 1` will
       be returned. Else, the array will be assured
       to be a `numpy.ndarray`.
    dtype : `numpy.dtype`
       the data-type of the array
    force : bool
       if True the returned value will *always* be a `numpy.ndarray`, otherwise
       if a single number is passed it will return a numpy dtype variable.
    """
    # Complex is the highest common type
    # Real, Integer inherit from Complex
    # So basically this checks whether it is a single
    # number
    if isinstance(arr, _ndarray):
        return _asarray(arr, dtype)
    elif isinstance(arr, Complex):
        if not force:
            return dtype(arr)
        return _array([arr], dtype)
    elif isiterable(arr):
        # a numpy.ndarray is also iterable
        # hence we *MUST* check that before...
        return _fromiter(arr, dtype)
    return _asarray(arr, dtype)


def ensure_dtype(arr, dtype=np.int32):
    """ Wrapper for `ensure_array(..., force=False)` for returning numbers as well

    See Also
    --------
    ensure_array
    """
    return ensure_array(arr, dtype, force=False)


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

    # First try and see if the variable is a sub-class
    # of ndarray (or something numpy-like)
    try:
        dtype = var.dtype
    except:
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
