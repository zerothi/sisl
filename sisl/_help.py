from __future__ import print_function, division

import sys
import collections

import numpy as np

__all__ = ['array_fill_repeat']
__all__ += ['isndarray', 'isiterable']
__all__ += ['get_dtype']
__all__ += ['dtype_complex_to_real', 'dtype_real_to_complex']

# Wrappers typically used
__all__ += ['_str', '_range', '_zip', '_map']
__all__ += ['is_python2', 'is_python3']
__all__ += ['xml_parse']


# Base-class for string object checks
is_python3 = sys.version_info >= (3, 0)
is_python2 = not is_python3
if is_python3:
    _str = str
    _range = range
    _zip = zip
    _map = map
else:
    _str = basestring
    _range = xrange
    from itertools import izip as _zip
    from itertools import imap as _map


# Load the correct xml-parser
try:
    from defusedxml.ElementTree import parse as xml_parse
    if sys.version_info > (3, 6):
        from defusedxml import __version__ as defusedxml_version
        try:
            defusedxml_version = list(map(int, defusedxml_version.split('.')))
            if defusedxml_version[0] == 0 and defusedxml_version[1] <= 5:
                raise ImportError
        except:
            raise ImportError
except ImportError:
    from xml.etree.ElementTree import parse as xml_parse


def array_fill_repeat(array, size, cls=None):
    """
    This repeats an array along the zeroth axis until it
    has size `size`. Note that initial size of `array` has
    to be an integer part of `size`.
    """
    try:
        reps = size // len(array)
    except:
        array = [array]
        reps = size // len(array)
    if size % len(array) != 0:
        # We do not have it correctly formatted (either an integer
        # repeatable part, full, or a single)
        raise ValueError('Repetition of or array is not divisible with actual length. '
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


_ndarray = np.ndarray


def isndarray(arr):
    """ Returns ``True`` if the input object is a `numpy.ndarray` object """
    return isinstance(arr, _ndarray)


def get_dtype(var, int=None, other=None):
    """ Returns the `numpy.dtype` equivalent of `var`.

    Parameters
    ----------
    var : object
       the variable that will be tried to be cast into
       a `numpy.dtype`.
    int : numpy.dtype of np.int*
       whether an integer would be allowed to be cast to
       the int64 equivalent.
       Because default integers in Python are of infinite
       precision, but `numpy` is limited to long, `numpy` will
       always select `np.int64` when an integer is tried
       to be converted.
       This will prohibit this conversion and will revert to int32.
    other : numpy.dtype
       If supplied the returned value will be extracted from:

       >>> np.result_type(dtype(var)(1), other(1)) # doctest: +SKIP

       such that one can select the highest among `var` and
       the input `other`.
       For instance:

    Examples
    --------
    >>> get_dtype(1., other=np.complex128) == np.complex128
    True
    >>> get_dtype(1., other=np.int32) == np.float64
    True
    >>> get_dtype(1, other=np.int32) == np.int32
    True
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


def dtype_complex_to_real(dtype):
    """ Return the equivalent precision real data-type if the `dtype` is complex """
    if dtype == np.complex128:
        return np.float64
    elif dtype == np.complex64:
        return np.float32
    return dtype


def dtype_real_to_complex(dtype):
    """ Return the equivalent precision complex data-type if the `dtype` is real """
    if dtype == np.float64:
        return np.complex128
    elif dtype == np.float32:
        return np.complex64
    return dtype
