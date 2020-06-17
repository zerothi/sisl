import sys
import functools
import warnings

import numpy as np

from ._internal import set_module

__all__ = ["array_fill_repeat"]
__all__ += ["isndarray", "isiterable", "voigt_matrix"]
__all__ += ["get_dtype"]
__all__ += ["dtype_complex_to_real", "dtype_real_to_complex"]
__all__ += ["wrap_filterwarnings"]

# Wrappers typically used
__all__ += ["xml_parse"]


# Base-class for string object checks
import collections.abc as collections_abc

# Load the correct xml-parser
try:
    from defusedxml.ElementTree import parse as xml_parse
    from defusedxml import __version__ as defusedxml_version
    try:
        defusedxml_version = list(map(int, defusedxml_version.split(".")))
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
        raise ValueError("Repetition of or array is not divisible with actual length. "
                         "Hence we cannot create a repeated size.")
    if cls is None:
        if reps > 1:
            return np.tile(array, reps)
        return array
    else:
        if reps > 1:
            return np.tile(np.array(array, dtype=cls), reps)
        return np.array(array, dtype=cls)


@set_module("sisl")
def voigt_matrix(M, to_voigt):
    r""" Convert a matrix from Voigt representation to dense, or from matrix to Voigt

    Parameters
    ----------
    M : array_like
       matrix with last dimension having the Voigt representation or the last 2 dimensions
       having the symmetric matrix
    to_voigt : logical
       if True, the input matrix is assumed *not* to be in Voigt representation and will
       be returned in Voigt notation. Otherwise, the opposite will happen
    """
    if to_voigt:
        m = np.empty(M.shape[:-2] + (6,), dtype=M.dtype)
        m[..., 0] = M[..., 0, 0] # xx
        m[..., 1] = M[..., 1, 1] # yy
        m[..., 2] = M[..., 2, 2] # zz
        m[..., 3] = (M[..., 2, 1] + M[..., 1, 2]) * 0.5 # zy
        m[..., 4] = (M[..., 2, 0] + M[..., 0, 2]) * 0.5 # zx
        m[..., 5] = (M[..., 1, 0] + M[..., 0, 1]) * 0.5 # xy
    else:
        m = np.empty(M.shape[:-1] + (3, 3), dtype=M.dtype)
        m[..., 0, 0] = M[..., 0] # xx
        m[..., 1, 1] = M[..., 1] # yy
        m[..., 2, 2] = M[..., 2] # zz
        m[..., 0, 1] = M[..., 5] # xy
        m[..., 1, 0] = M[..., 5] # xy
        m[..., 0, 2] = M[..., 4] # xz
        m[..., 2, 0] = M[..., 4] # xz
        m[..., 1, 2] = M[..., 3] # zy
        m[..., 2, 1] = M[..., 3] # zy
    return m

_Iterable = collections_abc.Iterable


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

       >>> np.result_type(dtype(var)(1), other(1))

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


def array_replace(array, *replace, **kwargs):
    """ Replace values in `array` using `replace`

    Replaces values in `array` using tuple/val in `replace`.

    Parameters
    ----------
    array : numpy.ndarray
       array in which to replace values from `replace`
    *replace : list of tuple arguments
       replacement values, interpreted as ``array[replace[0]] = replace[1]``.
    other : val
       value replaced in `array` for all indices not in ``replace[0]``

    Examples
    --------
    >>> ar = [1, 2, 3]
    >>> array_replace(ar, (1, 1), (2, 1), other=2)
    [2, 1, 1]
    >>> array_replace(ar, (1, 1), (2, 1))
    [1, 1, 1]
    >>> array_replace(ar, (1, 1), (0, 3))
    [3, 1, 3]
    """
    ar = array.copy()
    others = list()

    for idx, val in replace:
        if not val is None:
            ar[idx] = val
        others.append(np.asarray(idx).ravel())

    if "other" in kwargs:
        others = np.delete(np.arange(ar.size), np.unique(np.concatenate(others)))
        ar[others] = kwargs["other"]

    return ar


def wrap_filterwarnings(*args, **kwargs):
    """ Instead of creating nested `with` statements one can wrap entire functions with a filter

    The following two are equivalent:

    >>> def func():
    ...    with warnings.filterwarnings(*args, **kwargs):
    ...        ...

    >>> @wrap_filterwarnings(*args, **kwargs)
    >>> def func():
    ...    ...

    Parameters
    ----------
    *args :
       arguments passed to `warnings.filterwarnings`
    **kwargs :
       keyword arguments passed to `warnings.filterwarnings`
    """
    def decorator(func):
        @functools.wraps(func)
        def wrap_func(*func_args, **func_kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(*args, **kwargs)
                return func(*func_args, **func_kwargs)
        return wrap_func
    return decorator
