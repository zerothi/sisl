from __future__ import print_function, division

import numpy as np
import sys
from numbers import Integral

__all__ = ['array_fill_repeat', '_str', 'ensure_array']


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


def ensure_array(arr, dtype=np.int32):
    if isinstance(arr, Integral):
        return np.array([arr], dtype)
    return arr
