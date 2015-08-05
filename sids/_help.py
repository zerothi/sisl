from __future__ import print_function, division

import numpy as np

__all__ = ['array_fill_repeat']


def array_fill_repeat(array,size,cls=None):
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
        raise ValueError('Repetition of or array is not divisible with actual length. ' +
                         'Hence we cannot create a repeated size.')
    if cls is None:
        if reps > 1:
            return np.tile(array,reps)
        return array
    else:
        if reps > 1:
            return np.tile(np.array(array,dtype=cls),reps)
        return np.array(array,dtype=cls)
