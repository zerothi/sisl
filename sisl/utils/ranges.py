"""
Basic functionality of creating ranges from text-input and/or other types of information
"""
from __future__ import print_function, division

__all__ = ['str2range', 'fileindex']

import numpy as np

import re
# This reg-exp matches:
#   0, 1, 3, 3-9, etc.
re_ints = re.compile('[,]?([0-9-]+)[,]?')
# This reg-exp matches:
#   0, 1[0-1], 3, 3-9, etc.
re_sub  = re.compile('[,]?([0-9-]+\[[0-9,-]+\]|[0-9-]+)[,]?')

# Function to change a string to a range of atoms
def str2range(s, sub=False):
    """ Parses a string into a list of ranges 
    
    This range can be formatted like this:
      1,2,3-6
    in which case it will return:
      [1,2,3,4,5,6]
    """
    if sub:
        # We parse according to
        #   re_sub
        rng = []
        for la in re_sub.findall(s):
            # We accumulate a list of ranges with sub-ranges

            # First we split in []
            if '[' in la:
                # There are sub-ranges 
                t = la.split('[')
                # get the top-ranges
                cur = str2range(t[0])
                # get the sub-ranges
                sub = str2range(t[1].split(']')[0])
            else:
                cur = str2range(la)
                sub = str2range("")

            # Now append to the range
            for i in cur:
                rng.append([i, sub])

        return rng

    # This is regular ranges
    rng = []
    for la in re_ints.findall(s):
        # We accumulate a list of integers
        tmp = la.split('-')
        if len(tmp) > 2: 
            raise ValueError("Could not parse: '"+s+"' successfully.")
        elif len(tmp) == 2:
            bi, ei = tuple( map(int,tmp) )
            rng.extend( range(bi, ei+1) )
        else:
            rng.append( int(tmp[0]) )
    return np.asarray(rng)


# Function to retrieve an optional index from the
# filename
#   file[0] returns:
#     file, 0
#   file returns:
#     file, None
#   file[0-1] returns
#     file, [0,1]
def fileindex(f):
    """ Parses a filename string into the filename and the indices.
    
    This range can be formatted like this:
      file[1,2,3-6]
    in which case it will return:
      file, [1,2,3,4,5,6]
    """

    if not '[' in f:
        return f, None

    f = f.split('[')
    fname = f.pop(0)
    f = ''.join(f.split(']'))
    rng = str2range(f)
    if len(rng) == 1:
        return fname, rng[0]
    return fname, rng
