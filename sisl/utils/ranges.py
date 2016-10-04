"""
Basic functionality of creating ranges from text-input and/or other types of information
"""
from __future__ import print_function, division

__all__ = ['strmap', 'strseq', 'lstranges', 'erange', 'fileindex']

import numpy as np

import re
_eEfg = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
_re_eEfg = re.compile(_eEfg)
_re_segment = re.compile(r'(.+)\[(.+)\]|(.+)')
_re_irng = re.compile(r'\d+-\d+|\d+')

# This reg-exp matches:
#   0, 1, 3, 3-9, etc.
_re_ints = re.compile('[,]?([0-9-]+)[,]?')
# This reg-exp matches:
#   0, 1[0-1], 3, 3-9, etc.
_re_sub  = re.compile('[,]?([0-9-]+)\[([\[0-9,-\]]+)\]|([0-9-]+)[,]?')


# Function to change a string to a range of atoms
def strmap(func, s, recursive=True, sep=':'):
    """ Parse a string and map all entries using `func`.

    This parses a string and converts all `,`-separated to entries
    in the list.
    """

    # Create list
    l = []

    commas = s.split(',')
    i = 0
    while True:
        if i >= len(commas) - 1:
            break
        if commas[i].count('[') == commas[i].count(']'):
            i = i + 1
        else:
            # there must be more [ than ]
            commas[i] = commas[i] + "," + commas[i+1]
            del commas[i+1]
    i = len(commas) - 1
    if commas[i].count('[') != commas[i].count(']'):
        raise ValueError("Unbalanced string: not enough [ and ]")

    # Now we have the comma-separated list
    for seg in commas:
        # Split it in two parts
        m = _re_segment.findall(seg)[0]
        if len(m[2]) > 0:
            # the match is the last group
            l.append( strseq(func, m[2]) )
        elif recursive:
            l.append( (strseq(func, m[0]), strmap(func, m[1], sep=sep) ) )

    return l

def strseq(func, s):
    """ Accepts strings and returns tuples of content based on ranges.
    
    Parameters
    ----------
    func: function
       parser of the individual elements
    s: str
       string with content

    Examples
    --------
    >>> strseq(int, '3')
    3
    >>> strseq(int, '3-6')
    (3, 6)
    >>> strseq(int, '3:2:7')
    (3, 2, 7)
    >>> strseq(float, '3.2:6.3')
    (3.2, 6.3)
    """
    if ':' in s:
        return tuple(map(func, s.split(':')))
    elif '-' in s:
        return tuple(map(func, s.split('-')))
    return func(s)

def erange(*args):
    """ Returns the range with both ends includede """
    if len(args) == 3:
        return range(args[0], args[2]+1, args[1])
    return range(args[0], args[1]+1)

def lstranges(lst, func=erange):
    """ Convert a `strmap` list into expanded ranges """
    l = []
    # If an entry is a tuple, it means it is either
    # a range 0-1 == tuple(0, 1), or
    # a sub-range
    #   0[0-1], 0-1[0-1]
    if isinstance(lst, tuple):
        if len(lst) == 3:
            l.extend(func(*lst))
        else:
            head = lstranges(lst[0], func)
            bot = lstranges(lst[1], func)
            if isinstance(head, list):
                for el in head:
                    l.append([el, bot])
            elif isinstance(bot, list):
                l.append([head, bot])
            else:
                l.extend(func(head, bot))

    elif isinstance(lst, list):
        for lt in lst:
            ls = lstranges(lt, func)
            if isinstance(ls, list):
                l.extend(ls)
            else:
                l.append(ls)
    else:
        return lst
    return l


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
