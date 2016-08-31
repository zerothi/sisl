"""
Miscellaneous routines
"""
from __future__ import division

from numbers import Integral
from math import pi

__all__ = ['merge_instances', 'name_spec', 'direction', 'angle']


def merge_instances(*args, **kwargs):
    """ Merges an arbitrary number of instances together. 

    Parameters
    ----------
    name: str, "MergedClass"
       name of class
    """
    name = kwargs.get('name', 'MergedClass')
    # We must make a new-type class
    cls = type(name, (object,), {})
    # Create holder of class
    # We could have 
    m = cls()
    for arg in args:
        m.__dict__.update(arg.__dict__)
    return m


def name_spec(name):
    """ Checks whether `s` ends with `{..}`. Returns the split instances.
    
    Parameters
    ----------
    name: str
       string to split into proper `name` and specification

    Examples
    --------
    >>> name_spec('hello')
    'hello', None
    >>> name_spec('hello{TEST}')
    'hello', 'TEST'
    """
    if not name.endswith('}'):
        return name, None

    lname = name[:-1].split('{')
    return '{'.join(lname[:-1]), lname[-1] 


# Transform a string to a Cartesian direction
def direction(d):
    """ Return the index of the direction that the input represents

    Parameter
    ---------
    d: str, int
       If one of 'XYZ' or 'xyz' or 'ABC' or 'abc' it will return 012. If it is an integer, it is returned "as is"
    """
    if isinstance(d, Integral):
        return d

    # We take it as a string
    d = d.lower()
    # We must use an arry to not allow 'xy' input
    if d in 'x y z a b c'.split():
        return 'xaybzc'.index(d) // 2

    raise ValueError('Input direction is not an integer, nor a string in "xyzabc".')


# Transform an input to an angle
def angle(s, radians=True, in_radians=True):
    """ Convert the input string to an angle.

    Parameter
    ---------
    s: str
       If `s` starts with 'r' it is interpreted as radians [0:2pi].
       If `s` starts with 'a' it is interpreted as a regular angle [0:360].
       If `s` ends with 'r' it returns in radians.
       If `s` ends with 'a' it returns in regular angle.

       `s` may be any mathematical equation which can be 
       intercepted through `eval`.

    radians: bool, True
       Whether the returned angle is in radians. 
       Note than an 'r' at the end of `s` has precedence.
    in_radians: bool, True
       Whether the calculated angle is in radians. 
       Note than an 'r' at the beginning of `s` has precedence.
    """
    s = s.lower()
    
    if s.startswith('r'):
        in_radians = True
    elif s.startswith('a'):
        in_radians = False
    if s.endswith('r'):
        radians = True
    elif s.endswith('a'):
        radians = False

    # Remove all r/a's and remove white-space
    s = s.replace('r', '').replace('a', '').replace(' ', '')

    # Figure out if Pi is circumfered by */+-
    spi = s.split('pi')
    nspi = len(spi)
    if nspi > 1:
        # We have pi at least in one place.
        for i, si in enumerate(spi):
            # In case the last element is a pi
            if len(si) == 0:
                continue
            if i < nspi - 1:
                if not si.endswith(('*', '/', '+', '-')):
                    # it *MUST* be '*'
                    spi[i] = spi[i] + '*'
            if 0 < i:
                if not si.startswith(('*', '/', '+', '-')):
                    # it *MUST* be '*'
                    spi[i] = '*' + spi[i]

        # Now insert Pi dependent on the input type
        if in_radians:
            Pi = pi
        else:
            Pi = 180.
        
        s = ('{}'.format(Pi)).join(spi)

    # We have now transformed all values
    # to the correct numerical values and we calculate
    # the expression
    ra = eval(s)
    if radians and not in_radians:
        return ra / 180. * pi
    if not radians and in_radians:
        return ra / pi * 180.

    # Both radians and in_radians are equivalent
    # so return as-is
    return ra
