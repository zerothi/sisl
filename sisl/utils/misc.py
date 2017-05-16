"""
Miscellaneous routines
"""
from __future__ import division

from numbers import Integral
from math import pi

from sisl._help import _range as range

__all__ = ['merge_instances', 'str_spec', 'direction', 'angle']
__all__ += ['iter_shape']


def merge_instances(*args, **kwargs):
    """ Merges an arbitrary number of instances together. 

    Parameters
    ----------
    name: str or MergedClass
       name of class to merge
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


def iter_shape(shape):
    """ Generator for iterating a shape by returning consecutive slices 

    Parameters
    ----------
    shape : array_like
      the shape of the iterator

    Yields
    ------
    tuple of int
       a tuple of the same length as the input shape. The iterator
       is using the C-indexing.

    Examples
    --------
    >>> for slc in iter_shape([2, 1, 3]):
    >>>    print(slc)
    [0, 0, 0]
    [0, 0, 1]
    [0, 0, 2]
    [1, 0, 0]
    [1, 0, 1]
    [1, 0, 2]
    """
    shape1 = [i-1 for i in shape]
    ns = len(shape)
    ns1 = ns - 1
    # Create list for iterating
    # we require a list because tuple's are immutable
    slc = [0] * ns

    while slc[0] < shape[0]:
        for i in range(shape[ns1]):
            slc[ns1] = i
            yield slc

        # Increment the previous shape indices
        for i in range(ns1, 0, -1):
            if slc[i] >= shape1[i]:
                slc[i] = 0
                if i > 0:
                    slc[i-1] += 1


def str_spec(name):
    """ Split into a tuple of name and specifier, delimited by ``{...}``.

    Parameters
    ----------
    name: str
       string to split

    Returns
    -------
    tuple of str
       returns the name and the specifier (without delimiter) in a tuple

    Examples
    --------
    >>> str_spec('hello')
    'hello', None
    >>> str_spec('hello{TEST}')
    'hello', 'TEST'
    """
    if not name.endswith('}'):
        return name, None

    lname = name[:-1].split('{')
    return '{'.join(lname[:-1]), lname[-1]


# Transform a string to a Cartesian direction
def direction(d):
    """ Return the index coordinate index corresponding to the Cartesian coordinate system.

    Parameters
    ----------
    d: {0, 'X', 'x', 1, 'Y', 'y',  2, 'Z', 'z'}
       returns the integer that corresponds to the coordinate index.
       If it is an integer, it is returned *as is*.

    Returns
    -------
    int
       The index of the Cartesian coordinate system.

    Examples
    --------
    >>> direction(0)
    0
    >>> direction('Y')
    1
    >>> direction('z')
    2
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
    """ Convert the input string to an angle, either radians or degrees.

    Parameters
    ----------
    s : str
       If `s` starts with ``'r'`` it is interpreted as radians ``[0:2pi]``.
       If `s` starts with ``'a'`` it is interpreted as a regular angle ``[0:360]``.
       If `s` ends with ``'r'`` it returns in radians.
       If `s` ends with ``'a'`` it returns in regular angle.

       `s` may be any mathematical equation which can be 
       intercepted through `eval`.
    radians : bool
       Whether the returned angle is in radians. 
       Note than an ``'r'`` at the end of `s` has precedence.
    in_radians : bool
       Whether the calculated angle is in radians. 
       Note than an `'r'` at the beginning of `s` has precedence.

    Returns
    -------
    float
       the angle in the requested unit
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
