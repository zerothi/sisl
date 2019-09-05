r""" Lists with inter-element operations

In certain cases it may be advantegeous to make operations on list elements rather than doing
list extensions.

This sub-module implements a list which allows to make operations with it-self or with scalars.
"""
from __future__ import print_function, division

from functools import wraps

from ._help import isiterable


__all__ = ['oplist']


class oplist(list):
    """ list with element-wise operations

    List-inherited class implementing direct element operations instead of list-extensions/compressions.
    When having multiple lists and one wishes to create a sum of individual elements, thus
    creating the summed elements list one could do:

    >>> from functools import reduce
    >>> lists = list()
    >>> lists.append([1, 2])
    >>> lists.append([3, 4])
    >>> lists.append([5, 6])
    >>> list_sum = reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), lists)
    >>> print(list_sum)
    [9, 12]

    However, the above easily becomes tedious when the number of entries in the list
    becomes very large.

    Instead it may be advantageous to allow operations like this:

    >>> oplists = list()
    >>> oplists.append(oplist([1, 2]))
    >>> oplists.append(oplist([3, 4]))
    >>> oplists.append(oplist([5, 6]))
    >>> oplist_sum = reduce(sum, oplists)
    >>> print(oplist_sum)
    [9, 12]

    This is more natural when dealing with multiple variables and wanting
    to add them.

    >>> l1 = oplist([1, 2])
    >>> l2 = oplist([3, 4])
    >>> l3 = oplist([5, 6])
    >>> l = l1 + l2 + l3
    >>> print(l)
    [9, 12]
    >>> print(l * 2)
    [18, 24]


    The class also implements a decorator for automatic returning of
    oplist lists.

    >>> @oplist.decorate
    >>> def my_func():
    ...     return 1
    >>> isinstance(my_func(), oplist)
    True

    Currently this class implements the following elementwise mathematical operations

    - additions
    - subtractions
    - multiplications
    - divisions
    - powers

    Parameters
    ----------
    iterable : data
       elements in `oplist`
    """
    __slots__ = ()

    @classmethod
    def decorate(cls, func):
        """ Decorate a function to always return an `oplist`, regardless of return values from `func`

        Parameters
        ----------
        func : callable

        Returns
        -------
        callable
            a wrapped function which ensures the returned argument is an instance of `oplist`

        Examples
        --------

        >>> @oplist.decorate
        >>> def myret():
        ...    return 1
        >>> a = myret()
        >>> isinstance(a, oplist)
        True
        >>> print(a)
        [1]

        """
        @wraps(func)
        def wrap_func(*args, **kwargs):
            val = func(*args, **kwargs)
            if isinstance(val, cls):
                return val
            elif isinstance(val, (tuple, list)):
                # Currently we only capture these as converted
                return cls(val)
            # I should probably check all cases
            return cls([val])

        return wrap_func

    # Implement math operations

    def __add__(self, other):
        if isiterable(other):
            n = len(self)
            if n != len(other):
                raise ValueError('oplist requires other data to contain same number of elements.')
            return oplist([self[i] + other[i] for i in range(n)])
        return oplist([data + other for data in self])

    def __iadd__(self, other):
        n = len(self)
        if isiterable(other):
            if n != len(other):
                raise ValueError('oplist requires other data to contain same number of elements.')
            for i in range(n):
                self[i] += other[i]
        else:
            for i in range(n):
                self[i] += other
        return self

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isiterable(other):
            n = len(self)
            if n != len(other):
                raise ValueError('oplist requires other data to contain same number of elements.')
            return oplist([self[i] - other[i] for i in range(n)])
        return oplist([data - other for data in self])

    def __isub__(self, other):
        n = len(self)
        if isiterable(other):
            if n != len(other):
                raise ValueError('oplist requires other data to contain same number of elements.')
            for i in range(n):
                self[i] -= other[i]
        else:
            for i in range(n):
                self[i] -= other
        return self

    def __rsub__(self, other):
        if isiterable(other):
            n = len(self)
            if n != len(other):
                raise ValueError('oplist requires other data to contain same number of elements.')
            return oplist([other[i] - self[i] for i in range(n)])
        return oplist([other - data for data in self])

    def __mul__(self, other):
        if isiterable(other):
            n = len(self)
            if n != len(other):
                raise ValueError('oplist requires other data to contain same number of elements.')
            return oplist([self[i] * other[i] for i in range(n)])
        return oplist([data * other for data in self])

    def __imul__(self, other):
        n = len(self)
        if isiterable(other):
            if n != len(other):
                raise ValueError('oplist requires other data to contain same number of elements.')
            for i in range(n):
                self[i] *= other[i]
        else:
            for i in range(n):
                self[i] *= other
        return self

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        if isiterable(other):
            n = len(self)
            if n != len(other):
                raise ValueError('oplist requires other data to contain same number of elements.')
            return oplist([self[i] ** other[i] for i in range(n)])
        return oplist([data ** other for data in self])

    def __ipow__(self, other):
        n = len(self)
        if isiterable(other):
            if n != len(other):
                raise ValueError('oplist requires other data to contain same number of elements.')
            for i in range(n):
                self[i] **= other[i]
        else:
            for i in range(n):
                self[i] **= other
        return self

    def __rpow__(self, other):
        if isiterable(other):
            n = len(self)
            if n != len(other):
                raise ValueError('oplist requires other data to contain same number of elements.')
            return oplist([other[i] ** self[i] for i in range(n)])
        return oplist([other ** data for data in self])

    def __truediv__(self, other):
        if isiterable(other):
            n = len(self)
            if n != len(other):
                raise ValueError('oplist requires other data to contain same number of elements.')
            return oplist([self[i] / other[i] for i in range(n)])
        return oplist([data / other for data in self])

    def __itruediv__(self, other):
        n = len(self)
        if isiterable(other):
            if n != len(other):
                raise ValueError('oplist requires other data to contain same number of elements.')
            for i in range(n):
                self[i] /= other[i]
        else:
            for i in range(n):
                self[i] /= other
        return self

    def __rtruediv__(self, other):
        if isiterable(other):
            n = len(self)
            if n != len(other):
                raise ValueError('oplist requires other data to contain same number of elements.')
            return oplist([other[i] / self[i] for i in range(n)])
        return oplist([other / data for data in self])
