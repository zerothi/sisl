r""" Lists with inter-element operations

In certain cases it may be advantegeous to make operations on list elements rather than doing
list extensions.

This sub-module implements a list which allows to make operations with it-self or with scalars.
"""

from functools import wraps
import operator as op

from ._internal import set_module
from ._help import isiterable


__all__ = ['oplist']


def _crt_op(op, only_self=False):
    if only_self:
        def _(self):
            return self.__class__(op(s) for s in self)
    else:
        def _(self, other):
            if isiterable(other):
                if len(self) != len(other):
                    raise ValueError(f"{self.__class__.__name__} requires other data to contain same number of elements (or a scalar).")
                return self.__class__(op(s, o) for s, o in zip(self, other))
            return self.__class__(op(s, other) for s in self)
    return _


def _crt_rop(op):
    def _(self, other):
        if isiterable(other):
            if len(self) != len(other):
                raise ValueError(f"{self.__class__.__name__} requires other data to contain same number of elements (or a scalar).")
            return self.__class__(op(o, s) for s, o in zip(self, other))
        return self.__class__(op(other, s) for s in self)
    return _


def _crt_iop(op, only_self=False):
    if only_self:
        def _(self):
            for i in range(len(self)):
                self[i] = op(self[i])
            return self
    else:
        def _(self, other):
            if isiterable(other):
                if len(self) != len(other):
                    raise ValueError(f"{self.__class__.__name__} requires other data to contain same number of elements (or a scalar).")
                for i in range(len(self)):
                    self[i] = op(self[i], other[i])
            else:
                for i in range(len(self)):
                    self[i] = op(self[i], other)
            return self
    return _


@set_module("sisl")
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
            if isinstance(val, oplist):
                return val
            elif isinstance(val, cls):
                return val
            elif isinstance(val, (tuple, list)):
                # Currently we only capture these as converted
                return cls(val)
            # I should probably check all cases
            return cls([val])

        return wrap_func

    # Implement math operations
    __abs__ = _crt_op(op.abs, only_self=True)
    __add__ = _crt_op(op.add)
    __radd__ = _crt_rop(op.add)
    __iadd__ = _crt_iop(op.iadd)
    __floordiv__ = _crt_op(op.floordiv)
    __rfloordiv__ = _crt_rop(op.floordiv)
    __ifloordiv__ = _crt_iop(op.ifloordiv)
    __mod__ = _crt_op(op.mod)
    __imod__ = _crt_iop(op.imod)
    __mul__ = _crt_op(op.mul)
    __imul__ = _crt_iop(op.imul)
    __rmul__ = _crt_rop(op.mul)
    __matmul__ = _crt_op(op.matmul)
    __imatmul__ = _crt_iop(op.imatmul)
    __neg__ = _crt_op(op.neg, only_self=True)
    __pos__ = _crt_op(op.pos, only_self=True)
    __pow__ = _crt_op(op.pow)
    __ipow__ = _crt_iop(op.ipow)
    __rpow__ = _crt_rop(op.pow)
    __sub__ = _crt_op(op.sub)
    __isub__ = _crt_iop(op.isub)
    __rsub__ = _crt_rop(op.sub)
    __truediv__ = _crt_op(op.truediv)
    __itruediv__ = _crt_iop(op.itruediv)
    __rtruediv__ = _crt_rop(op.truediv)

    # boolean operations
    __eq__ = _crt_op(op.eq)
    __lt__ = _crt_op(op.lt)
    __le__ = _crt_op(op.le)
    __ge__ = _crt_op(op.ge)
    __gt__ = _crt_op(op.gt)
    __not__ = _crt_op(op.not_, only_self=True)
    __and__ = _crt_op(op.and_)
    __iand__ = _crt_iop(op.iand)
    __or__ = _crt_op(op.or_)
    __ior__ = _crt_iop(op.ior)
    __xor__ = _crt_op(op.xor)
