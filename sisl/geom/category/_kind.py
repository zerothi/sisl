from functools import reduce, partial, wraps
import operator as op
from numbers import Integral

import numpy as np
from numpy import dot

from sisl._internal import set_module, singledispatchmethod
from sisl._help import isiterable
from .base import AtomCategory, NullCategory, _sanitize_loop


__all__ = ["AtomZ", "AtomIndex", "AtomOdd", "AtomEven"]


@set_module("sisl.geom")
class AtomZ(AtomCategory):
    r""" Classify atoms based on atomic number

    Parameters
    ----------
    Z : int or array_like
       atomic number match for several values this is equivalent to AND
    """
    __slots__ = ("_Z",)

    def __init__(self, Z):
        if isiterable(Z):
            self._Z = set(Z)
        else:
            self._Z = set([Z])
        # using a sorted list ensures that we can compare
        super().__init__(f"Z={self._Z}")

    @_sanitize_loop
    def categorize(self, geometry, atoms=None):
        # _sanitize_loop will ensure that atoms will always be an integer
        if geometry.atoms.Z[atoms] in self._Z:
            return self
        return NullCategory()

    def __eq__(self, other):
        if isinstance(other, (list, tuple, np.ndarray)):
            # this *should* use the dispatch method for different classes
            return super().__eq__(other)

        if self.__class__ is other.__class__:
            return len(self._Z ^ other._Z) == 0
        return False


@set_module("sisl.geom")
class AtomIndex(AtomCategory):
    r""" Classify atoms based on indices

    Parameters
    ----------
    *args : int or list of int
       each value will be equivalent to ``in=set(args)``
    **kwargs : key, value
       if key is a function it must accept two values ``value, atom``
       where ``value`` is the value on this command. The function should
       return anything that can be interpreted as a True/False.
       Multiple ``key`` equates to an `and` statement.

    Examples
    --------
    >>> aidx = AtomIndex(1, 4, 5)
    >>> geom.sub(aidx) == geom.sub([1, 4, 5])
    >>> aidx = AtomIndex(mod=2) # odd indices
    >>> geom.sub(aidx) == geom.sub(range(1, len(geom), 2))
    >>> aidx = ~AtomIndex(mod=2) # even indices
    >>> geom.sub(aidx) == geom.sub(range(0, len(geom), 2))
    >>> aidx = ~AtomIndex(mod=3) # every 3rd atom
    >>> geom.sub(aidx) == geom.sub(range(0, len(geom), 3))
    >>> aidx = AtomIndex(mod=3) # [1, 2, 4, 5, ...]: range(na) - range(0, na, 3)
    >>> geom.sub(aidx) == geom.sub(range(0, len(geom), 3))
    """
    __slots__ = ("_op_val",)

    def __init__(self, *args, **kwargs):
        idx = set()
        for arg in args:
            if isinstance(arg, Integral):
                idx.add(arg)
            else:
                idx.update(arg)
        for key_a in ["eq", "in", "contains"]:
            for key in [key_a, f"__{key_a}__"]:
                arg = kwargs.pop(key, set())
                if isinstance(arg, Integral):
                    idx.add(arg)
                else:
                    idx.update(arg)

        # Now create the list of operators
        def wrap_func(func):
            @wraps(func)
            def make_partial(a, b):
                """ Wrapper to make partial useful """
                if isinstance(b, Integral):
                    return op.truth(func(a, b))
                is_true = True
                for ib in b:
                    is_true = is_true and func(a, ib)
                return is_true
            return make_partial

        if len(idx) == 0:
            operator = []
        else:
            @wraps(op.contains)
            def func_wrap(a, b):
                return op.contains(b, a)
            operator.append((func_wrap, b))

        for func, value in kwargs.items():
            if callable(func):
                # it has to be called like this:
                #   func(atom, value)
                # this makes it easier to handle stuff
                operator.append((wrap_func(func), value))
            else:
                # Certain attributes cannot be directly found (and => and_)
                try:
                    attr_op = getattr(op, func)
                except AttributeError:
                    attr_op = getattr(op, f"{func}_")
                operator.append((wrap_func(attr_op), value))

        # Create string
        self._op_val = operator
        super().__init__(" & ".join(map(lambda f, b: f"{f.__name__}[{b}]", *zip(*self._op_val))))

    @_sanitize_loop
    def categorize(self, geometry, atoms=None):
        # _sanitize_loop will ensure that atoms will always be an integer
        if reduce(op.and_, (f(atoms, b) for f, b in self._op_val), True):
            return self
        return NullCategory()

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            if len(self._op_val) == len(other._op_val):
                # Check they are the same
                return reduce(op.and_, (op_val in other._op_val for op_val in self._op_val), True)
        return False


class AtomEven(AtomCategory):
    r""" Classify atoms based on indices (even in this case)"""
    __slots__ = []

    def __init__(self):
        super().__init__("even")

    @_sanitize_loop
    def categorize(self, geometry, atoms):
        # _sanitize_loop will ensure that atoms will always be an integer
        if atoms % 2 == 0:
            return self
        return NullCategory()

    def __eq__(self, other):
        return self.__class__ is other.__class__


@set_module("sisl.geom")
class AtomOdd(AtomCategory):
    r""" Classify atoms based on indices (odd in this case)"""
    __slots__ = []

    def __init__(self):
        super().__init__("odd")

    @_sanitize_loop
    def categorize(self, geometry, atoms):
        # _sanitize_loop will ensure that atoms will always be an integer
        if atoms % 2 == 1:
            return self
        return NullCategory()

    def __eq__(self, other):
        return self.__class__ is other.__class__
