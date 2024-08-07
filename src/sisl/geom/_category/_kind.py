# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import operator as op
import re
from functools import reduce, wraps
from numbers import Integral
from typing import Union

import numpy as np

from sisl._core import Geometry
from sisl._help import isiterable
from sisl._internal import set_module
from sisl.typing import AtomsIndex
from sisl.utils import lstranges, strmap

from .base import AtomCategory, NullCategory, _sanitize_loop

__all__ = ["AtomZ", "AtomIndex", "AtomSeq", "AtomTag", "AtomOdd", "AtomEven"]


@set_module("sisl.geom")
class AtomZ(AtomCategory):
    r"""Classify atoms based on atomic number

    Parameters
    ----------
    Z : int or array_like
       atomic number match for several values this is equivalent to AND
    """

    __slots__ = ("_Z",)

    def __init__(self, Z: Union[int, Sequence[int]]):
        if isiterable(Z):
            self._Z = set(Z)
        else:
            self._Z = set([Z])
        # using a sorted list ensures that we can compare
        super().__init__(f"Z={self._Z}")

    @_sanitize_loop
    def categorize(self, geometry: Geometry, atoms: AtomsIndex = None):
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
class AtomTag(AtomCategory):
    r"""Classify atoms based on their tag.

    Parameters
    ----------
    tag : str
       The tag you want atoms to match. It can be a regex expression.
    """

    __slots__ = ("_compiled_re", "_re")

    def __init__(self, tag: str):
        self._re = tag
        self._compiled_re = re.compile(self._re)
        super().__init__(f"tag={self._re}")

    @_sanitize_loop
    def categorize(self, geometry: Geometry, atoms: AtomsIndex = None):
        # _sanitize_loop will ensure that atoms will always be an integer
        if self._compiled_re.match(geometry.atoms[atoms].tag):
            return self
        return NullCategory()

    def __eq__(self, other):
        if isinstance(other, (list, tuple, np.ndarray)):
            # this *should* use the dispatch method for different classes
            return super().__eq__(other)

        if self.__class__ is other.__class__:
            return self._re == other._re
        return False


@set_module("sisl.geom")
class AtomIndex(AtomCategory):
    r"""Classify atoms based on indices

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

        for key_a in ("eq", "in", "contains"):
            for key in (key_a, f"__{key_a}__"):
                arg = kwargs.pop(key, set())
                if isinstance(arg, Integral):
                    idx.add(arg)
                else:
                    idx.update(arg)

        # Now create the list of operators
        def wrap_func(func):
            @wraps(func)
            def make_partial(a, b):
                """Wrapper to make partial useful"""
                if isinstance(b, Integral):
                    return op.truth(func(a, b))

                is_true = True
                for ib in b:
                    is_true = is_true and func(a, ib)
                return is_true

            return make_partial

        operator = []
        if len(idx) > 0:

            @wraps(op.contains)
            def func_wrap(a, b):
                return op.contains(b, a)

            operator.append((func_wrap, idx))

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
        super().__init__(
            " & ".join(map(lambda f, b: f"{f.__name__}[{b}]", *zip(*self._op_val)))
        )

    @_sanitize_loop
    def categorize(self, geometry: Geometry, atoms: AtomsIndex = None):
        # _sanitize_loop will ensure that atoms will always be an integer
        if reduce(op.and_, (f(atoms, b) for f, b in self._op_val), True):
            return self
        return NullCategory()

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            if len(self._op_val) == len(other._op_val):
                # Check they are the same
                return reduce(
                    op.and_, (op_val in other._op_val for op_val in self._op_val), True
                )
        return False


@set_module("sisl.geom")
class AtomSeq(AtomIndex):
    r"""Classify atoms based on their indices using a sequence string.

    Parameters
    ----------
    seq: str
       sequence indicating the indices that you want to match (see examples)
    **kwargs : key, value
       if key is a function it must accept two values ``value, atom``
       where ``value`` is the value on this command. The function should
       return anything that can be interpreted as a True/False.
       Multiple ``key`` equates to an `and` statement.

    Examples
    --------
    >>> seq = AtomSeq("1-3")
    >>> geom.sub(seq) == geom.sub([1,2,3])
    >>> seq = AtomSeq("1-3,7")
    >>> geom.sub(seq) == geom.sub([1,2,3,7])
    >>> seq = AtomSeq("1-3,7:")
    >>> geom.sub(seq) == geom.sub([1,2,3,*range(7, len(geom))])
    >>> seq = AtomSeq("1-3,6,9:2:")
    >>> geom.sub(seq) == geom.sub([1,2,3,6,*range(9, len(geom), 2)])

    See also
    ---------
    `strmap`, `lstranges`:
        the functions used to parse the sequence string into indices.
    """

    __slots__ = ("_seq",)

    def __init__(self, seq):
        self._seq = seq
        self._name = seq

    @staticmethod
    def _sanitize_negs(indices_map, end):
        """Converts negative indices to their corresponding positive values.

        Parameters
        ----------
        indices_map: list
            The indices map returned by strmap.
        end: int
            The largest valid index.
        """

        def _sanitize(item):
            if isinstance(item, int):
                if item < 0:
                    return end + item + 1
                else:
                    return item
            elif isinstance(item, tuple):
                return (_sanitize(item[0]), *item[1:-1], _sanitize(item[-1]))
            elif item is None:
                return item
            else:
                raise ValueError(f"Item {item} could not be parsed")

        return [_sanitize(item) for item in indices_map]

    def categorize(self, geometry: Geometry, *args, **kwargs):
        # Now that we have the geometry, we know what is the end index
        # and we can finally safely convert the sequence to indices.
        indices_map = strmap(int, self._seq, start=0, end=geometry.na - 1)
        indices = lstranges(self._sanitize_negs(indices_map, end=geometry.na - 1))

        # Initialize the machinery of AtomIndex
        super().__init__(indices)
        self.name = self._seq
        # Finally categorize
        return super().categorize(geometry, *args, **kwargs)

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self._seq == other._seq
        return False


class AtomEven(AtomCategory):
    r"""Classify atoms based on indices (even in this case)"""

    __slots__ = ()

    def __init__(self, name="even"):
        super().__init__(name)

    @_sanitize_loop
    def categorize(self, geometry: Geometry, atoms: AtomsIndex = None):
        # _sanitize_loop will ensure that atoms will always be an integer
        if atoms % 2 == 0:
            return self
        return NullCategory()

    def __eq__(self, other):
        return self.__class__ is other.__class__


@set_module("sisl.geom")
class AtomOdd(AtomCategory):
    r"""Classify atoms based on indices (odd in this case)"""

    __slots__ = ()

    def __init__(self, name="odd"):
        super().__init__(name)

    @_sanitize_loop
    def categorize(self, geometry: Geometry, atoms: AtomsIndex = None):
        # _sanitize_loop will ensure that atoms will always be an integer
        if atoms % 2 == 1:
            return self
        return NullCategory()

    def __eq__(self, other):
        return self.__class__ is other.__class__
