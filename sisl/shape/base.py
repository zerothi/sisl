from __future__ import print_function, division

import numpy as np
from numpy import logical_and as log_and
from numpy import logical_or as log_or
from numpy import logical_xor as log_xor
from numpy import logical_not as log_not

from sisl._help import ensure_array


__all__ = ['Shape', 'PureShape']


class Shape(object):
    """ Baseclass for all shapes. Logical operations are implemented on this class.

    **This class *must* be sub classed.**

    Also all the required methods are predefined although they issue an error if they are
    not implemented in the sub-classed class.

    There are a few routines that are necessary when implementing
    an inherited class:

    `center`
      return the geometric center of the shape.

    `within`
      Returns a boolean array which defines whether a coordinate is within
      or outside the shape.

    `within_index`
      Equivalent to `within`, however only the indices of those within are returned.

    `copy`
      Create a new identical shape.

    The minimal requirement a shape can have are the above attributes.

    Subclassed shapes may have additional methods by which they are defined.

    Any `BaseShape` may be used to construct other shapes by applying set operations.
    Currently implemented binary operators are:

    `__or__`/`__add__` : set union
    `__and__` : set intersection
    `__sub__` : set complement
    `__xor__` : set disjunctive union
    """
    __slots__ = ('_center', )

    def __init__(self, center):
        """ Initialize the Shape with a center """
        if center is None:
            self._center = np.zeros(3, np.float64)
        else:
            self._center = np.array(center, np.float64)

    @property
    def center(self):
        """ The geometric center of the shape """
        return self._center

    def scale(self, scale):
        """ Return a new Shape with a scaled size """
        raise NotImplementedError('scale has not been implemented in: '+self.__class__.__name__)

    def within(self, other):
        """ Return ``True`` if `other` is fully within `self`

        If `other` is an array, an array will be returned for each of these.

        Parameters
        ----------
        other : array_like
           the array/object that is checked for containment
        """
        other = ensure_array(other, np.float64)
        ndim = other.ndim
        other.shape = (-1, 3)

        idx = self.within_index(other)
        # Initialize a boolean array with all false
        within = np.zeros(len(other), dtype=bool)
        within[idx] = True
        if ndim == 1 and other.size == 3:
            return within[0]
        return within

    def within_index(self, other):
        """ Return indices of the elements of `other` that are within the shape """
        raise NotImplementedError('within_index has not been implemented in: '+self.__class__.__name__)

    def __contains__(self, other):
        """ Checks whether all of `other` is within the shape """
        return np.all(self.within(other))

    def __repr__(self):
        return "{{" + self.__class__.__name__ + ' c({} {} {})'.format(*self.center)

    # Implement logical operators to enable composition of sets
    def __and__(self, other):
        return CompositeShape(self, other, CompositeShape._AND)

    def __or__(self, other):
        return CompositeShape(self, other, CompositeShape._OR)

    def __add__(self, other):
        return CompositeShape(self, other, CompositeShape._OR)

    def __sub__(self, other):
        return CompositeShape(self, other, CompositeShape._SUB)

    def __xor__(self, other):
        return CompositeShape(self, other, CompositeShape._XOR)


class CompositeShape(Shape):
    """ A composite shape consisting of two shapes

    This should take 2 shapes as arguments and a binary operator to define
    how the shapes are related.


    Parameters
    ----------
    A : BaseShape
       the left hand side of the set operation
    A : BaseShape
       the left hand side of the set operation
    """

    # Internal variables to handle set-operations
    _OR = 0
    _AND = 1
    _SUB = 2
    _XOR = 3

    __slots__ = ('A', 'B', 'op')

    def __init__(self, A, B, op):
        self.A = A.copy()
        self.B = B.copy()
        self.op = op

    def center(self):
        """ Average center of composite shapes """
        return (self.A.center() + self.B.center()) * 0.5

    def volume(self):
        # The volume for these set operators cannot easily be defined, so
        # we should rather not do anything about it.
        return -1.

    def within(self, *args, **kwargs):
        A = self.A.within(*args, **kwargs)
        B = self.B.within(*args, **kwargs)
        op = self.op
        if op == self._OR:
            return log_or(A, B)
        elif op == self._AND:
            return log_and(A, B)
        elif op == self._SUB:
            return log_and(A, log_not(B))
        elif op == self._XOR:
            return log_xor(A, B)

    def __repr__(self):
        A = repr(self.A).replace('\n', '\n ')
        B = repr(self.B).replace('\n', '\n ')
        op = {self._OR: 'OR', self._AND: 'AND', self._SUB: 'SUB', self._XOR: 'XOR'}.get(self.op)
        return '{0}{{op: {1},\n {2},\n {3}\n}}'.format(self.__class__.__name__, op, A, B)

    def scale(self, scale):
        return self.__class__(self.A.scale(scale), self.B.scale(scale), self.op)

    def copy(self):
        return self.__class__(self.A, self.B, self.op)


class PureShape(Shape):
    """ Extension of the `Shape` class for additional well defined shapes

    This shape should be used when subclassing shapes where the volume of the
    shape is *exactly* known.

    `volume`
      return the volume of the shape.
    """

    def volume(self):
        raise NotImplementedError('volume has not been implemented in: '+self.__class__.__name__)
