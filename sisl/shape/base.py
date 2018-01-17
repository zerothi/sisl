from __future__ import print_function, division

import numpy as np
from numpy import logical_and as log_and
from numpy import logical_or as log_or
from numpy import logical_xor as log_xor
from numpy import logical_not as log_not


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
        """ Return ``True`` if `other` is fully within `self` """
        raise NotImplementedError('within has not been implemented in: '+self.__class__.__name__)

    def within_index(self, other):
        """ Return indices of the elements of `other` that are within the shape """
        return self.within(other).nonzero()[0]

    def __contains__(self, other):
        """ Checks whether all of `other` is within the shape """
        return np.all(self.within(other))

    def __repr__(self):
        return self.__class__.__name__ + ' c({} {} {})'.format(*self.center)

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


class _CompositeShape(Shape):
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
        if isinstance(self.A, CompositeShape):
            A = '(' + repr(self.A) + ')'
        else:
            A = repr(self.A)
        if isinstance(self.B, CompositeShape):
            B = '(' + repr(self.B) + ')'
        else:
            B = repr(self.B)
        op = {self._OR: 'OR', self._AND: 'AND', self._SUB: 'SUB', self._XOR: 'XOR'}.get(self.op)
        return '{{{}\n {}\n {}}}'.format(A.replace('\n', '\n '), op, B.replace('\n', '\n '))

    def scale(self, scale):
        return self.__class__.__init__(self.A.scale(scale), self.B.scale(scale), self.op)

    def copy(self, *args, **kwargs):
        return self.__class__.__init__(self.A, self.B, self.op)


class PureShape(Shape):
    """ Extension of the `Shape` class for additional well defined shapes

    This shape should be used when subclassing shapes where the volume of the
    shape is *exactly* known.

    `volume`
      return the volume of the shape.
    """

    def volume(self):
        raise NotImplementedError('volume has not been implemented in: '+self.__class__.__name__)


class OldShape(Shape):
    """ A shape that Baseclass for shapes

    This class should always be sub-classed.

    There are a few routines that are *always* necessary
    to implement in a inherited class:

    `center`
      return the geometric center of the shape

    `origo`
      return the lowest left point in the shape

    `volume`
      return the volume of the shape.

    `displacement`
      return a vector that is the largest displacement vector such that
      a grid of the Shape will fully contian all space

    `within`
      Enables to check if coordinates, or other things
      are contained in this shape

    iwithin
      Returns only the indices of elements that are within

    enlarge
      Creates a new shape with a only the indices of elements that are within

    A `Shape` allows interaction with outside elements to check if geometric
    points are within the shapes.
    For instance to assert that a given point `x`, `y`, `z` is within a sphere
    of radius `r` with center `cx`, `cy`, `cz` on may do:

    This makes it very efficient to enable arbitrary shapes to be passed
    and used as determinations of regions of space.
    """

    def __init__(self, center):
        """ Initialize the Shape with a center """
        if center is None:
            self._center = np.zeros(3, np.float64)
        else:
            self._center = np.array(center, np.float64)

    def __call__(self, *args, **kwargs):
        """ Re-initialize the Shape """
        return self.__class__(*args, **kwargs)

    @property
    def origo(self):
        """ The geometric origo of the shape

        An origo should *always* be the lowest left coordinate of the shape.

        Notes
        -----
        Not all shapes have an origo. For instance a sphere only have a center,
        but an origo cannot be defined.
        """
        return None

    @property
    def center(self):
        """ The geometric center of the shape """
        return self._center

    @property
    def displacement(self):
        """ Return a displacement vector for full containment """
        raise NotImplementedError('displacement has not been implemented in: '+self.__class__.__name__)

    @property
    def volume(self):
        raise NotImplementedError('volume has not been implemented in: '+self.__class__.__name__)

    def enlarge(self, length):
        """ Return a new Shape with an increased size length """
        raise NotImplementedError('enlarge has not been implemented in: '+self.__class__.__name__)

    def within(self, other):
        """ Returns `True` if `other` is fully within `self` """
        raise NotImplementedError('__contains__ has not been implemented in: '+self.__class__.__name__)

    def iwithin(self, other):
        """ Returns indices of the elements of `other` that are within the shape """
        return self.within(other).nonzero()[0]

    def __contains__(self, other):
        return self.within(other)

    def __repr__(self):
        return self.__class__.__name__ + ' c({} {} {})'.format(*self.center)
