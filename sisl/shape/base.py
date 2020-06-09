from math import sqrt as msqrt

import numpy as np
from numpy import union1d, intersect1d, setdiff1d, setxor1d

from sisl._internal import set_module
import sisl._array as _a
from sisl.utils.mathematics import fnorm


__all__ = ["Shape", "PureShape", "NullShape",
           "CompositeShape", "OrShape", "XOrShape", "AndShape", "SubShape"]


@set_module("sisl.shape")
class Shape:
    """ Baseclass for all shapes. Logical operations are implemented on this class.

    **This class must be sub classed.**

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

    Any `Shape` may be used to construct other shapes by applying set operations.
    Currently implemented binary operators are:

    `__or__`/`__add__` : set union, either `|` or `+` operator (not `or`)

    `__and__` : set intersection, `&` operator (not `and`)

    `__sub__` : set complement, `-` operator

    `__xor__` : set disjunctive union, `^` operator

    Parameters
    ----------
    center : (3,)
       the center of the shape
    """
    __slots__ = ('_center', )

    def __init__(self, center):
        if center is None:
            self._center = _a.zerosd(3)
        else:
            self._center = _a.asarrayd(center)

    @property
    def center(self):
        """ The geometric center of the shape """
        return self._center

    def scale(self, scale):
        """ Return a new Shape with a scaled size """
        raise NotImplementedError(f"{self.__class__.__name__}.scale has not been implemented")

    def toSphere(self):
        """ Create a sphere which is surely encompassing the *full* shape """
        raise NotImplementedError(f"{self.__class__.__name__}.toSphere has not been implemented")

    def toEllipsoid(self):
        """ Create an ellipsoid which is surely encompassing the *full* shape """
        return self.toSphere().toEllipsoid()

    def toCuboid(self):
        """ Create a cuboid which is surely encompassing the *full* shape """
        return self.toEllipsoid().toCuboid()

    def within(self, other, *args, **kwargs):
        """ Return ``True`` if `other` is fully within `self`

        If `other` is an array, an array will be returned for each of these.

        Parameters
        ----------
        other : array_like
           the array/object that is checked for containment
        *args :
           passed directly to `within_index`
        **kwargs :
           passed directly to `within_index`
        """
        other = _a.asarrayd(other)
        ndim = other.ndim
        other.shape = (-1, 3)

        idx = self.within_index(other, *args, **kwargs)
        # Initialize a boolean array with all false
        within = np.zeros(len(other), dtype=bool)
        within[idx] = True
        if ndim == 1 and other.size == 3:
            return within[0]
        return within

    def within_index(self, other, *args, **kwargs):
        """ Return indices of the elements of `other` that are within the shape """
        raise NotImplementedError(f"{self.__class__.__name__}.within_index has not been implemented")

    def __contains__(self, other):
        """ Checks whether all of `other` is within the shape """
        return np.all(self.within(other))

    def __str__(self):
        return "{{" + self.__class__.__name__ + ' c({} {} {})'.format(*self.center)

    # Implement logical operators to enable composition of sets
    def __and__(self, other):
        return AndShape(self, other)

    def __or__(self, other):
        return OrShape(self, other)

    def __add__(self, other):
        return OrShape(self, other)

    def __sub__(self, other):
        return SubShape(self, other)

    def __xor__(self, other):
        return XOrShape(self, other)


@set_module("sisl.shape")
class CompositeShape(Shape):
    """ A composite shape consisting of two shapes, an abstract class

    This should take 2 shapes as arguments.

    Parameters
    ----------
    A : Shape
       the left hand side of the set operation
    B : Shape
       the right hand side of the set operation
    """
    __slots__ = ('A', 'B')

    def __init__(self, A, B):
        self.A = A.copy()
        self.B = B.copy()

    @property
    def center(self):
        """ Average center of composite shapes """
        return (self.A.center + self.B.center) * 0.5

    @staticmethod
    def volume():
        """ Volume of a composite shape is current undefined, so a negative number is returned (may change) """
        # The volume for these set operators cannot easily be defined, so
        # we should rather not do anything about it.
        # TODO we could *estimate* the volume by doing
        #      self.toSphere()
        #      grid of density 0.01
        #      within_index
        #      and calculate fractional volume
        #      This is very inaccurate, but would probably be
        #      good enough.
        return -1.

    def toSphere(self):
        """ Create a sphere which is surely encompassing the *full* shape """
        from .ellipsoid import Sphere

        # Retrieve spheres
        A = self.A.toSphere()
        Ar = A.radius
        Ac = A.center
        B = self.B.toSphere()
        Br = B.radius
        Bc = B.center

        center = (Ac + Bc) * 0.5
        A = Ar + fnorm(center - Ac)
        B = Br + fnorm(center - Bc)

        return Sphere(max(A, B), center)

    def scale(self, scale):
        return self.__class__(self.A.scale(scale), self.B.scale(scale))

    def copy(self):
        return self.__class__(self.A, self.B)


def _composite_name(sep):
    def _str(self):
        if isinstance(self.A, CompositeShape):
            A = "({})".format(str(self.A).replace('\n', '\n '))
        else:
            A = "{}".format(str(self.A).replace('\n', '\n '))
        if isinstance(self.B, CompositeShape):
            B = "({})".format(str(self.B).replace('\n', '\n '))
        else:
            B = "{}".format(str(self.B).replace('\n', '\n '))
        return f"{self.__class__.__name__}{{{A} {sep} {B}}}"
    return _str


@set_module("sisl.shape")
class OrShape(CompositeShape):
    """ Boolean ``A | B`` shape """
    __slots__ = ()
    __str__ = _composite_name("|")

    def within_index(self, *args, **kwargs):
        A = self.A.within_index(*args, **kwargs)
        B = self.B.within_index(*args, **kwargs)
        return union1d(A, B)


@set_module("sisl.shape")
class XOrShape(CompositeShape):
    """ Boolean ``A ^ B`` shape """
    __slots__ = ()
    __str__ = _composite_name("^")

    def within_index(self, *args, **kwargs):
        A = self.A.within_index(*args, **kwargs)
        B = self.B.within_index(*args, **kwargs)
        return setxor1d(A, B, assume_unique=True)


@set_module("sisl.shape")
class SubShape(CompositeShape):
    """ Boolean ``A - B`` shape """
    __slots__ = ()
    __str__ = _composite_name("-")

    def within_index(self, *args, **kwargs):
        A = self.A.within_index(*args, **kwargs)
        B = self.B.within_index(*args, **kwargs)
        return setdiff1d(A, B, assume_unique=True)


@set_module("sisl.shape")
class AndShape(CompositeShape):
    """ Boolean ``A & B`` shape """
    __slots__ = ()
    __str__ = _composite_name("&")

    def toSphere(self):
        """ Create a sphere which is surely encompassing the *full* shape """
        from .ellipsoid import Sphere

        # Retrieve spheres
        A = self.A.toSphere()
        Ar = A.radius
        Ac = A.center
        B = self.B.toSphere()
        Br = B.radius
        Bc = B.center

        # Calculate the distance between the spheres
        dist = fnorm(Ac - Bc)

        # If one is fully enclosed in the other, we can simply neglect the other
        if dist + Ar <= Br:
            # A is fully enclosed in B (or they are the same)
            return A

        elif dist + Br <= Ar:
            # B is fully enclosed in A (or they are the same)
            return B

        elif dist <= (Ar + Br):
            # We can reduce the sphere drastically because only the overlapping region is important
            # i_r defines the intersection radius, search for Sphere-Sphere Intersection
            dx = (dist ** 2 - Br ** 2 + Ar ** 2) / (2 * dist)

            if dx > dist:
                # the intersection is placed after the radius of B
                # And in this case B is smaller (otherwise dx < 0)
                return B
            elif dx < 0:
                return A

            i_r = msqrt(4 * (dist * Ar) ** 2 - (dist ** 2 - Br ** 2 + Ar ** 2) ** 2) / (2 * dist)

            # Now we simply need to find the dx point along the vector Bc - Ac
            # Then we can easily calculate the point from A
            center = Bc - Ac
            center = Ac + center / fnorm(center) * dx
            A = i_r
            B = i_r

        else:
            # In this case there is actually no overlap. So perhaps we should
            # create an infinitisemal sphere such that no point will ever be found
            # Or we should return a new Shape which *always* return False for indices etc.
            center = (Ac + Bc) * 0.5
            # Currently we simply use a very small sphere and put it in the middle between
            # the spheres
            # This should at least speed up comparisons
            A = 0.001
            B = 0.001

        return Sphere(max(A, B), center)

    def within_index(self, *args, **kwargs):
        A = self.A.within_index(*args, **kwargs)
        B = self.B.within_index(*args, **kwargs)
        return intersect1d(A, B, assume_unique=True)


@set_module("sisl.shape")
class PureShape(Shape):
    """ Extension of the `Shape` class for additional well defined shapes

    This shape should be used when subclassing shapes where the volume of the
    shape is *exactly* known.

    `volume`
      return the volume of the shape.
    """
    __slots__ = ()

    def volume(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}.volume has not been implemented")

    def expand(self, c):
        """ Expand the shape by a constant value """
        raise NotImplementedError(f"{self.__class__.__name__}.expand has not been implemented")


@set_module("sisl.shape")
class NullShape(PureShape):
    """ A unique shape which has no well-defined spatial volume or center

    This special shape is used when composite shapes turns out to have
    a null space.

    The center will be equivalent to the maximum floating point value
    divided by 100.

    Initialization of the NullShape takes no (or any) arguments.
    Since it has no volume of point in space, none of the arguments
    has any meaning.
    """
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        """ Initialize a null-shape """
        M = np.finfo(np.float64).max / 100
        self._center = np.array([M, M, M], np.float64)

    def within_index(self, other, *args, **kwargs):
        """ Always returns a zero length array """
        return np.empty(0, dtype=np.int32)

    def toEllipsoid(self):
        """ Return an ellipsoid with radius of size 1e-64 """
        from .ellipsoid import Ellipsoid
        return Ellipsoid(1.e-64, center=self.center.copy())

    def toSphere(self):
        """ Return a sphere with radius of size 1e-64 """
        from .ellipsoid import Sphere
        return Sphere(1.e-64, center=self.center.copy())

    def toCuboid(self):
        """ Return a cuboid with side-lengths 1e-64 """
        from .prism4 import Cuboid
        return Cuboid(1.e-64, center=self.center.copy())

    def volume(self, *args, **kwargs):
        """ The volume of a null shape is exactly 0. """
        return 0.
