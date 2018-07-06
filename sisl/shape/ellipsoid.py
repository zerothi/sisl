from __future__ import print_function, division

from math import pi
import numpy as np
from numpy import dot

from sisl.messages import warn
import sisl._array as _a
from sisl.utils.mathematics import orthogonalize, fnorm, fnorm2, expand
from sisl._math_small import product3
from sisl._indices import indices_in_sphere
from .base import PureShape


__all__ = ['Ellipsoid', 'Sphere']


class Ellipsoid(PureShape):
    """ 3D Ellipsoid shape

    Parameters
    ----------
    v : float or (3,) or (3, 3)
       radius/vectors defining the ellipsoid. For 3 values it corresponds to a Cartesian
       oriented ellipsoid. If the vectors are non-orthogonal they will be orthogonalized.
       I.e. the first vector is considered a principal axis, then the second vector will
       be orthogonalized onto the first, and this is the second principal axis. And so on.
    center : (3,), optional
       the center of the ellipsoid. Defaults to the origo.

    Examples
    --------
    >>> shape = Ellipsoid([2, 2.2, 2])
    >>> shape.within([0, 2, 0])
    True
    """
    __slots__ = ('_v', '_iv')

    def __init__(self, v, center=None):
        super(Ellipsoid, self).__init__(center)
        v = _a.asarrayd(v)
        if v.size == 1:
            self._v = np.identity(3, np.float64) * v # a "Euclidean" sphere
        elif v.size == 3:
            self._v = np.diag(v.ravel()) # a "Euclidean" ellipsoid
        elif v.size == 9:
            self._v = v.reshape(3, 3).astype(np.float64)
        else:
            raise ValueError(self.__class__.__name__ + " requires initialization with 3 vectors defining the ellipsoid")

        # If the vectors are not orthogonal, orthogonalize them and issue a warning
        vv = np.fabs(np.dot(self._v, self._v.T) - np.diag(fnorm2(self._v)))
        if vv.sum() > 1e-9:
            warn(self.__class__.__name__ + ' principal vectors are not orthogonal. '
                 'sisl orthogonalizes the vectors (retaining 1st vector)!')

        self._v[1, :] = orthogonalize(self._v[0, :], self._v[1, :])
        self._v[2, :] = orthogonalize(self._v[0, :], self._v[2, :])
        self._v[2, :] = orthogonalize(self._v[1, :], self._v[2, :])

        # Create the reciprocal cell
        self._iv = np.linalg.inv(self._v)

    def copy(self):
        return self.__class__(self._v, self.center)

    def __str__(self):
        cr = np.array([self.center, self.radius])
        return self.__class__.__name__ + ('{{c({0:.2f} {1:.2f} {2:.2f}) '
                                          'r({3:.2f} {4:.2f} {5:.2f})}}').format(*cr.ravel())

    def volume(self):
        """ Return the volume of the shape """
        return 4. / 3. * pi * product3(self.radius)

    def scale(self, scale):
        """ Return a new shape with a larger corresponding to `scale`

        Parameters
        ----------
        scale : float or (3,)
            the scale parameter for each of the vectors defining the `Ellipsoid`
        """
        scale = _a.asarrayd(scale)
        if scale.size == 3:
            scale.shape = (3, 1)
        return self.__class__(self._v * scale, self.center)

    def expand(self, radius):
        """ Expand ellipsoid by a constant value along each radial vector

        Parameters
        ----------
        radius : float or (3,)
           the extension in Ang per ellipsoid radial vector
        """
        radius = _a.asarrayd(radius)
        if radius.size == 1:
            v0 = expand(self._v[0, :], radius)
            v1 = expand(self._v[1, :], radius)
            v2 = expand(self._v[2, :], radius)
        elif radius.size == 3:
            v0 = expand(self._v[0, :], radius[0])
            v1 = expand(self._v[1, :], radius[1])
            v2 = expand(self._v[2, :], radius[2])
        else:
            raise ValueError(self.__class__.__name__ + '.expand requires the radius to be either (1,) or (3,)')
        return self.__class__([v0, v1, v2], self.center)

    def toEllipsoid(self):
        """ Return an ellipsoid that encompass this shape (a copy) """
        return self.copy()

    def toSphere(self):
        """ Return a sphere with a radius equal to the largest radial vector """
        r = self.radius.max()
        return Sphere(r, self.center)

    def toCuboid(self):
        """ Return a cuboid with side lengths equal to the diameter of each ellipsoid vectors """
        from .prism4 import Cuboid
        return Cuboid(self._v * 2, self.center)

    def set_center(self, center):
        """ Change the center of the object """
        super(Ellipsoid, self).__init__(center)

    def within_index(self, other):
        """ Return indices of the points that are within the shape """
        other = _a.asarrayd(other)
        other.shape = (-1, 3)

        # First check
        tmp = dot(other - self.center[None, :], self._iv)
        tol = 1.e-12

        # Get indices where we should do the more
        # expensive exact check of being inside shape
        # I.e. this reduces the search space to the box
        return indices_in_sphere(tmp, 1. + tol)

    @property
    def radius(self):
        """ Return the radius of the Ellipsoid """
        return fnorm(self._v)


class Sphere(Ellipsoid):
    """ 3D Sphere

    Parameters
    ----------
    r : float
       radius of the sphere
    """

    def __init__(self, radius, center=None):
        radius = _a.asarrayd(radius).ravel()
        if len(radius) > 1:
            raise ValueError(self.__class__.__name__ + ' is defined via a single radius. '
                             'An array with more than 1 element is not an allowed argument '
                             'to __init__.')
        super(Sphere, self).__init__(radius, center=center)

    def __str__(self):
        return '{0}{{c({2:.2f} {3:.2f} {4:.2f}) r({1:.2f})}}'.format(self.__class__.__name__, self.radius, *self.center)

    def copy(self):
        return self.__class__(self.radius, self.center)

    def volume(self):
        """ Return the volume of the sphere """
        return 4. / 3. * pi * self.radius ** 3

    def scale(self, scale):
        """ Return a new sphere with a larger radius

        Parameters
        ----------
        scale : float
            the scale parameter for the radius
        """
        return self.__class__(self.radius * scale, self.center)

    def expand(self, radius):
        """ Expand sphere by a constant radius

        Parameters
        ----------
        radius : float
           the extension in Ang per ellipsoid radial vector
        """
        return self.__class__(self.radius + radius, self.center)

    @property
    def radius(self):
        """ Return the radius of the Sphere """
        return self._v[0, 0]

    def toSphere(self):
        """ Return a copy of it-self """
        return self.copy()

    def toEllipsoid(self):
        """ Convert this sphere into an ellipsoid """
        return Ellipsoid(self.radius, self.center)
