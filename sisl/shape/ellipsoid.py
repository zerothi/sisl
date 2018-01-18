from __future__ import print_function, division

from math import pi, sqrt
import numpy as np
from numpy import dot
from numpy import fabs, logical_and

from sisl._help import ensure_array
import sisl._array as _a
from sisl.utils.mathematics import orthogonalize, fnorm, fnorm2, expand

from .base import PureShape


__all__ = ['Ellipsoid', 'Sphere']


class Ellipsoid(PureShape):
    """ 3D Ellipsoid shape

    Parameters
    ----------
    v : float or (3,) or (3, 3)
       radius/vectors defining the ellipsoid. For 3 values it corresponds to a Cartesian
       oriented ellipsoid.
    center : (3,), optional
       the center of the ellipsoid. Defaults to the origo.

    Examples
    --------
    >>> xyz = [0, 2, 0]
    >>> shape = Ellipsoid([2, 2.2, 2])
    >>> shape.within(xyz)
    array([ True], dtype=bool)
    """
    __slots__ = ('_v', '_iv')

    def __init__(self, v, center=None):
        super(Ellipsoid, self).__init__(center)
        v = ensure_array(v, np.float64)
        if v.size == 1:
            self._v = np.identity(3) * v # a "Euclidean" sphere
        elif v.size == 3:
            self._v = np.diag(v.ravel()) # a "Euclidean" ellipsoid
        elif v.size == 9:
            self._v = v.reshape(3, 3).astype(np.float64)
        else:
            raise ValueError(self.__class__.__name__ + " requires initialization with 3 vectors defining the ellipsoid")

        # Ensure each of the vectors are orthogonal to each other
        # When performing this operation we do not preserve length of vector.
        self._v[1, :] = orthogonalize(self._v[0, :], self._v[1, :])
        self._v[2, :] = orthogonalize(self._v[0, :], self._v[2, :])
        self._v[2, :] = orthogonalize(self._v[1, :], self._v[2, :])

        # Create the reciprocal cell
        self._iv = np.linalg.inv(self._v).T

    def copy(self):
        return self.__class__(self._v, self.center)

    def __repr__(self):
        cr = np.array([self.center, self.radius])
        return self.__class__.__name__ + ('{{c({0:.2f} {1:.2f} {2:.2f}) '
                                          'r({3:.2f} {4:.2f} {5:.2f})}}').format(*cr.ravel())

    def volume(self):
        """ Return the volume of the shape """
        return 4. / 3. * pi * np.product(self.radius)

    def scale(self, scale):
        """ Return a new shape with a larger corresponding to `scale`

        Parameters
        ----------
        scale : float or (3,)
            the scale parameter for each of the vectors defining the `Ellipsoid`
        """
        scale = ensure_array(scale, np.float64)
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
        radius = ensure_array(radius, np.float64)
        if radius.size == 1:
            v0 = expand(self._v[0, :], radius[0])
            v1 = expand(self._v[1, :], radius[0])
            v2 = expand(self._v[2, :], radius[0])
        elif radius.size == 3:
            v0 = expand(self._v[0, :], radius[0])
            v1 = expand(self._v[1, :], radius[1])
            v2 = expand(self._v[2, :], radius[2])
        else:
            raise ValueError(self.__class__.__name__ + '.expand requires the radius to be either (1,) or (3,)')
        return self.__class__([v0, v1, v2], self.center)

    def set_center(self, center):
        """ Change the center of the object """
        super(Ellipsoid, self).__init__(center)

    def within_index(self, other):
        """ Return indices of the points that are within the shape """
        other = ensure_array(other, np.float64)
        ndim = other.ndim
        other.shape = (-1, 3)

        # First check
        tmp = dot(other - self.center[None, :], self._iv.T)

        # Get indices where we should do the more
        # expensive exact check of being inside shape
        # I.e. this reduces the search space to the box
        within = logical_and.reduce(fabs(tmp) <= 1, axis=1).nonzero()[0]

        # Now only check exactly on those that are possible candidates
        tmp = tmp[within, :]
        wtmp = (fnorm2(tmp) <= 1).nonzero()[0]

        return within[wtmp]

    @property
    def radius(self):
        """ Return the radius of the Ellipsoid """
        return fnorm(self._v)


class Sphere(Ellipsoid):
    """ 3D Sphere

    Equivalent to ``Ellipsoid([r, r, r])``.

    Parameters
    ----------
    r : float
       radius of the sphere
    """

    def __init__(self, radius, center=None):
        radius = ensure_array(radius, np.float64).ravel()[0]
        super(Sphere, self).__init__(radius, center=center)
