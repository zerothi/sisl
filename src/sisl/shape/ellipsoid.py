# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from math import pi

import numpy as np

import sisl._array as _a
from sisl._indices import indices_in_sphere
from sisl._internal import set_module
from sisl._math_small import product3
from sisl.messages import deprecate_argument, deprecation, warn
from sisl.utils.mathematics import expand, fnorm, fnorm2, orthogonalize

from .base import PureShape, ShapeToDispatch

__all__ = ["Ellipsoid", "Sphere"]


@set_module("sisl.shape")
class Ellipsoid(PureShape):
    """3D Ellipsoid shape

    Parameters
    ----------
    v : float or (3,) or (3, 3)
       radius/vectors defining the ellipsoid. For 3 values it corresponds to a Cartesian
       oriented ellipsoid. If the vectors are non-orthogonal they will be orthogonalized.
       I.e. the first vector is considered a principal axis, then the second vector will
       be orthogonalized onto the first, and this is the second principal axis. And so on.
    center : (3,), optional
       the center of the ellipsoid. Defaults to the origin.

    Examples
    --------
    >>> shape = Ellipsoid([2, 2.2, 2])
    >>> shape.within([0, 2, 0])
    True
    """

    __slots__ = ("_v", "_iv")

    def __init__(self, v, center=None):
        super().__init__(center)
        v = _a.asarrayd(v)
        if v.size == 1:
            self._v = np.identity(3, np.float64) * v  # a "Euclidean" sphere
        elif v.size == 3:
            self._v = np.diag(v.ravel())  # a "Euclidean" ellipsoid
        elif v.size == 9:
            self._v = v.reshape(3, 3).astype(np.float64)
        else:
            raise ValueError(
                self.__class__.__name__
                + " requires initialization with 3 vectors defining the ellipsoid"
            )

        # If the vectors are not orthogonal, orthogonalize them and issue a warning
        vv = np.fabs(np.dot(self._v, self._v.T) - np.diag(fnorm2(self._v)))
        if vv.sum() > 1e-9:
            warn(
                self.__class__.__name__ + " principal vectors are not orthogonal. "
                "sisl orthogonalizes the vectors (retaining 1st vector)!"
            )

        self._v[1, :] = orthogonalize(self._v[0, :], self._v[1, :])
        self._v[2, :] = orthogonalize(self._v[0, :], self._v[2, :])
        self._v[2, :] = orthogonalize(self._v[1, :], self._v[2, :])

        # Create the reciprocal cell
        self._iv = np.linalg.inv(self._v)

    def copy(self):
        return self.__class__(self._v, self.center)

    @property
    def volume(self) -> float:
        """Return the volume of the shape"""
        return 4.0 / 3.0 * pi * product3(self.radius)

    @property
    def radius(self):
        """Return the radius of the Ellipsoid"""
        return fnorm(self._v)

    @property
    def radial_vector(self):
        """The radial vectors"""
        return self._v

    def __str__(self):
        cr = np.array([self.center, self.radius])
        return self.__class__.__name__ + (
            "{{c({0:.2f} {1:.2f} {2:.2f}) " "r({3:.2f} {4:.2f} {5:.2f})}}"
        ).format(*cr.ravel())

    def scale(self, scale):
        """Return a new shape with a larger corresponding to `scale`

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
        """Expand ellipsoid by a constant value along each radial vector

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
            raise ValueError(
                self.__class__.__name__
                + ".expand requires the radius to be either (1,) or (3,)"
            )
        return self.__class__([v0, v1, v2], self.center)

    @deprecation(
        "toEllipsoid is deprecated, use shape.to['ellipsoid'](...) instead.",
        "0.15",
        "0.17",
    )
    def toEllipsoid(self):
        """Return an ellipsoid that encompass this shape (a copy)"""
        return self.copy()

    @deprecation(
        "toSphere is deprecated, use shape.to['sphere'](...) instead.", "0.15", "0.17"
    )
    def toSphere(self):
        """Return a sphere with a radius equal to the largest radial vector"""
        r = self.radius.max()
        return Sphere(r, self.center)

    @deprecation(
        "toCuboid is deprecated, use shape.to['cuboid'](...) instead.", "0.15", "0.17"
    )
    def toCuboid(self):
        """Return a cuboid with side lengths equal to the diameter of each ellipsoid vectors"""
        from .prism4 import Cuboid

        return Cuboid(self._v * 2, self.center)

    @deprecate_argument(
        "tol",
        "rtol",
        "argument tol has been deprecated in favor of rtol, please update your code.",
        "0.15",
        "0.17",
    )
    def within_index(self, other, rtol: float = 1.0e-8):
        r"""Return indices of the points that are within the shape

        Parameters
        ----------
        other : array_like
           the object that is checked for containment
        rtol : float, optional
           absolute tolerance for boundaries
        """
        other = _a.asarrayd(other)
        other.shape = (-1, 3)

        # First check
        tmp = np.dot(other - self.center[None, :], self._iv)

        # Get indices where we should do the more
        # expensive exact check of being inside shape
        # I.e. this reduces the search space to the box
        return indices_in_sphere(tmp, 1.0 + rtol)


to_dispatch = Ellipsoid.to


class EllipsoidToEllipsoid(ShapeToDispatch):
    def dispatch(self, *args, **kwargs):
        return self._get_object().copy()


to_dispatch.register("Ellipsoid", EllipsoidToEllipsoid)


class EllipsoidToSphere(ShapeToDispatch):
    def dispatch(self, *args, **kwargs):
        shape = self._get_object()
        return Sphere(shape.radius.max(), shape.center)


to_dispatch.register("Sphere", EllipsoidToSphere)


class EllipsoidToCuboid(ShapeToDispatch):
    def dispatch(self, *args, **kwargs):
        from .prism4 import Cuboid

        shape = self._get_object()
        return Cuboid(shape._v * 2, shape.center)


to_dispatch.register("Cuboid", EllipsoidToCuboid)

del to_dispatch


@set_module("sisl.shape")
class Sphere(Ellipsoid, dispatchs=[("to", "keep")]):
    """3D Sphere

    Parameters
    ----------
    r : float
       radius of the sphere
    """

    __slots__ = ()

    def __init__(self, radius, center=None):
        radius = _a.asarrayd(radius).ravel()
        if len(radius) > 1:
            raise ValueError(
                self.__class__.__name__ + " is defined via a single radius. "
                "An array with more than 1 element is not an allowed argument "
                "to __init__."
            )
        super().__init__(radius, center=center)

    def __str__(self):
        return "{0}{{c({2:.2f} {3:.2f} {4:.2f}) r({1:.2f})}}".format(
            self.__class__.__name__, self.radius, *self.center
        )

    def copy(self):
        return self.__class__(self.radius, self.center)

    @property
    def volume(self) -> float:
        """Return the volume of the sphere"""
        return 4.0 / 3.0 * pi * self.radius**3

    @property
    def radius(self):
        """Return the radius of the Sphere"""
        return self._v[0, 0]

    def scale(self, scale):
        """Return a new sphere with a larger radius

        Parameters
        ----------
        scale : float
            the scale parameter for the radius
        """
        return self.__class__(self.radius * scale, self.center)

    def expand(self, radius):
        """Expand sphere by a constant radius

        Parameters
        ----------
        radius : float
           the extension in Ang per ellipsoid radial vector
        """
        return self.__class__(self.radius + radius, self.center)

    @deprecation(
        "toSphere is deprecated, use shape.to['sphere'](...) instead.", "0.15", "0.17"
    )
    def toSphere(self):
        """Return a copy of it-self"""
        return self.copy()

    @deprecation(
        "toEllipsoid is deprecated, use shape.to['ellipsoid'](...) instead.",
        "0.15",
        "0.17",
    )
    def toEllipsoid(self):
        """Convert this sphere into an ellipsoid"""
        return Ellipsoid(self.radius, self.center)
