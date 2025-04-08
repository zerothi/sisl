# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from math import pi

import numpy as np

import sisl._array as _a
from sisl._indices import indices_in_cylinder
from sisl._internal import set_module
from sisl.messages import deprecate_argument, deprecation, warn
from sisl.utils.mathematics import expand, fnorm, fnorm2, orthogonalize

from .base import PureShape, ShapeToDispatch

__all__ = ["EllipticalCylinder"]


@set_module("sisl.shape")
class EllipticalCylinder(PureShape):
    r"""3D elliptical cylinder

    Parameters
    ----------
    v : float or (2,) or (2, 3)
       radius/vectors defining the elliptical base.
       For 1 value the xy plane will be the elliptical base.
       For 2 values it corresponds to a Cartesian
       oriented ellipsoid base. If the vectors are non-orthogonal they will be orthogonalized.
       I.e. the first vector is considered a principal axis, then the second vector will
       be orthogonalized onto the first, and this is the second principal axis.
    h : float
        the height of the cylinder, this is a *right* cylinder (not oblique).
    axes : (2,), optional
       the axes where the elliptical base is defined, will not be used when `v` is of shape (2, 3).
       Defaults to the :math:`xy` plane.
    center : (3,), optional
        the center of the cylinder, defaults to the origin.

    Examples
    --------
    >>> shape = EllipticalCylinder(2, 3, axes=(1, 2))
    >>> shape.within([1.4, 0, 0])
    True
    >>> shape.within([1.4, 1.1, 0])
    False
    >>> shape.within([1.4, 0, 1.1])
    False
    """

    __slots__ = ("_v", "_nh", "_iv", "_h")

    def __init__(self, v, h: float, axes=(0, 1), center=None):
        super().__init__(center)

        v = _a.asarrayd(v)
        vv = _a.zerosd([2, 3])
        if v.size <= 2:
            vv[[0, 1], axes] = v
        elif v.size == 6:
            vv[:, :] = v
        else:
            raise ValueError(
                f"{self.__class__.__name__} expected 'v' to be of size (1,), (2,) or (2, 3), got {v.shape}"
            )

        # If the vectors are not orthogonal, orthogonalize them and issue a warning
        vv_ortho = np.fabs(vv @ vv.T - np.diag(fnorm2(vv)))
        if vv_ortho.sum() > 1e-9:
            warn(
                f"{self.__class__.__name__ } principal vectors are not orthogonal. "
                "sisl orthogonalizes the vectors (retaining 1st vector)!"
            )

        vv[1] = orthogonalize(vv[0], vv[1])

        # create the inverse
        vvv = _a.empty([3, 3])
        vvv[:2] = vv
        vvv[2] = np.cross(vv[0], vv[1])
        vvv[2] = h * vvv[2] / fnorm(vvv[2])
        ivv = np.linalg.inv(vvv)

        # this is only (2, 3)
        self._v = vv
        # note this is (3, 3)
        self._iv = ivv
        # normal vector, correct length (h)
        self._nh = vvv[2]
        # scalar
        self._h = h

    def copy(self):
        return self.__class__(self.radial_vector, self.height, self.center)

    @property
    def volume(self) -> float:
        """Return the volume of the shape"""
        return pi * np.prod(self.radius) * self.height

    @property
    def height(self) -> float:
        """Height of the cylinder"""
        return self._h

    @property
    def radius(self):
        """Radius of the ellipse base vectors"""
        return fnorm(self._v)

    @property
    def radial_vector(self):
        """The radial vectors"""
        return self._v

    @property
    def height_vector(self):
        """The height vector"""
        return self._nh

    def scale(self, scale: float):
        """Create a new shape with all dimensions scaled according to `scale`

        Parameters
        ----------
        scale : float or (3,)
            scale parameter for each of the ellipse vectors (first two), and for the
            height of the cylinder (last element).
        """
        scale = _a.asarrayd(scale)
        if scale.size == 3:
            v = self._v * scale[:2].reshape(-1, 1)
            h = self._h * scale[2]
        else:
            v = self._v * scale
            h = self._h * scale
        return self.__class__(v, h, self.center)

    def expand(self, radius):
        """Expand elliptical cylinder by a constant value along each vector and height

        Parameters
        ----------
        radius : float or (3,)
           the extension in Ang per elliptical vector and height
        """
        radius = _a.asarrayd(radius)
        if radius.size == 1:
            v0 = expand(self._v[0], radius)
            v1 = expand(self._v[1], radius)
            h = self.height + radius
        elif radius.size == 3:
            v0 = expand(self._v[0], radius[0])
            v1 = expand(self._v[1], radius[1])
            h = self.height + radius[2]
        else:
            raise ValueError(
                f"{self.__class__.__name__}.expand requires the radius to be either (1,) or (3,)"
            )
        return self.__class__([v0, v1], h, self.center)

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
        rtol :
           relative tolerance for boundaries.
        """
        other = _a.asarrayd(other)
        other.shape = (-1, 3)

        # First check
        tmp = np.dot(other - self.center, self._iv)

        # Get indices where we should do the more
        # expensive exact check of being inside shape
        # I.e. this reduces the search space to the box
        return indices_in_cylinder(tmp, 1.0 + rtol, 1.0 + rtol)

    @deprecation(
        "toSphere is deprecated, use shape.to.Sphere(...) instead.", "0.15", "0.17"
    )
    def toSphere(self):
        """Convert to a sphere"""
        from .ellipsoid import Sphere

        # figure out the distance from the center to the edge (along longest radius)
        h = self.height / 2
        r = self.radius.max()
        # now figure out the actual distance
        r = (r**2 + h**2) ** 0.5
        # Rescale each vector
        return Sphere(r, self.center.copy())

    @deprecation(
        "toCuboid is deprecated, use shape.to.Cuboid(...) instead.", "0.15", "0.17"
    )
    def toCuboid(self):
        """Return a cuboid with side lengths equal to the diameter of each ellipsoid vectors"""
        from .prism4 import Cuboid

        return Cuboid([self._v[0], self._v[1], self._nh], self.center)


to_dispatch = EllipticalCylinder.to


class EllipticalCylinderToSphere(ShapeToDispatch):
    def dispatch(self, *args, **kwargs):
        from .ellipsoid import Sphere

        shape = self._get_object()
        # figure out the distance from the center to the edge (along longest radius)
        h = shape.height / 2
        r = shape.radius.max()
        # now figure out the actual distance
        r = (r**2 + h**2) ** 0.5
        # Rescale each vector
        return Sphere(r, shape.center.copy())


to_dispatch.register("Sphere", EllipticalCylinderToSphere)


class EllipticalCylinderToCuboid(ShapeToDispatch):
    def dispatch(self, *args, **kwargs):
        from .prism4 import Cuboid

        shape = self._get_object()
        return Cuboid([shape._v[0], shape._v[1], shape._nh], shape.center)


to_dispatch.register("Cuboid", EllipticalCylinderToCuboid)

del to_dispatch
