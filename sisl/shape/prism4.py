# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
from numpy import dot

import sisl._array as _a
from sisl._internal import set_module
from sisl.linalg import inv
from sisl.utils.mathematics import fnorm, expand
from sisl._math_small import dot3, cross3
from sisl._indices import indices_gt_le

from .base import PureShape, ShapeToDispatcher


__all__ = ['Cuboid', 'Cube']


@set_module("sisl.shape")
class Cuboid(PureShape):
    """ A cuboid/rectangular prism (P4)

    Parameters
    ----------
    v : float or (3,) or (3, 3)
       vectors describing the cuboid, if only 3 the cuboid will be
       along the Euclidean vectors.
    center : (3,), optional
       the center of the cuboid. Defaults to the origin.
       Not allowed as argument if `origin` is passed.
    origin : (3,), optional
       the offset for the cuboid. The center will be equal to ``v.sum(0) + origin``.
       Not allowed as argument if `center` is passed.

    Examples
    --------
    >>> shape = Cuboid([2, 2.2, 2])
    >>> shape.within([0, 2.1, 0])
    False
    >>> shape.within([0, 1.1, 0])
    True
    """
    __slots__ = ('_v', '_iv')

    # Define a dispatcher for converting Shapes
    #  Cuboid().to.ellipsoid() will convert to an sisl.shape.Ellipsoid object
    to = PureShape.to.copy()

    def __init__(self, v, center=None, origin=None):

        v = _a.asarrayd(v)
        if v.size == 1:
            self._v = np.identity(3) * v # a "Euclidean" cube
        elif v.size == 3:
            self._v = np.diag(v.ravel()) # a "Euclidean" rectangle
        elif v.size == 9:
            self._v = v.reshape(3, 3).astype(np.float64)
        else:
            raise ValueError(f"{self.__class__.__name__} requires initialization with 3 vectors defining the cuboid")

        if center is not None and origin is not None:
            raise ValueError(f"{self.__class__.__name__} only allows either origin or center argument")
        elif origin is not None:
            center = self._v.sum(0) / 2 + origin

        # initialize the center
        super().__init__(center)

        # Create the reciprocal cell
        self._iv = inv(self._v)

    def copy(self):
        return self.__class__(self._v, self.center)

    def __str__(self):
        return self.__class__.__name__ + '{{O({1} {2} {3}), vol: {0}}}'.format(self.volume(), *self.origin)

    def volume(self):
        """ Return volume of Cuboid """
        return abs(dot3(self._v[0, :], cross3(self._v[1, :], self._v[2, :])))

    def scale(self, scale):
        """ Scale the cuboid box size (center is retained)

        Parameters
        ----------
        scale : float or (3,)
            the scale parameter for each of the vectors defining the `Cuboid`
        """
        scale = _a.asarrayd(scale)
        if scale.size == 3:
            scale.shape = (3, 1)
        return self.__class__(self._v * scale, self.center)

    def expand(self, length):
        """ Expand the cuboid by a constant value along side vectors

        Parameters
        ----------
        length : float or (3,)
           the extension in Ang per cuboid vector.
        """
        length = _a.asarrayd(length)
        if length.size == 1:
            v0 = expand(self._v[0, :], length)
            v1 = expand(self._v[1, :], length)
            v2 = expand(self._v[2, :], length)
        elif length.size == 3:
            v0 = expand(self._v[0, :], length[0])
            v1 = expand(self._v[1, :], length[1])
            v2 = expand(self._v[2, :], length[2])
        else:
            raise ValueError(self.__class__.__name__ + '.expand requires the length to be either (1,) or (3,)')
        return self.__class__([v0, v1, v2], self.center)

    def toEllipsoid(self):
        """ Return an ellipsoid that encompass this cuboid """
        from .ellipsoid import Ellipsoid

        # Rescale each vector
        return Ellipsoid(self._v / 2 * 3 ** .5, self.center.copy())

    def toSphere(self):
        """ Return a sphere that encompass this cuboid """
        from .ellipsoid import Sphere

        return Sphere(self.edge_length.max() / 2 * 3 ** .5, self.center.copy())

    def toCuboid(self):
        """ Return a copy of itself """
        return self.copy()

    def within_index(self, other, tol=1.e-8):
        """ Return indices of the `other` object which are contained in the shape

        Parameters
        ----------
        other : array_like
           the object that is checked for containment
        tol : float, optional
           absolute tolerance for boundaries
        """
        other = _a.asarrayd(other).reshape(-1, 3)

        # Offset origin
        tmp = dot(other - self.origin[None, :], self._iv)

        # First reject those that are definitely not inside
        # The proximity is 1e-12 of the inverse cell.
        # So, sadly, the bigger the cell the bigger the tolerance
        # However due to numerics this is probably best anyway
        return indices_gt_le(tmp, -tol, 1. + tol)

    @property
    def origin(self):
        """ Return the origin of the Cuboid (lower-left corner) """
        return self.center - (self._v * 0.5).sum(0)

    @origin.setter
    def origin(self, origin):
        """ Re-setting the origin can sometimes be necessary """
        super().__init__(origin + (self._v * 0.5).sum(0))

    @property
    def edge_length(self):
        """ The lengths of each of the vector that defines the cuboid """
        return fnorm(self._v)


to_dispatch = Cuboid.to


class CuboidToEllipsoid(ShapeToDispatcher):
    def dispatch(self, *args, **kwargs):
        from .ellipsoid import Ellipsoid
        shape = self._obj
        # Rescale each vector
        return Ellipsoid(shape._v / 2 * 3 ** .5, shape.center.copy())

to_dispatch.register("ellipsoid", CuboidToEllipsoid)
to_dispatch.register("Ellipsoid", CuboidToEllipsoid)


class CuboidToSphere(ShapeToDispatcher):
    def dispatch(self, *args, **kwargs):
        from .ellipsoid import Sphere
        shape = self._obj
        # Rescale each vector
        return Sphere(shape.edge_length.max() / 2 * 3 ** .5, shape.center.copy())

to_dispatch.register("sphere", CuboidToSphere)
to_dispatch.register("Sphere", CuboidToSphere)


class CuboidToCuboid(ShapeToDispatcher):
    def dispatch(self, *args, **kwargs):
        return self._obj.copy()

to_dispatch.register("cuboid", CuboidToCuboid)
to_dispatch.register("Cuboid", CuboidToCuboid)


del to_dispatch


@set_module("sisl.shape")
class Cube(Cuboid):
    """ 3D Cube with equal sides

    Equivalent to ``Cuboid([r, r, r])``.

    Parameters
    ----------
    side : float
       side-length of the cube, or vector
    center : (3,), optional
       the center of the cuboid. Defaults to the origin.
       Not allowed as argument if `origin` is passed.
    origin : (3,), optional
       the lower left corner of the cuboid.
       Not allowed as argument if `center` is passed.
    """
    __slots__ = ()

    def __init__(self, side, center=None, origin=None):
        side = _a.asarrayd(side).ravel()[0]
        super().__init__(side, center, origin)
