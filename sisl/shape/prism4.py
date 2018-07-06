from __future__ import print_function, division

import numpy as np
from numpy import dot

import sisl._array as _a
from sisl.linalg import inv
from sisl.utils.mathematics import fnorm, expand
from sisl._math_small import dot3, cross3
from sisl._indices import indices_gt_le


from .base import PureShape


__all__ = ['Cuboid', 'Cube']


class Cuboid(PureShape):
    """ A cuboid/rectangular prism (P4)

    Parameters
    ----------
    v : float or (3,) or (3, 3)
       vectors describing the cuboid, if only 3 the cuboid will be
       along the Euclidean vectors.
    center : (3,), optional
       the center of the cuboid. Defaults to the origo.

    Examples
    --------
    >>> shape = Cuboid([2, 2.2, 2])
    >>> shape.within([0, 2.1, 0])
    False
    >>> shape.within([0, 1.1, 0])
    True
    """
    __slots__ = ('_v', '_iv')

    def __init__(self, v, center=None):
        super(Cuboid, self).__init__(center)
        v = _a.asarrayd(v)
        if v.size == 1:
            self._v = np.identity(3) * v # a "Euclidean" cube
        elif v.size == 3:
            self._v = np.diag(v.ravel()) # a "Euclidean" rectangle
        elif v.size == 9:
            self._v = v.reshape(3, 3).astype(np.float64)
        else:
            raise ValueError(self.__class__.__name__ + " requires initialization with 3 vectors defining the cuboid")

        # Create the reciprocal cell
        self._iv = inv(self._v)

    def copy(self):
        return self.__class__(self._v, self.center)

    def __str__(self):
        return self.__class__.__name__ + '{{O({1} {2} {3}), vol: {0}}}'.format(self.volume(), *self.origo)

    def volume(self):
        """ Return volume of Cuboid """
        return abs(dot3(self._v[0, :], cross3(self._v[1, :], self._v[2, :])))

    def set_center(self, center):
        """ Re-setting the center can sometimes be necessary """
        super(Cuboid, self).__init__(center)

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

    def within_index(self, other):
        """ Return indices of the `other` object which are contained in the shape

        Parameters
        ----------
        other : array_like
           the object that is checked for containment
        """
        other = _a.asarrayd(other).reshape(-1, 3)

        # Offset origo
        tmp = dot(other - self.origo[None, :], self._iv)
        tol = 1.e-12

        # First reject those that are definitely not inside
        # The proximity is 1e-12 of the inverse cell.
        # So, sadly, the bigger the cell the bigger the tolerance
        # However due to numerics this is probably best anyway
        return indices_gt_le(tmp, -tol, 1. + tol)

    @property
    def origo(self):
        """ Return the origin of the Cuboid (lower-left corner) """
        return self.center - (self._v * 0.5).sum(0)

    def set_origo(self, origo):
        """ Re-setting the origo can sometimes be necessary """
        super(Cuboid, self).__init__(origo + (self._v * 0.5).sum(0))

    @property
    def edge_length(self):
        """ The lengths of each of the vector that defines the cuboid """
        return fnorm(self._v)


class Cube(Cuboid):
    """ 3D Cube with equal sides

    Equivalent to ``Cuboid([r, r, r])``.

    Parameters
    ----------
    side : float
       side-length of the cube, or vector
    """

    def __init__(self, side, center=None):
        side = _a.asarrayd(side).ravel()[0]
        super(Cube, self).__init__(side, center)
