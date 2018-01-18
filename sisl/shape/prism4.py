from __future__ import print_function, division

from math import pi, sqrt
import numpy as np
from numpy import dot, cross
from numpy import fabs, logical_and

from sisl._help import ensure_array

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
    >>> xyz = [0, 2.1, 0]
    >>> shape = Cuboid([2, 2.2, 2])
    >>> shape.within(xyz)
    array([ True], dtype=bool)
    """
    __slots__ = ('_v', '_iv')

    def __init__(self, v, center=None):
        super(Cuboid, self).__init__(center)
        v = ensure_array(v, np.float64)
        if v.size == 1:
            self._v = np.identity(3) * v # actually a Cube
        elif v.size == 3:
            self._v = np.diag(v.ravel()) # an "Euclidean" rectangle
        elif v.size == 9:
            self._v = v.reshape(3, 3).astype(np.float64)
        else:
            raise ValueError(self.__class__.__name__ + " requires initialization with 3 vectors defining the cuboid")

        # Create the reciprocal cell
        self._iv = np.linalg.inv(self._v).T

    def copy(self):
        return self.__class__(self._v, self.center)

    def __repr__(self):
        return self.__class__.__name__ + '{{O({1} {2} {3}), vol: {0}}}'.format(self.volume(), *self.origo)

    def volume(self):
        """ Return volume of Cuboid """
        return dot(self._v[0, :], cross(self._v[1, :], self._v[2, :]))

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
        scale = ensure_array(scale, np.float64)
        if scale.size == 3:
            scale.shape = (3, 1)
        return self.__class__(self._v * scale, self.center)

    def within_index(self, other):
        """ Return indices of the `other` object which are contained in the shape

        Parameters
        ----------
        other : array_like
           the object that is checked for containment
        """
        other = ensure_array(other, np.float64)
        ndim = other.ndim
        other.shape = (-1, 3)

        # Offset origo
        tmp = dot(other - self.origo[None, :], self._iv.T)

        # First reject those that are definitely not inside
        within = logical_and.reduce(tmp >= 0., axis=1).nonzero()[0]

        tmp = tmp[within, :]
        wtmp = logical_and.reduce(tmp <= 1., axis=1).nonzero()[0]

        return within[wtmp]

    @property
    def origo(self):
        """ Return the origin of the Cuboid (lower-left corner) """
        return self.center - (self._v * 0.5).sum(0)

    def set_origo(self, origo):
        """ Re-setting the origo can sometimes be necessary """
        super(Cuboid, self).__init__(origo + (self._v * 0.5).sum(0))


class Cube(Cuboid):
    """ 3D Cube with equal sides

    Equivalent to ``Cuboid([r, r, r])``.

    Parameters
    ----------
    side : float
       side-length of the cube, or vector
    """

    def __init__(self, side, center=None):
        super(Cube, self).__init__(side, center)
