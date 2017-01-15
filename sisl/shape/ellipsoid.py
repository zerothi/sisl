""" 
Implement a set of simple shapes that
"""

from numbers import Real
from math import pi
import numpy as np

from .shape import Shape


__all__ = ['Ellipsoid', 'Spheroid', 'Sphere']


class Ellipsoid(Shape):
    """ A 3D Ellipsoid shape

    Parameters
    ----------
    x : `float` 
       the radius along x-direction
    y : `float` 
       the radius along y-direction
    z : `int` 
       the radius along z-direction
    """

    def __init__(self, x, y, z, center=None):
        super(Ellipsoid, self).__init__(center)
        self._radius = np.array([x, y, z], np.float64)

    @property
    def radius(self):
        """ Return the radius of the Ellipsoid """
        return self._radius

    @property
    def displacement(self):
        """ Return the displacement vector of the Ellipsoid """
        return self.radius * 0.5 ** 0.5 * 2

    @property
    def volume(self):
        """ Return the volume of the shape """
        r = self.radius
        return 4. / 3. * pi * r[0] * r[1] * r[2]

    def expand(self, length):
        """ Return a new shape with a larger corresponding to `length` """
        r = self.radius + length
        return self(*r, center=self.center)

    def set_center(self, center):
        """ Change the center of the object """
        self.__init__(*self.radius, center=center)

    def within(self, other, return_sub=False):
        """ Return whether the points are within the shape """

        if isinstance(other, (list, tuple)):
            other = np.asarray(other, np.float64)

        if isinstance(other, np.ndarray):
            # Figure out if th
            other.shape = (-1, 3)

            # First check
            fabs = np.fabs
            landr = np.logical_and.reduce
            center = self.center
            radius = self.radius
            tmp = other - center[None, :]
            within = landr(fabs(tmp[:, :]) <= radius[0], axis=1)

            # Now only check exactly on those that are possible
            # candidates
            tmp = tmp[within, :]
            wtmp = (tmp[:, 0] / radius[0]) ** 2 + \
                   (tmp[:, 1] / radius[1]) ** 2 + \
                   (tmp[:, 2] / radius[2]) ** 2 <= 1.

            # Set values
            within[within] = wtmp
            if return_sub:
                tmp = tmp[wtmp, :] + self.center[None, :]

            if return_sub:
                return within, tmp
            return within

    def iwithin(self, other, return_sub=False):
        """ Return indices of the points that are within the shape """

        if isinstance(other, (list, tuple)):
            other = np.asarray(other, np.float64)

        if not isinstance(other, np.ndarray):
            raise ValueError('Could not index the other list')

        other.shape = (-1, 3)

        # First check
        where = np.where
        fabs = np.fabs
        landr = np.logical_and.reduce
        center = self.center
        radius = self.radius[0]
        tmp = other[:, :] - center[None, :]

        # Get indices where we should do the more
        # expensive exact check of being inside shape
        within = where(landr(fabs(tmp[:, :]) <= radius, axis=1))[0]

        # Now only check exactly on those that are possible candidates
        tmp = tmp[within, :]
        wtmp = where(tmp[:, 0] ** 2 + tmp[:, 1] ** 2 + tmp[:, 2] ** 2
                     <= radius * radius)[0]

        within = within[wtmp]

        if return_sub:
            return within, other[within, :]
        return within


class Spheroid(Ellipsoid):
    """ A 3D spheroid shape

    Parameters
    ----------
    a : `float` 
       the first spheroid axis radius
    b : `float` 
       the second spheroid axis radius
    axis : `int` 
       the symmetry axis of the Spheroid
    """

    def __init__(self, a, b, axis=2, center=None):
        if axis == 2: # z-axis
            super(Spheroid, self).__init__(a, a, b, center)
        elif axis == 1: # y-axis
            super(Spheroid, self).__init__(a, b, a, center)
        elif axis == 0: # x-axis
            super(Spheroid, self).__init__(b, a, a, center)
        else:
            raise ValueError('Symmetry axis of Spheroid must be `0 <= axis < 3`')

    def set_center(self, center):
        """ Change the center of the object """
        super(Spheroid, self).__init__(*self.radius, center=center)


class Sphere(Spheroid):
    """ A sphere """

    def __init__(self, radius, center=None):
        super(Sphere, self).__init__(radius, radius, center=center)

    def set_center(self, center):
        """ Change the center of the object """
        self.__init__(self.radius[0], center=center)

    def within(self, other, return_sub=False):
        """ Return whether the points are within the shape """

        if isinstance(other, (list, tuple)):
            other = np.asarray(other, np.float64)

        if isinstance(other, np.ndarray):
            # Figure out if th
            other.shape = (-1, 3)

            # First check
            where = np.where
            fabs = np.fabs
            landr = np.logical_and.reduce
            center = self.center
            radius = self.radius[0]
            tmp = other[:, :] - center[None, :]

            within = landr(fabs(tmp[:, :]) <= radius, axis=1)

            # Now only check exactly on those that are possible
            # candidates
            tmp = tmp[within, :]
            wtmp = tmp[:, 0] ** 2 + tmp[:, 1] ** 2 + tmp[:, 2] ** 2 <= radius ** 2

            within[within] = wtmp
            if return_sub:
                tmp = tmp[wtmp, :] + self.center[None, :]

            if return_sub:
                return within, tmp
            return within

    def iwithin(self, other, return_sub=False):
        """ Return indices of the points that are within the shape """

        if isinstance(other, (list, tuple)):
            other = np.asarray(other, np.float64)

        if not isinstance(other, np.ndarray):
            raise ValueError('Could not index the other list')

        other.shape = (-1, 3)

        # First check
        where = np.where
        fabs = np.fabs
        landr = np.logical_and.reduce
        center = self.center
        radius = self.radius[0]
        tmp = other[:, :] - center[None, :]

        within = where(landr(fabs(tmp[:, :]) <= radius, axis=1))[0]

        # Now only check exactly on those that are possible candidates
        wtmp = where((tmp[within, :] ** 2).sum(1) <= radius ** 2)[0]

        within = within[wtmp]

        if return_sub:
            return within, other[within, :]
        return within

    def __repr__(self):
        s = self.__class__.__name__ + ' c({1:.2f} {2:.2f} {3:.2f}) r={0:.2f}'.format(self.radius[0], *self.center)

        return s
