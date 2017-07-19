""" 
Implement a set of simple shapes that
"""

from numbers import Real
import numpy as np
import numpy.linalg as la

from .shape import Shape


__all__ = ['Cuboid', 'Cube']


class Cuboid(Shape):
    """ A cuboid/rectangular prism (P4) with equi-opposite faces """

    def __init__(self, edge_length, center=None):
        super(Cuboid, self).__init__(center)
        if isinstance(edge_length, Real):
            # now this is really a Cube...
            edge_length = [edge_length] * 3
        self._edge_length = np.copy(edge_length, np.float64)

    @property
    def displacement(self):
        """ Return the displacement vector of the Cuboid """
        return self.edge_length

    @property
    def volume(self):
        """ Return the edge-length of the Cuboid """
        return np.product(self.edge_length)

    @property
    def origo(self):
        """ Return the origin of the Cuboid (lower-left corner) """
        return self.center - self.edge_length / 2

    def set_center(self, center):
        """ Re-setting the center can sometimes be necessary """
        self.__init__(self.edge_length, center)

    def set_origo(self, origo):
        """ Re-setting the origo can sometimes be necessary """
        self.__init__(self.edge_length, origo + self.edge_length * .5)

    @property
    def edge_length(self):
        """ Return the edge-length of the Cuboid """
        return self._edge_length

    def expand(self, length):
        """ Return a new shape with a larger corresponding to `length` """
        return self(self.edge_length + length, center=self.center)

    def within(self, other):
        """ Return a `True/False` value of whether the `other` object is contained in this shape

        Parameters
        ----------
        other : (`numpy.ndarray`, list, tuple)
           the object that is checked for containment
        """

        # Retrieve indices
        idx = self.iwithin(other)

        if isinstance(other, (list, tuple)):
            other = np.asarray(other, np.float64)

        if isinstance(other, np.ndarray):
            other.shape = (-1, 3)

            full = np.empty(len(other), dtype='bool')
            full[:] = False
            full[idx] = True
            return full

        raise NotImplementedError('within could not determine the extend of the `other` object')

    def iwithin(self, other):
        """ Return indices of the `other` object which are contained in the shape

        Parameters
        ----------
        other : (`numpy.ndarray`, list, tuple)
           the object that is checked for containment
        """

        if isinstance(other, (list, tuple)):
            other = np.asarray(other, np.float64)

        if not isinstance(other, np.ndarray):
            raise ValueError('Could not index the other list')

        other.shape = (-1, 3)

        # Offset origo
        tmp = other[:, :] - self.origo[None, :]
        el = self.edge_length[:]

        voxel = np.diagflat(self.edge_length)
        # First reject those that are definitely not inside
        land = np.logical_and
        ix = np.where(land(land(land(0 <= tmp[:, 0],
                                     tmp[:, 0] <= el[0]),
                                land(0 <= tmp[:, 1],
                                     tmp[:, 1] <= el[1])),
                           land(0 <= tmp[:, 2],
                                tmp[:, 2] <= el[2])))[0]
        return ix
        # This below is for a skewed box
        within = la.solve(voxel, tmp[ix, :].T).T

        # Reduce to check if they are within
        within = ix[land.reduce(land(0. <= within, within <=1), axis=1)]

        return within


class Cube(Cuboid):
    """ A cuboid/rectangular prism (P4) with all-equi faces """

    def __init__(self, edge_length, center=None):
        super(Cube, self).__init__(edge_length, center)
