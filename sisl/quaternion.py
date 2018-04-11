from __future__ import print_function, division

import math as m
import numpy as np

__all__ = ['Quaternion']


class Quaternion(object):
    """
    Quaternion object to enable easy rotational quantities.
    """

    def __init__(self, angle=0., v=None, rad=False):
        """ Create quaternion object with angle and vector """
        if rad:
            half = angle / 2
        else:
            half = angle / 360 * m.pi
        self._v = np.empty([4], np.float64)
        self._v[0] = m.cos(half)
        if v is None:
            v = np.array([1, 0, 0], np.float64)
        self._v[1:] = np.array(v[:3], np.float64) * m.sin(half)

    def copy(self):
        """ Return deepcopy of itself """
        q = Quaternion()
        q._v = np.copy(self._v)
        return q

    def conj(self):
        """ Returns the conjugate of it-self """
        q = self.copy()
        q._v[1:] *= -1
        return q

    def norm(self):
        """ Returns the norm of this quaternion """
        return np.sqrt(np.sum(self._v**2))

    @property
    def degree(self):
        """ Returns the angle associated with this quaternion (in degree)"""
        return m.acos(self._v[0]) * 360. / m.pi

    @property
    def radian(self):
        """ Returns the angle associated with this quaternion (in radians)"""
        return m.acos(self._v[0]) * 2.

    angle = radian

    def rotate(self, v):
        """ Rotates 3-dimensional vector ``v`` with the associated quaternion """
        if len(v.shape) == 1:
            q = self.copy()
            q._v[0] = 1.
            q._v[1:] = v[:]
            q = self * q * self.conj()
            return q._v[1:]
        # We have a matrix of vectors
        # Instead of doing it per-vector, we do it in chunks
        v1 = np.copy(self._v)
        v2 = np.copy(self.conj()._v)
        s = np.copy(v.shape)
        # First "flatten"
        v.shape = (-1, 3)
        f = np.empty([4, v.shape[0]], v.dtype)
        f[0, :] = v1[0] - v1[1] * v[:, 0] - v1[2] * v[:, 1] - v1[3] * v[:, 2]
        f[1, :] = v1[0] * v[:, 0] + v1[1] + v1[2] * v[:, 2] - v1[3] * v[:, 1]
        f[2, :] = v1[0] * v[:, 1] - v1[1] * v[:, 2] + v1[2] + v1[3] * v[:, 0]
        f[3, :] = v1[0] * v[:, 2] + v1[1] * v[:, 1] - v1[2] * v[:, 0] + v1[3]
        # Create actual rotated array
        nv = np.empty(v.shape, v.dtype)
        nv[:, 0] = f[0, :] * v2[1] + f[1, :] * \
            v2[0] + f[2, :] * v2[3] - f[3, :] * v2[2]
        nv[:, 1] = f[0, :] * v2[2] - f[1, :] * \
            v2[3] + f[2, :] * v2[0] + f[3, :] * v2[1]
        nv[:, 2] = f[0, :] * v2[3] + f[1, :] * \
            v2[2] - f[2, :] * v2[1] + f[3, :] * v2[0]
        del f
        # re-create shape
        nv.shape = s
        return nv

    def __eq__(self, other):
        """ Returns whether two Quaternions are equal """
        return np.allclose(self._v, other._v)

    def __neg__(self):
        """ Returns the negative quaternion """
        q = self.copy()
        q._v = -q._v
        return q

    def __add__(self, other):
        """ Returns the added quantity """
        q = self.copy()
        if isinstance(other, Quaternion):
            q._v += other._v
        else:
            q._v += other
        return q

    def __sub__(self, other):
        """ Returns the subtracted quantity """
        q = self.copy()
        if isinstance(other, Quaternion):
            q._v -= other._v
        else:
            q._v -= other
        return q

    def __mul__(self, other):
        """ Multiplies with another instance or scalar """
        q = self.copy()
        if isinstance(other, Quaternion):
            v1 = np.copy(self._v)
            v2 = other._v
            q._v[0] = v1[0] * v2[0] - v1[1] * \
                v2[1] - v1[2] * v2[2] - v1[3] * v2[3]
            q._v[1] = v1[0] * v2[1] + v1[1] * \
                v2[0] + v1[2] * v2[3] - v1[3] * v2[2]
            q._v[2] = v1[0] * v2[2] - v1[1] * \
                v2[3] + v1[2] * v2[0] + v1[3] * v2[1]
            q._v[3] = v1[0] * v2[3] + v1[1] * \
                v2[2] - v1[2] * v2[1] + v1[3] * v2[0]
        else:
            q._v *= other
        return q

    def __div__(self, other):
        """ Divides with a scalar """
        if isinstance(other, Quaternion):
            raise ValueError("Do not know how to divide a quaternion " +
                             "with a quaternion.")
        return self * (1. / other)
    __truediv__ = __div__

    def __iadd__(self, other):
        """ In-place addition """
        if isinstance(other, Quaternion):
            self._v += other._v
        else:
            self._v += other
        return self

    def __isub__(self, other):
        """ In-place subtraction """
        if isinstance(other, Quaternion):
            self._v -= other._v
        else:
            self._v -= other
        return self

    # The in-place operators
    def __imul__(self, other):
        """ In-place multiplication """
        if isinstance(other, Quaternion):
            v1 = np.copy(self._v)
            v2 = other._v
            self._v[0] = v1[0] * v2[0] - v1[1] * \
                v2[1] - v1[2] * v2[2] - v1[3] * v2[3]
            self._v[1] = v1[0] * v2[1] + v1[1] * \
                v2[0] + v1[2] * v2[3] - v1[3] * v2[2]
            self._v[2] = v1[0] * v2[2] - v1[1] * \
                v2[3] + v1[2] * v2[0] + v1[3] * v2[1]
            self._v[3] = v1[0] * v2[3] + v1[1] * \
                v2[2] - v1[2] * v2[1] + v1[3] * v2[0]
        else:
            self._v *= other
        return self

    def __idiv__(self, other):
        """ In-place division """
        if isinstance(other, Quaternion):
            raise ValueError("Do not know how to divide a quaternion " +
                             "with a quaternion.")
        # use imul
        self._v /= other
        return self
    __itruediv__ = __idiv__
