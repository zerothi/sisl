from __future__ import print_function, division

import math as m
import numpy as np

__all__ = ['Quaternion']


class Quaternion(object):
    """
    Quaternion object to enable easy rotational quantities.
    """

    def __init__(self, angle=0., v=None, radians=False):
        """ Create quaternion object with angle and vector """
        if not radians:
            half = angle / 180 * m.pi / 2
        else:
            half = angle / 2
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
        return m.acos(self._v[0]) * 2. / m.pi * 180.

    @property
    def radians(self):
        """ Returns the angle associated with this quaternion (in radians)"""
        return m.acos(self._v[0]) * 2.

    angle = radians

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

    def __eq__(a, b):
        """ Returns whether two Quaternions are equal """
        return np.allclose(a._v, b._v)

    def __neg__(a):
        """ Returns the negative quaternion """
        q = a.copy()
        q._v = -q._v
        return q

    def __add__(a, b):
        """ Returns the added quantity """
        q = a.copy()
        if isinstance(b, Quaternion):
            q._v += b._v
        else:
            q._v += b
        return q

    def __sub__(a, b):
        """ Returns the subtracted quantity """
        q = a.copy()
        if isinstance(b, Quaternion):
            q._v -= b._v
        else:
            q._v -= b
        return q

    def __mul__(a, b):
        """ Multiplies with another instance or scalar """
        q = a.copy()
        if isinstance(b, Quaternion):
            v1 = np.copy(a._v)
            v2 = b._v
            q._v[0] = v1[0] * v2[0] - v1[1] * \
                v2[1] - v1[2] * v2[2] - v1[3] * v2[3]
            q._v[1] = v1[0] * v2[1] + v1[1] * \
                v2[0] + v1[2] * v2[3] - v1[3] * v2[2]
            q._v[2] = v1[0] * v2[2] - v1[1] * \
                v2[3] + v1[2] * v2[0] + v1[3] * v2[1]
            q._v[3] = v1[0] * v2[3] + v1[1] * \
                v2[2] - v1[2] * v2[1] + v1[3] * v2[0]
        else:
            q._v *= b
        return q

    def __div__(a, b):
        """ Divides with a scalar """
        if isinstance(b, Quaternion):
            raise ValueError("Do not know how to divide a quaternion " +
                             "with a quaternion.")
        return a * (1. / b)
    __truediv__ = __div__

    def __iadd__(a, b):
        """ In-place addition """
        if isinstance(b, Quaternion):
            a._v += b._v
        else:
            a._v += b
        return a

    def __isub__(a, b):
        """ In-place subtraction """
        if isinstance(b, Quaternion):
            a._v -= b._v
        else:
            a._v -= b
        return a

    # The in-place operators
    def __imul__(a, b):
        """ In-place multiplication """
        if isinstance(b, Quaternion):
            v1 = np.copy(a._v)
            v2 = b._v
            a._v[0] = v1[0] * v2[0] - v1[1] * \
                v2[1] - v1[2] * v2[2] - v1[3] * v2[3]
            a._v[1] = v1[0] * v2[1] + v1[1] * \
                v2[0] + v1[2] * v2[3] - v1[3] * v2[2]
            a._v[2] = v1[0] * v2[2] - v1[1] * \
                v2[3] + v1[2] * v2[0] + v1[3] * v2[1]
            a._v[3] = v1[0] * v2[3] + v1[1] * \
                v2[2] - v1[2] * v2[1] + v1[3] * v2[0]
        else:
            a._v *= b
        return a

    def __idiv__(a, b):
        """ In-place division """
        if isinstance(b, Quaternion):
            raise ValueError("Do not know how to divide a quaternion " +
                             "with a quaternion.")
        # use imul
        a._v /= b
        return a
    __itruediv__ = __idiv__
