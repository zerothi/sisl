# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import math as m

import numpy as np

from sisl._internal import set_module
from sisl.messages import deprecate_argument, deprecation

__all__ = ["Quaternion"]


@set_module("sisl")
class Quaternion:
    r"""
    Quaternion object to enable easy rotational quantities.

    The rotation using quaternions amounts to a rotation around a vector :math:`v`
    by an angle :math:`\phi`.

    The rotation follows the Euler-Rodrigues formula.

    The components of the quaternion will be referenced as:

    .. math::

        a + b\mathbf i + c\mathbf j + d\mathbf k

    The general way this gets translated is the complex matrix:

    .. math::

        \begin{bmatrix}
            a + b i & c + d i\\
            -c + d i & a - b i
        \end{bmatrix}

    Parameters
    ----------
    *args :
        For 1 argument, it is the length 4 vector describing the quaternion.
        For 2 arguments, it is the angle, and vector.
    rad :
        when passing two arguments, for angle and vector, the `rad` value
        decides which unit `angle` is in, for ``rad=True`` it is in radians.
        Otherwise it will be in degrees.

    Examples
    --------

    Construct a quaternion with angle 45°, all 3 are equivalent:

    >>> q1 = Quaternion(45, [1, 2, 3], rad=False)
    >>> q2 = Quaternion(np.pi/4, [1, 2, 3], rad=True)
    >>> q3 = Quaternion([1, 2, 3], 45, rad=False)

    If you have the full quaternion complex number, one can also
    instantiate it directly without having to consider angles:

    >>> q = Quaternion([1, 2, 3, 4])
    """

    def __init__(self, *args, rad: bool = True):
        """Create quaternion object"""
        if len(args) == 1:
            v = args[0]
        elif len(args) == 2:
            angle, v = args
            try:
                if len(v) == 3:
                    # correct order, no need to swap
                    pass
                else:
                    raise TypeError
            except TypeError:
                angle, v = v, angle

            if not rad:
                angle = angle * m.pi / 180
            half = angle / 2

            if len(v) != 3:
                raise ValueError(
                    "Arguments for Quaternion are wrong? "
                    "The vector must have length 3"
                )

            c = m.cos(half)
            s = m.sin(half)

            v = [c, *[i * s for i in v]]
        else:
            raise ValueError(
                f"{type(self).__name__} got wrong number of arguments, "
                "only 1 or 2 arguments allowed."
            )

        if len(v) != 4:
            raise ValueError(
                "Arguments for Quaternion are wrong? " "The vector must have length 4"
            )
        self._v = np.empty([4], np.float64)
        self._v[:] = v

    def __str__(self) -> str:
        """Stringify this quaternion object"""
        angle = self.angle(rad=True)
        v = self.rotation_vector()
        return f"{type(self).__name__}{{θ={angle:.4f}, v={v}}}"

    def __repr__(self) -> str:
        """Representation of this quaternion object"""
        angle = self.angle(rad=True)
        v = self.rotation_vector()
        return f"<{type(self).__name__} θ={angle:.4f}, v={v}>"

    def copy(self) -> Quaternion:
        """Return a copy of itself"""
        return Quaternion(self._v)

    def conj(self) -> Quaternion:
        """Returns the conjugate of the quaternion"""
        return self.__class__(self._v * [1, -1, -1, -1])

    def norm(self) -> float:
        """Returns the norm of this quaternion"""
        return self.norm2() ** 0.5

    def norm2(self) -> float:
        """Returns the squared norm of this quaternion"""
        v = self._v
        return v.dot(v)

    @deprecate_argument(
        "in_rad",
        "rad",
        "argument in_rad has been deprecated in favor of rad, please update your code.",
        "0.16.3",
        "0.18",
    )
    def angle(self, rad: bool = True) -> float:
        """Return the angle of this quaternion, in requested unit"""
        angle = m.acos(self._v[0]) * 2
        if rad:
            return angle
        return angle / m.pi * 180

    @property
    @deprecation("Use .angle(rad=False) instead of .degree", "0.15.3", "0.17")
    def degree(self) -> float:
        """Returns the angle associated with this quaternion (in degrees)"""
        return self.angle(rad=False)

    @property
    @deprecation("Use .angle(rad=True) instead of .radian", "0.15.3", "0.17")
    def radian(self) -> float:
        """Returns the angle associated with this quaternion (in radians)"""
        return self.angle(rad=True)

    def rotation_vector(self) -> np.ndarray:
        """Retrieve the vector the rotation rotates about"""
        angle = self.angle(rad=True)
        sh = m.sin(angle / 2)
        return self._v[1:] / sh

    def rotate(self, v) -> np.ndarray:
        r""" Rotate a vector `v` by this quaternion

        This rotation method uses the *fast* method which can be expressed as:

        .. math::

           \mathbf v' = \mathbf q \mathbf v \mathbf q ^*

        But using a faster approach (more numerically stable), we can use
        this relation:

        .. math::

           \mathbf t  = 2\mathbf q \cross \mathbf v
           \\
           \mathbf v' = \mathbf v + q_w \mathbf t + \mathbf q \cross \mathbf t

        """
        qw = self._v[0]
        q = self._v[1:]

        t = 2 * np.cross(q, v)
        vp = v + qw * t + np.cross(q, t)
        return vp

    def rotation_matrix_su2(self) -> np.ndarray:
        r"""Returns the special unitary group 2 rotation matrix.

        The SU(2) group defines the unitary group encompassing spin
        matrices enables one to rotate the spin-block-matrices.

        The quaternion can directly be translated to the 2x2 matrix
        defining the rotation of the spin-block matrix:

        .. math::

            U = a \mathbf I
            - ib\sigma_x - ic\sigma_y -id\sigma_z

        Notes
        -----
        This implementation uses the *alternate* representation.

        .. math::

            \begin{bmatrix}
                a - d i & -c - b i\\
                c - b i & a + d i
            \end{bmatrix}
        """
        from sisl.physics.spin import Spin

        U = self._v[0] * np.identity(2, dtype=np.complex128) + 1j * (
            -self._v[1] * Spin.X - self._v[2] * Spin.Y - self._v[3] * Spin.Z
        )
        return U

    def rotation_matrix(self) -> np.ndarray:
        """Determine the Cartesian rotation matrix from the quaternion."""
        s = 2 / self.norm2()
        a, b, c, d = self._v
        _, bs, cs, ds = self._v * s
        R = np.array(
            [
                [1 - c * cs - d * ds, b * cs - a * ds, b * ds + a * cs],
                [b * cs + a * ds, 1 - b * bs - d * ds, c * ds - a * bs],
                [b * ds - a * cs, c * ds + a * bs, 1 - b * bs - c * cs],
            ]
        )
        return R

    def __eq__(self, other):
        """Returns whether two Quaternions are equal"""
        return np.allclose(self._v, other._v)

    def __neg__(self):
        """Returns the negative quaternion"""
        q = self.copy()
        q._v = -q._v
        return q

    def __add__(self, other):
        """Returns the added quantity"""
        if isinstance(other, Quaternion):
            q = self.__class__(self._v + other._v)
        else:
            q = self.__class__(self._v + other)
        return q

    def __sub__(self, other):
        """Returns the subtracted quantity"""
        if isinstance(other, Quaternion):
            q = self.__class__(self._v - other._v)
        else:
            q = self.__class__(self._v - other)
        return q

    def __mul__(self, other):
        """Multiplies with another instance or scalar

        Two quaternions multiplied together will result in the Hamilton product.
        """
        if isinstance(other, Quaternion):
            v = np.empty([4], np.float64)
            q1 = self._v
            q2 = other._v
            v[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
            v[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
            v[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
            v[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
        else:
            v = self._v * other
        return self.__class__(v)

    def __div__(self, other):
        """Divides with a scalar"""
        if isinstance(other, Quaternion):
            raise NotImplementedError(
                "Do not know how to divide a quaternion with a quaternion."
            )
        return self * (1.0 / other)

    __truediv__ = __div__

    def __iadd__(self, other):
        """In-place addition"""
        if isinstance(other, Quaternion):
            self._v += other._v
        else:
            self._v += other
        return self

    def __isub__(self, other):
        """In-place subtraction"""
        if isinstance(other, Quaternion):
            self._v -= other._v
        else:
            self._v -= other
        return self

    # The in-place operators
    def __imul__(self, other):
        """In-place multiplication"""
        if isinstance(other, Quaternion):
            q1 = np.copy(self._v)
            q2 = other._v
            self._v[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
            self._v[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
            self._v[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
            self._v[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
        else:
            self._v *= other
        return self

    def __idiv__(self, other):
        """In-place division"""
        if isinstance(other, Quaternion):
            raise NotImplementedError(
                "Do not know how to divide a quaternion with a quaternion."
            )
        # use imul
        self._v /= other
        return self

    __itruediv__ = __idiv__
