# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import math as m
from numbers import Integral

import numpy as np
from numpy import (
    arccos,
    arctan2,
    argsort,
    asarray,
    concatenate,
    cos,
    delete,
    divide,
    dot,
    empty,
    sin,
    sqrt,
    square,
    take,
    zeros_like,
)
from scipy.special import sph_harm

from sisl import _array as _a
from sisl._indices import indices_le
from sisl.typing import CartesianAxes

from .misc import direction

__all__ = ["fnorm", "fnorm2", "expand", "orthogonalize"]
__all__ += ["spher2cart", "cart2spher", "spherical_harm"]
__all__ += ["curl", "close"]
__all__ += ["rotation_matrix"]


def fnorm(array, axis=-1):
    r"""Fast calculation of the norm of a vector

    Parameters
    ----------
    array : (..., *)
       the vector/matrix to perform the norm on, norm performed along `axis`
    axis : int, optional
       the axis to take the norm against, default to last axis.
    """
    return sqrt(square(array).sum(axis))


def fnorm2(array, axis=-1):
    r"""Fast calculation of the squared norm of a vector

    Parameters
    ----------
    array : (..., *)
       the vector/matrix to perform the squared norm on, norm performed along `axis`
    axis : int, optional
       the axis to take the norm against, default to last axis.
    """
    return square(array).sum(axis)


def expand(vector, length):
    r"""Expand `vector` by `length` such that the norm of the vector is increased by `length`

    The expansion of the vector can be written as:

    .. math::
        \mathbf v' = \mathbf v + \hat{\mathbf v} l

    Parameters
    ----------
    vector : array_like
        original vector
    length : float
        the length to be added along the vector

    Returns
    -------
    new_vector : the new vector with increased length
    """
    return vector * (1 + length / fnorm(vector))


def orthogonalize(ref, vector):
    r"""Ensure `vector` is orthogonal to `ref`, `vector` must *not* be parallel to `ref`.

    Enable an easy creation of a vector orthogonal to a reference vector. The length of the vector
    is not necessarily preserved (if they are not orthogonal).

    The orthogonalization is performed by:

    .. math::
        \mathbf v_{\perp} = \mathbf v - \hat{\mathbf r}(\hat{\mathbf r} \cdot\mathbf v)

    which is subtracting the projected part from :math:`\mathbf v`.

    Parameters
    ----------
    ref : array_like
       reference vector to make `vector` orthogonal too
    vector : array_like
       the vector to orthogonalize, must have same dimension as `ref`

    Returns
    -------
    ortho : the orthogonalized vector

    Raises
    ------
    ValueError
        if `vector` is parallel to `ref`
    """
    ref = asarray(ref).ravel()
    nr = fnorm(ref)
    vector = asarray(vector).ravel()
    d = dot(ref, vector) / nr
    if abs(1.0 - abs(d) / fnorm(vector)) < 1e-7:
        raise ValueError(
            f"orthogonalize: requires non-parallel vectors to perform an orthogonalization: ref.vector = {d}"
        )
    return vector - ref * d / nr


def spher2cart(r, theta, phi):
    r"""Convert spherical coordinates to cartesian coordinates

    Parameters
    ----------
    r : array_like
       radius
    theta : array_like
       azimuthal angle in the :math:`xy` plane
    phi : array_like
       polar angle from the :math:`z` axis
    """
    r = asarray(r)
    theta = asarray(theta)
    phi = asarray(phi)
    shape = _a.broadcast_shapes(r.shape, theta.shape, phi.shape)
    R = _a.empty(shape + (3,), dtype=r.dtype)
    sphi = sin(phi)
    R[..., 0] = r * cos(theta) * sphi
    R[..., 1] = r * sin(theta) * sphi
    del sphi
    R[..., 2] = r * cos(phi)
    return R


def cart2spher(r, theta: bool = True, cos_phi: bool = False, maxR=None):
    r"""Transfer a vector to spherical coordinates with some possible differences

    Parameters
    ----------
    r : array_like
       the cartesian vectors
    theta :
       if ``True`` also calculate the theta angle and return it
    cos_phi :
       if ``True`` return :math:`\cos(\phi)` rather than :math:`\phi` which may
       be useful in some subsequent mathematical calculations
    maxR : float, optional
       cutoff of the spherical coordinate calculations. If ``None``, calculate
       and return for all.

    Returns
    -------
    idx : numpy.ndarray
       indices of points with ``r <= maxR``
    r : numpy.ndarray
       radius in spherical coordinates, only for `maxR` different from ``None``
    theta : numpy.ndarray
       angle in the :math:`xy` plane from :math:`x` (azimuthal)
       Only calculated if input `theta` is ``True``, otherwise None is returned.
    phi : numpy.ndarray
       If `cos_phi` is ``True`` this is :math:`\cos(\phi)`, otherwise
       :math:`\phi` is returned (the polar angle from the :math:`z` axis)
    """
    r = _a.asarray(r)
    if maxR is None:
        rr = sqrt(square(r).sum(-1))
        if theta:
            theta = arctan2(r[..., 1], r[..., 0])
        else:
            theta = None
        phi = zeros_like(rr)
        idx = rr != 0.0
        divide(r[..., 2], rr, out=phi, where=idx)
        if not cos_phi:
            arccos(phi, out=phi, where=idx)
        return rr, theta, phi

    if r.ndim != 2:
        raise NotImplementedError(
            "cart2spher(..., maxR=1) not allowed for !=2D arrays."
        )

    rr = square(r).sum(-1)
    idx = indices_le(rr, maxR**2)
    r = take(r, idx, 0)
    rr = sqrt(take(rr, idx))
    if theta:
        theta = arctan2(r[..., 1], r[..., 0])
    else:
        theta = None

    phi = zeros_like(rr)
    idx0 = rr != 0.0
    divide(r[..., 2], rr, out=phi, where=idx0)
    if not cos_phi:
        arccos(phi, out=phi, where=idx0)
    return idx, rr, theta, phi


def spherical_harm(m, l, theta, phi):
    r"""Calculate the spherical harmonics using :math:`Y_l^m(\theta, \varphi)` with :math:`\mathbf r\to \{r, \theta, \varphi\}`.

    .. math::
        Y^m_l(\theta,\varphi) = (-1)^m\sqrt{\frac{2l+1}{4\pi} \frac{(l-m)!}{(l+m)!}}
             e^{i m \theta} P^m_l(\cos(\varphi))

    which is the spherical harmonics with the Condon-Shortley phase.

    Parameters
    ----------
    m : int
       order of the spherical harmonics
    l : int
       degree of the spherical harmonics
    theta : array_like
       angle in :math:`xy` plane (azimuthal)
    phi : array_like
       angle from :math:`z` axis (polar)
    """
    # Probably same as:
    # return (-1) ** m * ( (2*l+1)/(4*pi) * factorial(l-m) / factorial(l+m) ) ** 0.5 \
    #    * lpmv(m, l, cos(theta)) * exp(1j * m * phi)
    return sph_harm(m, l, theta, phi) * (-1) ** m


def curl(M, axis=-2, axisv=-1):
    r""" Determine the curl of a matrix `M` where `M` contains the differentiated quantites along `axisv`.

    The curl is calculated as:

    .. math::
       \mathrm{curl} \mathbf M|_x &= \frac{\partial\mathbf M_z}{\partial y} - \frac{\partial\mathbf M_y}{\partial z}
       \\
       \mathrm{curl} \mathbf M|_y &= \frac{\partial\mathbf M_x}{\partial z} - \frac{\partial\mathbf M_z}{\partial x}
       \\
       \mathrm{curl} \mathbf M|_z &= \frac{\partial\mathbf M_y}{\partial x} - \frac{\partial\mathbf M_x}{\partial y}

    where the `axis` are the :math:`\partial x` axis and `axisv` are the :math:`\partial M_x` axis.

    Parameters
    ----------
    M : numpy.ndarray
       matrix to calculate the curl of
    axis : int, optional
       axis that contains the direction vectors, this dimension is removed from the returned curl
    axisv : int, optional
       axis that contains the differentiated vectors

    Returns
    -------
    curl : the curl of the matrix shape of `m` without axis `axis`
    """
    if M.shape[axis] != 3:
        raise ValueError("curl requires 3 vectors to calculate the curl of!")
    elif M.shape[axisv] != 3:
        raise ValueError("curl requires the vectors to have 3 components!")

    # Check that no two axis are used for the same thing
    axis %= M.ndim
    axisv %= M.ndim
    if axis == axisv:
        raise ValueError("curl requires axis and axisv to be different axes")

    # Create lists for correct slices
    slx = [slice(None) for _ in M.shape]
    sly = slx[:]
    slz = slx[:]
    vx = slx[:]
    vy = slx[:]
    vz = slx[:]
    slx[axis] = 0
    sly[axis] = 1
    slz[axis] = 2

    # Prepare the curl elements
    vx[axisv] = 0
    vy[axisv] = 1
    vz[axisv] = 2
    vx.pop(axis)
    vy.pop(axis)
    vz.pop(axis)

    slx = tuple(slx)
    sly = tuple(sly)
    slz = tuple(slz)
    vx = tuple(vx)
    vy = tuple(vy)
    vz = tuple(vz)

    # Create curl by removing the v dimension
    curl = empty(delete(M.shape, axis), dtype=M.dtype)
    curl[vx] = M[sly][vz] - M[slz][vy]
    curl[vy] = M[slz][vx] - M[slx][vz]
    curl[vz] = M[slx][vy] - M[sly][vx]
    return curl


def close(a, b, /, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Equivalent to `numpy.allclose` for scalars"""
    return abs(a - b) <= (atol + rtol * abs(b))


def intersect_and_diff_sets(a, b):
    """See numpy.intersect1d(a, b, assume_unique=True, return_indices=True).
    In addition to that, this function also returns the indices in a and b which
    are *not* in the intersection.
    This saves a bit compared to doing np.delete() afterwards.
    """
    aux = concatenate((a, b))
    aux_sort_indices = argsort(aux, kind="mergesort")
    aux = aux[aux_sort_indices]
    # find elements that are the same in both arrays
    # after sorting we should have at most 2 same elements
    mask = aux[1:] == aux[:-1]
    int1d = aux[:-1][mask]

    aover = aux_sort_indices[:-1][mask]
    bover = aux_sort_indices[1:][mask] - a.size

    nobuddy_lr = concatenate([[True], ~mask, [True]])
    no_buddy = nobuddy_lr[:-1]  # no match left
    no_buddy &= nobuddy_lr[1:]  # no match right

    aonly = aux_sort_indices < a.size
    bonly = ~aonly
    aonly &= no_buddy
    bonly &= no_buddy
    # the below is for some reason slower even though its only two ops
    # aonly &= no_buddy
    # bonly = aonly ^ no_buddy

    aonly = aux_sort_indices[aonly]
    bonly = aux_sort_indices[bonly] - a.size

    return int1d, aover, bover, aonly, bonly


def rotation_matrix(
    x: float, y: float, z: float, rad: bool = False, order: CartesianAxes = "zyx"
) -> np.ndarray:
    r"""Create the rotation matrix defined by the Cartesian rotation angles.

    The rotation matrix will be composed of matrices:

    .. math::

        R(x) &= \begin{bmatrix}
        1 & 0 & 0 \\
        0 & \cos(x) & -\sin(x) \\
        0 & \sin(x) & \cos(x)
        \end{bmatrix}
        \\
        R(y) &= \begin{bmatrix}
        \cos(x) & 0 & \sin(x) \\
        0 & 1 & 0 \\
        -\sin(x) & 0 & \cos(x)
        \end{bmatrix}
        R(z) &= \begin{bmatrix}
        \cos(x) & -\sin(x) & 0 \\
        \sin(x) & \cos(x) & 0 \\
        0 & 0 & 1
        \end{bmatrix}

    Parameters
    ----------
    x :
        angle to rotate around the :math:`x`-axis
    y :
        angle to rotate around the :math:`y`-axis
    z :
        angle to rotate around the :math:`z`-axis
    rad :
        if true, the angles are in radians, otherwise in degrees.
    order :
        specify the order of the rotation matrix.
        Last letter will be the first one to be rotated.
        It defaults to ``zyx``, which results in :math:`R = R(z)R(y)R(x)`.
        If a direction is omitted, it will not be part of the rotation,
        regardless of the angle passed.

    Examples
    --------

    Ensure that rotation and back rotation works. Note the order
    has to be reversed.

    >>> R1 = rotation_matrix(10, 24, 50, order="xyz")
    >>> R2 = rotation_matrix(-10, -24, -50, order="zyx")
    >>> assert np.allclose(R1 @ R2, np.identity(3))

    Rotate around x, then y, then x again (same angle twice)

    >>> R = rotation_matrix(10, 24, 0, order="xyx")
    """
    if rad:

        def cos_sin(a):
            return m.cos(a), m.sin(a)

    else:

        def cos_sin(a):
            a = a / 180 * m.pi
            return m.cos(a), m.sin(a)

    # define rotation matrix
    c, s = cos_sin(x)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    c, s = cos_sin(y)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    c, s = cos_sin(z)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    Rs = (Rx, Ry, Rz)
    R = np.identity(3)
    if isinstance(order, Integral):
        order = [order]
    for dir in order:
        idir = direction(dir)
        R = R @ Rs[idir]

    return R
