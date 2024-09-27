# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.linalg as la

from .base import eigh, svd

__all__ = ["signsqrt", "sqrth", "invsqrth", "lowdin"]


def signsqrt(a):
    r"""Calculate the sqrt of the elements `a` by retaining the sign.

    This only influences negative values in `a` by returning ``-abs(a)**0.5``

    Parameters
    ----------
    a : array_like
       real array to be processed
    """
    return np.sign(a) * np.sqrt(np.fabs(a))


def sqrth(a, overwrite_a=False):
    r""":math:`\mathbf H^{1/2}` for a Hermitian matrix `a`.

    This method is not exactly equivalent to `scipy.linalg.sqrtm` since the latter
    is general, whereas this one is for Hermitian matrices.

    In the Hermitian case the :math:`\sqrt{\mbox{}}` of the eigenvalues are a bit
    more precise. E.g. when comparing ``H12 @ H12`` vs. ``H12 @ H12.T.conj()``. The latter is what
    we need.
    """
    eig, ev = eigh(a, overwrite_a=overwrite_a)
    eig = np.emath.sqrt(eig)
    return (ev * eig) @ ev.conj().T


def invsqrth(a, overwrite_a=False):
    """Calculate the inverse sqrt of the Hermitian matrix `a`

    We do this by using eigh and taking the sqrt of the eigenvalues.

    This yields a slightly better value compared to `scipy.linalg.sqrtm`
    when comparing ``H12 @ H12`` vs. ``H12 @ H12.T.conj()``. The latter is what
    we need.
    """
    eig, ev = eigh(a, overwrite_a=overwrite_a)
    eig = np.emath.sqrt(eig)
    np.divide(1, eig, where=(eig != 0), out=eig)
    return (ev * eig) @ ev.conj().T


def lowdin(
    a,
    b=None,
    overwrite_a: bool = False,
    driver: Literal["eigh", "gesdd", "gesvd", "schur"] = "eigh",
):
    r"""Calculate the Lowdin transformation matrix, optionally convert the matrix `b` into the orthogonal basis

    Convert the matrix `b` in the basis `a` into an orthogonal basis using the Lowdin transformation

    .. math::

       \mathbf B' = \mathbf A^{-1/2} \mathbf B \mathbf A^{-1/2}

    Note, this method assumes `a` to be Hermitian.

    Parameters
    ----------
    a : array_like
       basis matrix to convert to the Lowdin basis
    b : array_like
       matrix to convert, if not provided as an argument, :math:`\mathbf A^{-1/2}` is
       returned.
    overwrite_a :
        specificy whether `a` can be altered in the call.
    driver :
        which driver to use for calculating the Lowdin transformed `a` matrix.
        When `a` is a matrix with rank ``len(a)`` with algebraic multiplicity
        :math:`\mu_A(1)` equal to the rank, then the SVD method can be used
        to construct the Lowdin transformation.
    """
    if driver == "eigh":
        a12 = invsqrth(a, overwrite_a=overwrite_a)
    elif driver.startswith("ges"):
        U, _, Vh = svd(
            a, compute_uv=True, overwrite_a=overwrite_a, lapack_driver=driver
        )
        a12 = U @ Vh
        del U, _, Vh
    elif driver == "schur":
        a12 = la.sqrtm(a)
    else:
        raise ValueError(f"lowdin: got unknown driver argument '{driver}'")

    if b is None:
        return a12
    return a12 @ b @ a12
