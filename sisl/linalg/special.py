# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np

from .base import eigh


__all__ = ['signsqrt', 'sqrth', 'invsqrth', 'lowdin']


def signsqrt(a):
    r""" Calculate the sqrt of the elements `a` by retaining the sign.

    This only influences negative values in `a` by returning ``-abs(a)**0.5``

    Parameters
    ----------
    a : array_like
       real array to be processed
    """
    return np.sign(a) * np.sqrt(np.fabs(a))


def sqrth(a, overwrite_a=False):
    r""":math:`H^{1/2}` for a Hermitian matrix `A`.

    This method is not exactly equivalent to `scipy.linalg.sqrtm` since the latter
    is general, whereas this one is for Hermitian matrices.

    In the Hermitian case the :math:`\sqrt{\mbox{}}` of the eigenvalues are a bit
    more precise. E.g. when comparing H12 @ H12 vs. H12 @ H12.T.conj(). The latter is what
    we need.
    """
    eig, ev = eigh(a, overwrite_a=overwrite_a)
    eig = np.emath.sqrt(eig)
    return (ev * eig) @ ev.conj().T


def invsqrth(a, overwrite_a=False):
    """ Calculate the inverse sqrt of the Hermitian matrix `H`

    We do this by using eigh and taking the sqrt of the eigenvalues.

    This yields a slightly better value compared to scipy.linalg.sqrtm
    when comparing H12 @ H12 vs. H12 @ H12.T.conj(). The latter is what
    we need.
    """
    eig, ev = eigh(a, overwrite_a=overwrite_a)
    eig = np.emath.sqrt(eig)
    np.divide(1, eig, where=(eig != 0), out=eig)
    return (ev * eig) @ ev.conj().T


def lowdin(a, b, overwrite_a=False):
    r""" Convert the matrix `b` in the basis `a` into an orthogonal basis using the Lowdin transformation

    .. math::

       \mathbf B' = \mathbf A^{-1/2} \mathbf B \mathbf A^{-1/2}

    Parameters
    ----------
    a : array_like
       basis matrix to convert to the Lowdin basis
    b : array_like
       matrix to convert
    """
    a12 = invsqrth(a, overwrite_a=overwrite_a)
    return a12 @ b @ a12
