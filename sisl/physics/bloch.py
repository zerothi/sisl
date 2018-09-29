r"""Bloch's theorem
===================

.. module:: sisl.physics.bloch
   :noindex:

Bloch's theorem is a very powerful proceduce that enables one to utilize
the periodicity of a given direction to describe the complete system.


.. autosummary::
   :toctree:

   Bloch

"""
from __future__ import print_function, division

from itertools import product
import numpy as np
from numpy import pi, exp, add

from sisl._help import dtype_real_to_complex
import sisl._array as _a
from sisl._array import aranged


__all__ = ['Bloch']


class Bloch(object):
    r""" Bloch's theorem object containing unfolding factors and unfolding algorithms

    Parameters
    ----------
    bloch : (3,) int
       Bloch repetitions along each direction
    tile : bool, optional
       for true the unfolding is a tiling, else it is repeating
    """

    def __init__(self, bloch, tile=True):
        """ Create `Bloch` object """
        self._bloch = _a.arrayi(bloch)
        self._bloch = np.where(self._bloch < 1, 1, self._bloch)
        self._is_tile = tile

    def __len__(self):
        """ Return unfolded size """
        return np.prod(self.bloch)

    def __str__(self):
        """ Representation of the Bloch model """
        B = self._bloch
        if self.is_tile:
            return self.__class__.__name__ + '{{tile: [{0}, {1}, {2}]}}'.format(B[0], B[1], B[2])
        return self.__class__.__name__ + '{{repeat: [{0}, {1}, {2}]}}'.format(B[0], B[1], B[2])

    @property
    def bloch(self):
        """ Number of Bloch expansions along each lattice vector """
        return self._bloch

    @property
    def is_tile(self):
        """ Whether the Bloch unfolding will be using the tiling construct """
        return self._is_tile

    @property
    def is_repeat(self):
        """ Whether the Bloch unfolding will be using the repeat construct """
        return not self._is_tile

    def unfold_points(self, k):
        r""" Return a list of k-points to be evaluated for this objects unfolding

        The k-point `k` is with respect to the unfolded geometry.
        The return list of `k` points are the k-points required to be sampled in the
        folded geometry (``this.parent``).

        Parameters
        ----------
        k : (3,) of float
           k-point evaluation corresponding to the unfolded unit-cell

        Returns
        -------
        k_unfold : a list of ``np.prod(self.bloch)`` k-points used for the unfolding
        """
        k = _a.arrayd(k)

        # Create expansion points
        B = self._bloch
        unfold = _a.emptyd([B[2], B[1], B[0], 3])
        # Use B-casting rules (much simpler)
        unfold[:, :, :, 0] = (aranged(B[0]).reshape(1, 1, -1) + k[0]) / B[0]
        unfold[:, :, :, 1] = (aranged(B[1]).reshape(1, -1, 1) + k[1]) / B[1]
        unfold[:, :, :, 2] = (aranged(B[2]).reshape(-1, 1, 1) + k[2]) / B[2]
        # Back-transform shape
        unfold.shape = (-1, 3)
        return unfold

    def __call__(self, func, k, *args, **kwargs):
        """ Return a functions return values as the Bloch unfolded equivalent according to this object

        Calling the `Bloch` object is a shorthand for the manual use of the `Bloch.unfold_points` and `Bloch.unfold`
        methods.

        This call structure is a shorthand for:

        >>> bloch = Bloch([2, 1, 2])
        >>> k_unfold = bloch.unfold_points([0] * 3)
        >>> M = [func(*args, k=k) for k in k_unfold]
        >>> bloch.unfold(M, k_unfold)

        Notes
        -----
        The function passed *must* have a keyword argument ``k``.

        Parameters
        ----------
        func : callable
           method called which returns a matrix.
        k : (3, ) of float
           k-point to be unfolded
        *args : list
           arguments passed directly to `func`
        **kwargs: dict
           keyword arguments passed directly to `func`

        Returns
        -------
        M : unfolded Bloch matrix
        """
        K_unfold = self.unfold_points(k)
        return self.unfold([func(*args, k=K, **kwargs) for K in K_unfold], K_unfold)

    def unfold(self, M, k_unfold):
        r""" Unfold the matrix list of matrices `M` into a corresponding k-point (unfolding k-points are `k_unfold`)

        Parameters
        ----------
        M : list of numpy arrays
            matrices used for unfolding
        k_unfold : (*, 3) of float
            unfolding k-points as returned by `Bloch.unfold_points`

        Returns
        -------
        M_unfold : unfolded matrix of size ``M[0].shape * k_unfold.shape[0] ** 2``
        """
        Bi, Bj, Bk = self.bloch
        if self.is_tile:
            shape = (Bk, Bj, Bi, M[0].shape[0], Bk, Bj, Bi, M[0].shape[1])
            Mshape = (M[0].shape[0], 1, 1, 1, M[0].shape[1])
        else:
            raise NotImplementedError(self.__class__.__name__ + '.unfold currently does not implement repeating!')

        # Allocate the unfolded matrix
        Mu = np.zeros(shape, dtype=dtype_real_to_complex(M[0].dtype))

        # Use B-casting rules (much simpler)
        pi2 = 2 * pi

        # Perform unfolding
        N = len(self)
        w = 1 / N
        if Bi == 1:
            if Bj == 1:
                K = aranged(Bk).reshape(1, Bk, 1, 1, 1)
                for T in range(N):
                    m = M[T].reshape(Mshape) * w
                    kjpi2 = 1j * pi2 * k_unfold[T, 2]
                    for k in range(Bk):
                        add(Mu[k, 0, 0], m * exp(kjpi2 * (K - k)), out=Mu[k, 0, 0])

            elif Bk == 1:
                J = aranged(Bj).reshape(1, 1, Bj, 1, 1)
                for T in range(N):
                    m = M[T].reshape(Mshape) * w
                    kjpi2 = 1j * pi2 * k_unfold[T, 1]
                    for j in range(Bj):
                        add(Mu[0, j, 0], m * exp(kjpi2 * (J - j)), out=Mu[0, j, 0])

            else:
                J = aranged(Bj).reshape(1, 1, Bj, 1, 1)
                K = aranged(Bk).reshape(1, Bk, 1, 1, 1)
                for T in range(N):
                    m = M[T].reshape(Mshape) * w
                    kpi2 = pi2 * k_unfold[T, :]
                    for k in range(Bk):
                        Kk = (K - k) * kpi2[2]
                        for j in range(Bj):
                            add(Mu[k, j, 0], m * exp(1j * (Kk + kpi2[1] * (J - j))), out=Mu[k, j, 0])
        elif Bj == 1:
            if Bk == 1:
                I = aranged(Bi).reshape(1, 1, 1, Bi, 1)
                for T in range(N):
                    m = M[T].reshape(Mshape) * w
                    kjpi2 = 1j * pi2 * k_unfold[T, 0]
                    for i in range(Bi):
                        add(Mu[0, 0, i], m * exp(kjpi2 * (I - i)), out=Mu[0, 0, i])

            else:
                I = aranged(Bi).reshape(1, 1, 1, Bi, 1)
                K = aranged(Bk).reshape(1, Bk, 1, 1, 1)
                for T in range(N):
                    m = M[T].reshape(Mshape) * w
                    kpi2 = pi2 * k_unfold[T, :]
                    for k in range(Bk):
                        Kk = (K - k) * kpi2[2]
                        for i in range(Bi):
                            add(Mu[k, 0, i], m * exp(1j * (Kk + kpi2[0] * (I - i))), out=Mu[k, 0, i])

        elif Bk == 1:
            I = aranged(Bi).reshape(1, 1, 1, Bi, 1)
            J = aranged(Bj).reshape(1, 1, Bj, 1, 1)
            for T in range(N):
                m = M[T].reshape(Mshape) * w
                kpi2 = pi2 * k_unfold[T, :]
                for j in range(Bj):
                    Jj = (J - j) * kpi2[1]
                    for i in range(Bi):
                        add(Mu[0, j, i], m * exp(1j * (Jj + kpi2[0] * (I - i))), out=Mu[0, j, i])

        else:
            I = aranged(Bi).reshape(1, 1, 1, Bi, 1)
            J = aranged(Bj).reshape(1, 1, Bj, 1, 1)
            K = aranged(Bk).reshape(1, Bk, 1, 1, 1)
            for T in range(N):
                m = M[T].reshape(Mshape) * w
                kpi2 = pi2 * k_unfold[T, :]
                for k in range(Bk):
                    Kk = (K - k) * kpi2[2]
                    for j in range(Bj):
                        KkJj = Kk + (J - j) * kpi2[1]
                        for i in range(Bi):
                            # Calculate phases and add for all expansions
                            add(Mu[k, j, i], m * exp(1j * (KkJj + kpi2[0] * (I - i))), out=Mu[k, j, i])

        return Mu.reshape(N * M[0].shape[0], N * M[0].shape[1])
