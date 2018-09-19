r"""Bloch's theorem
==================

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
from numpy import pi, exp

from sisl._help import dtype_real_to_complex
import sisl._array as _a


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
        self._is_tile = tile

    def __len__(self):
        """ Return unfolded size """
        return np.prod(self.bloch)

    @property
    def bloch(self):
        return self._bloch

    @property
    def is_tile(self):
        return self._is_tile

    @property
    def is_repeat(self):
        return not self._is_tile

    def unfold_k_points(self, k):
        r""" Return a list of k-points to be evaluated for this objects unfolding

        Parameters
        ----------
        k : (3,) of float
           k-point evaluation

        Returns
        -------
        k_unfold : a list of ``np.prod(self.bloch)`` k-points used for the unfolding
        """
        k = _a.arrayd(k)

        # Create expansion points
        B = self._bloch
        unfold = _a.emptyd([B[2], B[1], B[0], 3])
        # Use B-casting rules (much simpler)
        unfold[:, :, :, 0] = (_a.aranged(B[0]) / B[0]).reshape(1, 1, -1) + k[0]
        unfold[:, :, :, 1] = (_a.aranged(B[1]) / B[1]).reshape(1, -1, 1) + k[1]
        unfold[:, :, :, 2] = (_a.aranged(B[2]) / B[2]).reshape(-1, 1, 1) + k[2]
        # Back-transform shape
        unfold.shape = (-1, 3)
        return unfold

    def unfold(self, M, k_unfold):
        r""" Unfold the matrix list of matrices `M` into a corresponding k-point (unfolding k-points are `k_unfold`)

        Parameters
        ----------
        M : list of numpy arrays
            matrices used for unfolding
        k_unfold : (*, 3) of float
            unfolding k-points as returned by `Bloch.unfold_k`

        Returns
        -------
        M_unfold : unfolded matrix of size ``M[0].shape * k_unfold.shape[0] ** 2``
        """
        B = self.bloch
        if self.is_tile:
            shape = (B[2], B[1], B[0], M[0].shape[0], B[2], B[1], B[0], M[0].shape[1])
            Mshape = (M[0].shape[0], 1, 1, 1, M[0].shape[1])
            kshape = (1, B[2], 1, 1, 1)
            jshape = (1, 1, B[1], 1, 1)
            ishape = (1, 1, 1, B[0], 1)
        else:
            raise NotImplementedError(self.__class__.__name__ + '.unfold currently does not implement repeating!')

        # Allocate the unfolded matrix
        Mu = np.zeros(shape, dtype=dtype_real_to_complex(M[0].dtype))

        # Use B-casting rules (much simpler)
        ijk = (_a.arangei(B[0]), _a.arangei(B[1]), _a.arangei(B[2]))
        jpi2 = 2j * pi

        # Perform unfolding
        N = len(self)
        for T in range(N):
            m = M[T].reshape(Mshape)
            k2jpi = jpi2 * k_unfold[T, :]
            for k in ijk[2]:
                kp = (ijk[2] - k).reshape(kshape)
                for j in ijk[1]:
                    jp = (ijk[1] - j).reshape(jshape)
                    for i in ijk[0]:
                        ip = (ijk[0] - i).reshape(ishape)

                        # Calculate phases and add for all expansions
                        Mu[k, j, i, :, :, :, :, :] += m * exp(k2jpi[2] * kp +
                                                              k2jpi[1] * jp +
                                                              k2jpi[0] * ip)

        return Mu.reshape(N * M[0].shape[0], N * M[0].shape[1]) / N
