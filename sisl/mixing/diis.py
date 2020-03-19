from functools import reduce
from operator import add, mul
import numpy as np

import sisl._array as _a
from sisl.linalg import solve
from .base import History, Metric
from .linear import LinearMixer


__all__ = ['DIISMixer', 'PulayMixer']


class DIISMixer(History, LinearMixer, Metric):
    r""" DIIS mixer """

    def __init__(self, weight=0.1, history=2, metric=None, **kwargs):
        # This will call History.__init__
        super().__init__(history, 2)
        LinearMixer.__init__(self, weight)
        Metric.__init__(self, metric)

    def coefficients(self):
        r""" Calculate the coefficients according to Pulay's method """

        n_h = self.history
        if n_h == 1:
            return _a.arrayd([self.weight])

        # Initialize the matrix to be solved against
        B = _a.emptyd([n_h + 1, n_h + 1])

        # Fill matrix B
        for i in range(n_h):
            ei = self._hist[1][i]
            for j in range(i + 1, n_h):
                ej = self._hist[1][j]

                B[i, j] = self.inner(ei, ej)
                B[j, i] = B[i, j]
            B[i, i] = self.inner(ei, ei)
        B[:, n_h] = 1.
        B[n_h, :] = 1.
        B[n_h, n_h] = 0.

        # Although B contains 1 and a number on the order of
        # number of elements (self._hist[0].size), it seems very
        # numerically stable.
        
        # Create RHS
        RHS = _a.zerosd(n_h + 1)
        RHS[-1] = 1

        try:
            # Apparently we cannot use assume_a='sym'
            # Is this because sym also implies positive definitiness?
            # However, these are matrices of order ~30, so we don't care
            c = solve(B, RHS)
        except:
            # We have a LinalgError
            return _a.arrayd([self.weight])

        # -c[-1] == Lagrange multiplier (currently not used for anything)
        lagrange = -c[-1]
        return c[:-1]

    def __call__(self, f, df):
        # Add to history
        self.append(f, df)
        # Calculate new mixing quantity
        return self.mix(self.coefficients())

    def mix(self, coeff):
        r""" Calculate a new variable :math:`f'` using history and input coefficients
        
        Parameters
        ----------
        coeff : numpy.ndarray
           coefficients used for extrapolation
        """
        if len(coeff) == 1:
            return self._hist[0][0] + self._hist[1][0] * coeff[0]
        return reduce(add, map(mul, coeff, self._hist[0])) + \
            reduce(add, map(mul, coeff * self.weight, self._hist[1]))


PulayMixer = DIISMixer
