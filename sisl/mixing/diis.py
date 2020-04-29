from functools import reduce
from operator import add, mul
from numbers import Real
import numpy as np

from sisl._internal import set_module
import sisl._array as _a
from sisl.linalg import solve
from .base import History, Metric
from .linear import LinearMixer


__all__ = ['DIISMixer', 'PulayMixer']
__all__ += ['AdaptiveDIISMixer', 'AdaptivePulayMixer']


@set_module("sisl.mixing")
class DIISMixer(History, LinearMixer, Metric):
    r""" DIIS mixer """

    def __init__(self, weight=0.1, history=2, metric=None):
        # This will call History.__init__
        super().__init__(history, 2)
        LinearMixer.__init__(self, weight)
        Metric.__init__(self, metric)

    def __str__(self):
        r""" String representation """
        hist = History.__str__(self).replace(self.__class__.__name__, History.__name__)
        return self.__class__.__name__ + f"{{weight: {self.weight:.4f},\n  {hist}\n}}"

    def solve_lagrange(self):
        r""" Calculate the coefficients according to Pulay's method, return everything + Lagrange multiplier """
        n_h = len(self._hist[1])
        if n_h == 0:
            # Externally the coefficients should reflect the weight per previous iteration.
            # The mixing weight is an additional parameter
            return _a.arrayd([1.]), 100.
        elif n_h == 1:
            return _a.arrayd([1.]), self.inner(self._hist[1][0], self._hist[1][0])

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
            return c[:-1], -c[-1]
        except np.linalg.LinAlgError:
            # We have a LinalgError
            return _a.arrayd([1.]), self.inner(self._hist[1][-1], self._hist[1][-1])

    def coefficients(self):
        r""" Calculate coefficients of the Lagrangian """
        c, lagrange = self.solve_lagrange()
        return c

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
        return reduce(add, map(mul, coeff, self._hist[0])) + \
            reduce(add, map(mul, coeff * self.weight, self._hist[1]))


PulayMixer = set_module("sisl.mixing")(type("PulayMixer", (DIISMixer, ), {}))


@set_module("sisl.mixing")
class AdaptiveDIISMixer(DIISMixer):
    r""" Adapt the mixing weight according to the Lagrange multiplier

    The Lagrange multiplier calculated in a DIIS/Pulay mixing scheme
    is the squared norm of the residual that is minimized using the
    Lagrange method. It holds information on the closeness of the functional
    to a minimum.

    Thus we can use the Lagrange multiplier to adjust the weight such that
    for large values we know our next guess (:math:`f_{\mathrm{new}}`) will
    be relatively far from the true saddle point, and for small values we
    will be close to the saddle point.
    """

    def __init__(self, weight=(0.03, 0.5), history=2, metric=None):
        if isinstance(weight, Real):
            weight = (max(0.001, weight - 0.1), min(1., weight + 0.1))
        super().__init__(weight[0], history, metric)
        self._weight_min = weight[0]
        self._weight_delta = weight[1] - weight[0]

    def adjust_weight(self, lagrange, offset=13, spread=7):
        r""" Adjust the weight according to the Lagrange multiplier.

        Once close to convergence the Lagrange multiplier will be close to 0, otherwise it will go
        towards infinity.
        We here adjust using the Fermi-function to hit the minimum/maximum weight with a
        suitable spread
        """
        exp_lag_log = np.exp((np.log(lagrange) + offset) / spread)
        self._weight = self._weight_min + self._weight_delta / (exp_lag_log + 1)

    def coefficients(self):
        r""" Calculate coefficients and adjust weights according to a Lagrange multiplier """
        c, lagrange = self.solve_lagrange()
        self.adjust_weight(lagrange)
        return c


AdaptivePulayMixer = set_module("sisl.mixing")(type("AdaptivePulayMixer", (AdaptiveDIISMixer, ), {}))
