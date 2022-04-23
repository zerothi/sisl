# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from functools import reduce
from operator import add
from numbers import Real
import numpy as np

from sisl._internal import set_module
import sisl._array as _a
from sisl.linalg import solve
from .base import BaseHistoryMixer, History


__all__ = ['DIISMixer', 'PulayMixer']
__all__ += ['AdaptiveDIISMixer', 'AdaptivePulayMixer']


@set_module("sisl.mixing")
class DIISMixer(BaseHistoryMixer):
    r""" Direct inversion of the iterative subspace (DIIS mixing)

    This mixing method (also known as Pulay mixing) estimates the next
    trial function given a set of previously inputs and derivatives of
    those inputs.

    Its implementation is general in the sense that one can manually define
    which values are used for the subspace. I.e. generally the subspace
    metric is calculated using:

    .. math::

       \delta_i &= F_i^{\mathrm{out}} - F_i^{\mathrm{in}}
       \\
       m_{ij} &= \langle \delta_i | \delta_j\rangle

    And then the mixing coefficients is calculated using the regular method
    for a matrix :math:`\mathbf m`.
    Generally the metric is calculated using :math:`\delta`, however, by
    calling the object with an optional 3rd argument, the metric will use
    that argument instead of :math:`\delta` but still use :math:`\delta` when
    extrapolating the coefficients.
    This may be useful for testing various metrics based on alternate values.

    Alternatively one can pass a `metric` argument that can pre-process the
    :math:`\delta` variable.

    Parameters
    ----------
    weight : float, optional
       weight used for the derivative of the functional.
       The mixer will use a weight of :math:`1-w` for the *old* value
    history : int or History, optional
       how many history steps it will use in the estimation of the
       new functional
    metric : callable, optional
       the metric used for the two values, defaults to ``lambda a, b: a.ravel().conj().dot(b.ravel).real``
    """
    __slots__ = ("_metric",)

    def __init__(self, weight=0.1, history=2, metric=None):
        # This will call History.__init__
        super().__init__(weight, history)
        if metric is None:
            def metric(a, b):
                return a.ravel().conj().dot(b.ravel()).real
        self._metric = metric

    def __str__(self):
        r""" String representation """
        hist = str(self.history).replace("\n", "\n  ")
        return f"{self.__class__.__name__}{{weight: {self.weight:.4f},\n  {hist}\n}}"

    def solve_lagrange(self):
        r""" Calculate the coefficients according to Pulay's method, return everything + Lagrange multiplier """
        hist = self.history
        n_h = len(hist)
        metric = self._metric

        if n_h == 0:
            # Externally the coefficients should reflect the weight per previous iteration.
            # The mixing weight is an additional parameter
            return _a.arrayd([1.]), 100.
        elif n_h == 1:
            return _a.arrayd([1.]), metric(hist[0][-1], hist[0][-1])

        # Initialize the matrix to be solved against
        B = _a.emptyd([n_h + 1, n_h + 1])

        # Fill matrix B
        for i in range(n_h):
            ei = hist[i][-1]
            B[i, i] = metric(ei, ei)
            for j in range(i + 1, n_h):
                ej = hist[j][-1]

                B[i, j] = metric(ei, ej)
                B[j, i] = B[i, j]
        B[:, n_h] = 1.
        B[n_h, :] = 1.
        B[n_h, n_h] = 0.

        # Although B contains 1 and a number on the order of
        # number of elements (self._hist.size), it seems very
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
        except np.linalg.LinAlgError as e:
            # We have a LinalgError
            return _a.arrayd([1.]), metric(hist[-1][-1], hist[-1][-1])

    def coefficients(self):
        r""" Calculate coefficients of the Lagrangian """
        c, lagrange = self.solve_lagrange()
        return c

    def mix(self, coefficients):
        r""" Calculate a new variable :math:`f'` using history and input coefficients

        Parameters
        ----------
        coefficients : numpy.ndarray
           coefficients used for extrapolation
        """
        def frac_hist(coef, hist):
            return coef * (hist[0] + self.weight * hist[1])
        return reduce(add, map(frac_hist, coefficients, self.history))

    def __call__(self, f, df, delta=None, append=True):
        # Add to history
        if append:
            if delta is None:
                self.history.append(f, df)
            else:
                self.history.append(f, df, delta)

        # Calculate new mixing quantity
        return self.mix(self.coefficients())


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
    __slots__ = ("_weight_min", "_weight_delta")

    def __init__(self, weight=(0.03, 0.5), history=2, metric=None):
        if isinstance(weight, Real):
            weight = (max(0.001, weight * 0.1), min(1., weight * 2))
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
