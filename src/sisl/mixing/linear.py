# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Any, Optional

import numpy.typing as npt

from sisl._internal import set_module

from .base import BaseHistoryWeightMixer, T

__all__ = ["LinearMixer", "AndersonMixer"]


@set_module("sisl.mixing")
class LinearMixer(BaseHistoryWeightMixer):
    r"""Linear mixing

    The linear mixing is solely defined using a weight, and the resulting functional
    may then be calculated via:

    .. math::

        \mathbf f^{i+1} = \mathbf f^i + w \delta \mathbf f^i

    Parameters
    ----------
    weight : float, optional
       mixing weight
    """

    __slots__ = ()

    def __call__(self, f: T, df: T, append: bool = True) -> T:
        r"""Calculate a new variable :math:`\mathbf f'` using input and output of the functional

        Parameters
        ----------
        f : object
           input variable for the functional
        df : object
           derivative of the functional
        append : bool, optional
           whether to append to the history
        """
        super().__call__(f, df, append=append)
        return f + self.weight * df


class AndersonMixer(BaseHistoryWeightMixer):
    r""" Anderson mixing

    The Anderson mixing assumes that the mixed input/output are linearly
    related. Hence

    .. math::

       |\bar{n}^{m}_{\mathrm{in}/\mathrm{out}\rangle =
          (1 - \beta)|n^{m}_{\mathrm{in}/\mathrm{out}\rangle
          + \beta|n^{m-1}_{\mathrm{in}/\mathrm{out}\rangle

    Here the optimal choice :math:`\beta` is calculated as:

    .. math::

       \boldsymbol\delta_i &= \mathbf f_i^{\mathrm{out}} - \mathbf f_i^{\mathrm{in}}
       \\
       \beta &= \frac{\langle \boldsymbol\delta_i | \boldsymbol\delta_i - \boldsymbol\delta_{i-1}\rangle}
         {\langle \boldsymbol\delta_i - \boldsymbol\delta_{i-1}| \boldsymbol\delta_i - \boldsymbol\delta_{i-1} \rangle}

    Finally the resulting output becomes:

    .. math::

          |n^{m+1}\rangle =
          (1 - \alpha)|\bar n^m_{\mathrm{in}}\rangle
          + \alpha|\bar n^m_{\mathrm{out}}\rangle

    See :cite:`Johnson1988` for more details.
    """

    __slots__ = ()

    @staticmethod
    def _beta(df1: T, df2: T) -> npt.NDArray:
        # Minimize the average densities for the delta variable
        def metric(a, b):
            return a.ravel().conj().dot(b.ravel()).real

        ddf = df2 - df1
        beta = metric(df2, ddf) / metric(ddf, ddf)

        return beta

    def __call__(
        self, f: T, df: T, delta: Optional[Any] = None, append: bool = True
    ) -> T:
        r"""Calculate a new variable :math:`\mathbf f'` using input and output of the functional

        Parameters
        ----------
        f : object
           input variable for the functional
        df : object
           derivative of the functional
        """
        if delta is None:
            # not a copy, simply the same reference
            delta = df

        # Get last elements
        if len(self.history) > 0:
            last = self.history[-1]
            f1 = last[0]
            fdf1 = last[1]
            d1 = last[-1]
        else:
            f1 = None

        # the current iterations input + output variables
        f2 = f
        fdf2 = f + df
        d2 = delta

        # store new last variables
        #   delta is used for calculating beta, nothing more
        # here n refers to the variable (density) we are mixing
        #  and the integer corresponds to the iteration count
        super().__call__(f2, fdf2, d2, append=append)

        if f1 is None:
            # this is linear mixing for the first step
            return f + self.weight * df

        # calculate next position
        beta = self._beta(d1, d2)

        # Now calculate the new averages
        nin = (1 - beta) * f2 + beta * f1
        nout = (1 - beta) * fdf2 + beta * fdf1

        return (1 - self.weight) * nin + self.weight * nout
