# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from sisl._internal import set_module
from .base import BaseWeightMixer


__all__ = ["LinearMixer", "AndersonMixer"]


@set_module("sisl.mixing")
class LinearMixer(BaseWeightMixer):
    r""" Linear mixing

    The linear mixing is solely defined using a weight, and the resulting functional
    may then be calculated via:

    .. math::

        f^{i+1} = f^i + w \delta f^i

    Parameters
    ----------
    weight : float, optional
       mixing weight
    """
    __slots__ = ()

    def __str__(self):
        r""" String representation """
        return f"{self.__class__.__name__}{{weight: {self.weight:.4f}}}"

    __repr__ = __str__

    def __call__(self, f, df):
        r""" Calculate a new variable :math:`f'` using input and output of the functional

        Parameters
        ----------
        f : object
           input variable for the functional
        df : object
           derivative of the functional
        """
        return f + self.weight * df


class AndersonMixer(BaseWeightMixer):
    r""" Anderson mixing

    The Anderson mixing assumes that the mixed input/output are linearly
    related. Hence 

    .. math::

       |\bar{n}^{m}_{\mathrm{in}/\mathrm{out}\rangle =
          (1 - \beta)|n^{m}_{\mathrm{in}/\mathrm{out}\rangle
          + \beta|n^{m-1}_{\mathrm{in}/\mathrm{out}\rangle

    Here the optimal choice :math:`\beta` is calculated as:

    .. math::

       \delta_i &= F_i^{\mathrm{out}} - F_i^{\mathrm{in}}
       \\
       \beta &= \frac{\langle \delta_i | \delta_i - \delta_{i-1}\rangle}
         {\langle \delta_i - \delta_{i-1}| \delta_i - \delta_{i-1} \rangle}

    Finally the resulting output becomes:

    .. math::

       |n_^{m+1}\rangle =
          (1 - \alpha)|\bar n^{m}_{\mathrm{in}\rangle
          + \alpha|\bar n^{m}_{\mathrm{out}\rangle

    See :cite:`Johnson1988` for more details.
    """

    __slots__ = ("_last",)

    def __init__(self, weight=0.2):
        super().__init__(weight)
        self._last = None

    def __str__(self):
        r""" String representation """
        return f"{self.__class__.__name__}{{weight: {self.weight:.4f}}}"

    __repr__ = __str__

    @staticmethod
    def _beta(df1, df2):
        # Minimize the average densities for the delta variable
        def metric(a, b):
            return a.ravel().conj().dot(b.ravel()).real

        ddf = df2 - df1
        beta = metric(df2, ddf) / metric(ddf, ddf)

        return beta

    def __call__(self, f, df, delta=None):
        r""" Calculate a new variable :math:`f'` using input and output of the functional

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

        # store new last variables
        #   delta is used for calculating beta, nothing more
        # here n refers to the variable (density) we are mixing
        #  and the integer corresponds to the iteration count
        n1 = self._last
        n2 = (f, f + df, delta)

        self._last = n2

        if n1 is None:
            return f + self.weight * df

        # calculate next position
        beta = self._beta(n1[2], n2[2])

        # Now calculate the new averages
        nin = (1 - beta) * n2[0] + beta * n1[0]
        nout = (1 - beta) * n2[1] + beta * n1[1]

        return (1 - self.weight) * nin + self.weight * nout
