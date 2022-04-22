# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from sisl._internal import set_module
from .base import BaseMixer


__all__ = ['LinearMixer']


@set_module("sisl.mixing")
class LinearMixer(BaseMixer):
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

    def __init__(self, weight=0.2):
        # No parameters passed
        super().__init__(weight)

    def __str__(self):
        r""" String representation """
        return f"{self.__class__.__name__}{{weight: {self.weight:.4f}}}"

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
