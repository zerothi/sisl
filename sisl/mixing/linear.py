from functools import reduce

from sisl._internal import set_module
from .base import Mixer


__all__ = ['LinearMixer']


@set_module("sisl.mixing")
class LinearMixer(Mixer):
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

    def __init__(self, weight=0.1):
        # No parameters passed
        super().__init__()
        assert weight > 0
        self._weight = weight

    def __str__(self):
        r""" String representation """
        return self.__class__.__name__ + f"{{weight: {self.weight:.4f}}}"

    @property
    def weight(self):
        r""" Weight used for the linear mixing """
        return self._weight

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
