from functools import reduce
from .base import Mixer


__all__ = ['LinearMixer']


class LinearMixer(Mixer):
    r""" Linear mixing

    The linear mixing is solely defined using a weight, and the resulting functional
    may then be calculated via:

    .. math::

        f_{\mathrm{in}}^{i+1} = (1 - w) f_{\mathrm{in}}^i + w f_{\mathrm{out}}^i

    Parameters
    ----------
    weight : float, optional
       mixing weight
    """

    def __init__(self, weight=0.1):
        # No parameters passed
        super().__init__()
        self._weight = weight

    @property
    def weight(self):
        r""" Weight used for the linear mixing weights """
        return self._weight

    def __call__(self, f_in, f_out):
        r""" Calculate a new variable :math:`f'` using input and output of the functional
        
        Parameters
        ----------
        f_in : object
           input variable for the functional
        f_out : object
           output variable of the functional
        """
        return (1 - self.weight) * f_in + self.weight * f_out
