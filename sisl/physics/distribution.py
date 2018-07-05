"""Distribution functions
=========================

.. module:: sisl.physics.distribution
   :noindex:

Various distributions using different smearing techniques.

.. autosummary::
   :toctree:

   get_distribution
   gaussian
   lorentzian
   fermi_dirac
   bose_einstein
   cold


.. autofunction:: gaussian
   :noindex:
.. autofunction:: lorentzian
   :noindex:
.. autofunction:: fermi_dirac
   :noindex:
.. autofunction:: bose_einstein
   :noindex:
.. autofunction:: cold
   :noindex:

"""
from __future__ import print_function, division

from functools import partial

import numpy as np
from scipy.special import erf
_pi = np.pi
_sqrt_2pi = (2 * _pi) ** 0.5

__all__ = ['get_distribution', 'gaussian', 'lorentzian']
__all__ += ['fermi_dirac', 'bose_einstein', 'cold']


def get_distribution(method, smearing=0.1, x0=0.):
    r""" Create a distribution function, Gaussian, Lorentzian etc.

    See the details regarding the distributions in their respective documentation.

    Parameters
    ----------
    method: {'gaussian', 'lorentzian'}
       distribution function
    smearing: float, optional
       smearing parameter for the method (:math:`\sigma` for Gaussian, :math:`\gamma` for Lorenztian etc.)
    x0: float, optional
       maximum of the distribution function

    Returns
    -------
    callable
        a function which accepts one argument
    """
    m = method.lower()
    if m in ['gauss', 'gaussian']:
        return partial(gaussian, sigma=smearing, x0=x0)
    elif m in ['lorentz', 'lorentzian']:
        return partial(lorentzian, gamma=smearing, x0=x0)
    elif m in ['fd', 'fermi_dirac']:
        return partial(fermi_dirac, kT=smearing, mu=x0)
    elif m in ['bose_einstein']:
        return partial(bose_einstein, kT=smearing, mu=x0)
    elif m in ['cold']:
        return partial(cold, kT=smearing, mu=x0)
    raise ValueError("get_distribution does not implement the {} distribution function, have you mispelled?".format(method))


def gaussian(x, sigma=0.1, x0=0.):
    r""" Gaussian distribution function

    .. math::
        G(x,\sigma,x_0) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\Big[\frac{- (x - x_0)^2}{2\sigma^2}\Big]

    Parameters
    ----------
    x: array_like
        points at which the Gaussian distribution is calculated
    sigma: float, optional
        spread of the Gaussian
    x0: float, optional
        maximum position of the Gaussian

    Returns
    -------
    numpy.ndarray
        the Gaussian distribution, same length as `x`
    """
    return np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) / (_sqrt_2pi * sigma)


def lorentzian(x, gamma=0.1, x0=0.):
    r""" Lorentzian distribution function

    .. math::
        L(x,\gamma,x_0) = \frac{1}{\pi}\frac{\gamma}{(x-x_0)^2 + \gamma^2}

    Parameters
    ----------
    x: array_like
        points at which the Lorentzian distribution is calculated
    gamma: float, optional
        spread of the Lorentzian
    x0: float, optional
        maximum position of the Lorentzian

    Returns
    -------
    numpy.ndarray
        the Lorentzian distribution, same length as `x`
    """
    return (gamma / _pi) / ((x - x0) ** 2 + gamma ** 2)


def fermi_dirac(E, kT=0.1, mu=0.):
    r""" Fermi-Dirac distribution function

    .. math::
        n_F(E,k_BT,\mu) = \frac{1}{\exp\Big[\frac{E - \mu}{k_BT}\Big] + 1}

    Parameters
    ----------
    E: array_like
        energy evaluation points
    kT: float, optional
        temperature broadening
    mu: float, optional
        chemical potential

    Returns
    -------
    numpy.ndarray
        the Fermi-Dirac distribution, same length as `E`
    """
    return 1. / (np.exp((E - mu) / kT) + 1.)


def bose_einstein(E, kT=0.1, mu=0.):
    r""" Bose-Einstein distribution function

    .. math::
        n_B(E,k_BT,\mu) = \frac{1}{\exp\Big[\frac{E - \mu}{k_BT}\Big] - 1}

    Parameters
    ----------
    E: array_like
        energy evaluation points
    kT: float, optional
        temperature broadening
    mu: float, optional
        chemical potential

    Returns
    -------
    numpy.ndarray
        the Bose-Einstein distribution, same length as `E`
    """
    return 1. / (np.exp((E - mu) / kT) - 1.)


def cold(E, kT=0.1, mu=0.):
    r""" Cold smearing function, Marzari-Vanderbilt, PRL 82, 16, 1999

    .. math::
        C(E,k_BT,\mu) = \frac12 + \mathrm{erf}\Big(-\frac{E-\mu}{k_BT}-\frac1{\sqrt2}\Big)
        + \frac1{\sqrt{2\pi}} \exp\Bigg\{-\Big[\frac{E-\mu}{k_BT}+\frac1{\sqrt2}\Big]^2\Bigg\}

    Parameters
    ----------
    E: array_like
        energy evaluation points
    kT: float, optional
        temperature broadening
    mu: float, optional
        chemical potential

    Returns
    -------
    numpy.ndarray
        the Cold smearing distribution function, same length as `E`
    """
    x = - (E - mu) / kT - 1 / 2 ** 0.5
    return 0.5 + 0.5 * erf(x) + 1 / _sqrt_2pi * np.exp(-x**2)
