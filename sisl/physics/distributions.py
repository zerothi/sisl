"""Distribution functions
=========================

.. module:: sisl.physics.distributions
   :noindex:

Various distributions using different smearing techniques.

.. autosummary::
   :toctree:

   distribution
   gaussian
   lorentzian

.. autofunction:: gaussian
   :noindex:
.. autofunction:: lorentzian
   :noindex:

"""
from __future__ import print_function, division

from functools import partial

import numpy as np
_pi = np.pi
_sqrt_2pi = (2 * _pi) ** 0.5

__all__ = ['distribution', 'gaussian', 'lorentzian']


def distribution(method, smearing=0.1):
    r""" Create a distribution function for input in e.g. `DOS`. Gaussian, Lorentzian etc.

    The Gaussian distribution is calculated as:

    .. math::
        G(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\Big[\frac{- x^2}{2\sigma^2}\Big]

    where :math:`\sigma` is the `smearing` parameter.

    The Lorentzian distribution is calculated as:

    .. math::
        L(x) = \frac{1}{\pi}\frac{\gamma}{x^2 + \gamma^2}

    where :math:`\gamma` is the `smearing` parameter, note that here :math:`\gamma` is the
    half-width at half-maximum (:math:`2\gamma` the full-width at half-maximum).

    Parameters
    ----------
    method: {'gaussian', 'lorentzian'}
        the distribution function
    smearing: float, optional
        the smearing parameter for the method (:math:`\sigma` for Gaussian, :math:`\gamma` for Lorenztian etc.)

    Returns
    -------
    callable
        a function which accepts one argument
    """
    if method.lower() in ['gauss', 'gaussian']:
        return partial(gaussian, sigma=smearing)
    elif method.lower() in ['lorentz', 'lorentzian']:
        return partial(lorentzian, gamma=smearing)
    raise ValueError("distribution currently only implements 'gaussian' or "
                     "'lorentzian' distribution functions")


def gaussian(x, sigma=0.1):
    r""" Gaussian distribution function

    The Gaussian distribution is calculated as:

    .. math::
        G(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\Big[\frac{- x^2}{2\sigma^2}\Big]

    Parameters
    ----------
    x: array_like
        the points at which the Gaussian distribution is calculated
    sigma: float, optional
        the spread of the Gaussian

    Returns
    -------
    numpy.ndarray
        the Gaussian distribution, same length as `x`
    """
    return np.exp(-x ** 2 / (2 * sigma ** 2)) / (_sqrt_2pi * sigma)


def lorentzian(x, gamma=0.1):
    r""" Lorentzian distribution function

    The Lorentzian distribution is calculated as:

    .. math::
        L(x) = \frac{1}{\pi}\frac{\gamma}{x^2 + \gamma^2}

    Parameters
    ----------
    x: array_like
        the points at which the Lorentzian distribution is calculated
    gamma: float, optional
        the spread of the Lorentzian

    Returns
    -------
    numpy.ndarray
        the Lorentzian distribution, same length as `x`
    """
    return (gamma / _pi) / (x ** 2 + gamma ** 2)
