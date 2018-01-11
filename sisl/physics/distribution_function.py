from __future__ import print_function, division

from functools import partial

import numpy as np
pi = np.pi
pi_sqrt = pi ** 0.5

__all__ = ['distribution', 'gaussian', 'lorentzian']


def distribution(method, smearing=0.1):
    r""" Create a distribution function for input in e.g. `DOS`. Gaussian, Lorentzian etc.

    In the following :math:`\epsilon` are the eigenvalues contained in this `EigenState`.

    The Gaussian distribution is calculated as:

    .. math::
        G(E) = \sum_i \frac{1}{\sqrt{2\pi\sigma^2}}\exp\big[- (E - \epsilon_i)^2 / (2\sigma^2)\big]

    where :math:`\sigma` is the `smearing` parameter.

    The Lorentzian distribution is calculated as:

    .. math::
        L(E) = \sum_i \frac{1}{\pi}\frac{\gamma}{(E - \epsilon_i)^2 + \gamma^2}

    where :math:`\gamma` is the `smearing` parameter, note that here :math:`\gamma` is the
    half-width at half-maximum (:math:`2\gamma` would be the full-width at half-maximum).

    Parameters
    ----------
    method : {'gaussian', 'lorentzian'}
        the distribution function
    smearing : float, optional
        the smearing parameter for the method (:math:`\sigma` for Gaussian, etc.)

    Returns
    -------
    func : a function which accepts one argument
    """
    if method.lower() in ['gauss', 'gaussian']:
        return partial(gaussian, sigma=smearing)
    elif method.lower() in ['lorentz', 'lorentzian']:
        return partial(lorentzian, gamma=smearing)
    else:
        raise ValueError("distribution currently only implements 'gaussian' or "
                         "'lorentzian' distribution functions")
    return func


def gaussian(x, sigma=0.1):
    r""" Gaussian distribution function

    The Gaussian distribution is calculated as:

    .. math::
        G(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\big[- x^2 / (2\sigma^2)\big]

    Parameters
    ----------
    x : array_like
        the points at which the Gaussian distribution is calculated
    sigma : float, optional
        the spread of the Gaussian

    Returns
    -------
    y : array_like
        the Gaussian distribution, same length as `x`
    """
    return np.exp(-x ** 2 / (2 * sigma ** 2)) / ((2 * pi) ** 0.5 * sigma)


def lorentzian(x, gamma=0.1):
    r""" Lorentzian distribution function

    The Lorentzian distribution is calculated as:

    .. math::
        L(x) = \frac{1}{\pi}\frac{\gamma}{x^2 + \gamma^2}

    Parameters
    ----------
    x : array_like
        the points at which the Lorentzian distribution is calculated
    gamma : float, optional
        the spread of the Lorentzian

    Returns
    -------
    y : array_like
        the Lorentzian distribution, same length as `x`
    """
    return (gamma / pi) / (x ** 2 + gamma ** 2)
