"""Distribution functions
=========================

Various distributions using different smearing techniques.

   get_distribution
   gaussian
   lorentzian
   fermi_dirac
   bose_einstein
   cold
   step_function
   heaviside

"""

from functools import partial

import numpy as np
from numpy import exp, expm1
from scipy.special import erf

from sisl._internal import set_module

_pi = np.pi
_sqrt_2pi = (2 * _pi) ** 0.5

__all__ = ['get_distribution', 'gaussian', 'lorentzian']
__all__ += ['fermi_dirac', 'bose_einstein', 'cold']
__all__ += ['step_function', 'heaviside']


@set_module("sisl.physics")
def get_distribution(method, smearing=0.1, x0=0.):
    r""" Create a distribution function, Gaussian, Lorentzian etc.

    See the details regarding the distributions in their respective documentation.

    Parameters
    ----------
    method : {'gaussian', 'lorentzian', 'fermi_dirac', 'bose_einstein', 'step_function', 'heaviside'}
       distribution function
    smearing : float, optional
       smearing parameter for methods that have a smearing
    x0 : float, optional
       maximum/middle of the distribution function

    Returns
    -------
    callable
        a function which accepts one argument
    """
    m = method.lower().replace('-', '_')
    if m in ['gauss', 'gaussian']:
        return partial(gaussian, sigma=smearing, x0=x0)
    elif m in ['lorentz', 'lorentzian']:
        return partial(lorentzian, gamma=smearing, x0=x0)
    elif m in ['fd', 'fermi', 'fermi_dirac']:
        return partial(fermi_dirac, kT=smearing, mu=x0)
    elif m in ['be', 'bose_einstein']:
        return partial(bose_einstein, kT=smearing, mu=x0)
    elif m in ['cold']:
        return partial(cold, kT=smearing, mu=x0)
    elif m in ['step', 'step_function']:
        return partial(step_function, x0=x0)
    elif m in ['heavi', 'heavy', 'heaviside']:
        return partial(heaviside, x0=x0)
    raise ValueError(f"get_distribution does not implement the {method} distribution function, have you mispelled?")


@set_module("sisl.physics")
def gaussian(x, sigma=0.1, x0=0.):
    r""" Gaussian distribution function

    .. math::
        G(x,\sigma,x_0) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\Big[\frac{- (x - x_0)^2}{2\sigma^2}\Big]

    Parameters
    ----------
    x : array_like
        points at which the Gaussian distribution is calculated
    sigma : float, optional
        spread of the Gaussian
    x0 : float, optional
        maximum position of the Gaussian

    Returns
    -------
    numpy.ndarray
        the Gaussian distribution, same length as `x`
    """
    dx = (x - x0) / (sigma * 2 ** 0.5)
    return exp(- dx * dx) / (_sqrt_2pi * sigma)


@set_module("sisl.physics")
def lorentzian(x, gamma=0.1, x0=0.):
    r""" Lorentzian distribution function

    .. math::
        L(x,\gamma,x_0) = \frac{1}{\pi}\frac{\gamma}{(x-x_0)^2 + \gamma^2}

    Parameters
    ----------
    x : array_like
        points at which the Lorentzian distribution is calculated
    gamma : float, optional
        spread of the Lorentzian
    x0 : float, optional
        maximum position of the Lorentzian

    Returns
    -------
    numpy.ndarray
        the Lorentzian distribution, same length as `x`
    """
    return (gamma / _pi) / ((x - x0) ** 2 + gamma * gamma)


@set_module("sisl.physics")
def fermi_dirac(E, kT=0.1, mu=0.):
    r""" Fermi-Dirac distribution function

    .. math::
        n_F(E,k_BT,\mu) = \frac{1}{\exp\Big[\frac{E - \mu}{k_BT}\Big] + 1}

    Parameters
    ----------
    E : array_like
        energy evaluation points
    kT : float, optional
        temperature broadening
    mu : float, optional
        chemical potential

    Returns
    -------
    numpy.ndarray
        the Fermi-Dirac distribution, same length as `E`
    """
    return 1. / (expm1((E - mu) / kT) + 2.)


@set_module("sisl.physics")
def bose_einstein(E, kT=0.1, mu=0.):
    r""" Bose-Einstein distribution function

    .. math::
        n_B(E,k_BT,\mu) = \frac{1}{\exp\Big[\frac{E - \mu}{k_BT}\Big] - 1}

    Parameters
    ----------
    E : array_like
        energy evaluation points
    kT : float, optional
        temperature broadening
    mu : float, optional
        chemical potential

    Returns
    -------
    numpy.ndarray
        the Bose-Einstein distribution, same length as `E`
    """
    return 1. / expm1((E - mu) / kT)


@set_module("sisl.physics")
def cold(E, kT=0.1, mu=0.):
    r""" Cold smearing function, Marzari-Vanderbilt, PRL 82, 16, 1999

    .. math::
        C(E,k_BT,\mu) = \frac12 &+ \mathrm{erf}\Big(-\frac{E-\mu}{k_BT}-\frac1{\sqrt2}\Big)
        \\
        &+ \frac1{\sqrt{2\pi}} \exp\Bigg\{-\Big[\frac{E-\mu}{k_BT}+\frac1{\sqrt2}\Big]^2\Bigg\}

    Parameters
    ----------
    E : array_like
        energy evaluation points
    kT : float, optional
        temperature broadening
    mu : float, optional
        chemical potential

    Returns
    -------
    numpy.ndarray
        the Cold smearing distribution function, same length as `E`
    """
    x = - (E - mu) / kT - 1 / 2 ** 0.5
    return 0.5 + 0.5 * erf(x) + exp(- x * x) / _sqrt_2pi


@set_module("sisl.physics")
def heaviside(x, x0=0.):
    r""" Heaviside step function

    .. math::
      :nowrap:

       \begin{align}
        H(x,x_0) = \left\{\begin{aligned}0&\quad \text{for }x < x_0
               \\
               0.5&\quad \text{for }x = x_0
               \\
               1&\quad \text{for }x>x_0
             \end{aligned}\right.
       \end{align}


    Parameters
    ----------
    x : array_like
        points at which the Heaviside step distribution is calculated
    x0 : float, optional
        step position

    Returns
    -------
    numpy.ndarray
        the Heaviside step function distribution, same length as `x`
    """
    H = np.zeros_like(x)
    H[x > x0] = 1.
    H[x == x0] = 0.5
    return H


@set_module("sisl.physics")
def step_function(x, x0=0.):
    r""" Step function, also known as :math:`1 - H(x)`

    This function equals one minus the Heaviside step function

    .. math::
      :nowrap:

       \begin{align}
        S(x,x_0) = \left\{\begin{aligned}1&\quad \text{for }x < x_0
               \\
               0.5&\quad \text{for }x = x_0
               \\
               0&\quad \text{for }x>x_0
             \end{aligned}\right.
       \end{align}


    Parameters
    ----------
    x : array_like
        points at which the step distribution is calculated
    x0 : float, optional
        step position

    Returns
    -------
    numpy.ndarray
        the step function distribution, same length as `x`
    """
    s = np.ones_like(x)
    s[x > x0] = 0.
    s[x == x0] = 0.5
    return s
