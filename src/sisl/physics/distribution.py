# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

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
from typing import Union

import numpy as np
import numpy.typing as npt
from numpy import exp, expm1
from scipy.special import erf

from sisl._internal import set_module
from sisl.typing import DistributionFunc, DistributionStr

_pi = np.pi
_sqrt_2pi = (2 * _pi) ** 0.5

__all__ = ["get_distribution", "gaussian", "lorentzian"]
__all__ += ["fermi_dirac", "bose_einstein", "cold"]
__all__ += ["step_function", "heaviside"]


@set_module("sisl.physics")
def get_distribution(
    method: DistributionStr,
    smearing: float = 0.1,
    x0: Union[float, npt.ArrayLike] = 0.0,
) -> DistributionFunc:
    r"""Create a distribution function, Gaussian, Lorentzian etc.

    See the details regarding the distributions in their respective documentation.

    Parameters
    ----------
    method :
       distribution function
    smearing :
       smearing parameter for methods that have a smearing
    x0 :
       maximum/middle of the distribution function

    Returns
    -------
    callable
        a function which accepts one argument
    """
    m = method.lower().replace("-", "_")
    if m in ("gauss", "gaussian"):
        return partial(gaussian, sigma=smearing, x0=x0)
    elif m in ("lorentz", "lorentzian"):
        return partial(lorentzian, gamma=smearing, x0=x0)
    elif m in ("fd", "fermi", "fermi_dirac"):
        return partial(fermi_dirac, kT=smearing, mu=x0)
    elif m in ("be", "bose_einstein"):
        return partial(bose_einstein, kT=smearing, mu=x0)
    elif m in ("cold"):
        return partial(cold, kT=smearing, mu=x0)
    elif m in ("step", "step_function"):
        return partial(step_function, x0=x0)
    elif m in ("heavi", "heavy", "heaviside"):
        return partial(heaviside, x0=x0)
    raise ValueError(
        f"get_distribution does not implement the {method} distribution function, have you misspelled?"
    )


@set_module("sisl.physics")
def gaussian(
    x: npt.ArrayLike, sigma: float = 0.1, x0: Union[float, npt.ArrayLike] = 0.0
) -> np.ndarray:
    r"""Gaussian distribution function

    .. math::
        G(x,\sigma,x_0) = \frac1{\sqrt{2\pi\sigma^2}}\exp\Big[\frac{- (x - x_0)^2}{2\sigma^2}\Big]

    Parameters
    ----------
    x : array_like
        points at which the Gaussian distribution is calculated
    sigma :
        spread of the Gaussian
    x0 : array_like, optional
        maximum position of the Gaussian

    Returns
    -------
    numpy.ndarray
        the Gaussian distribution, same length as `x`
    """
    dx = (x - x0) / (sigma * 2**0.5)
    return exp(-dx * dx) / (_sqrt_2pi * sigma)


@set_module("sisl.physics")
def lorentzian(
    x: npt.ArrayLike, gamma: float = 0.1, x0: Union[float, npt.ArrayLike] = 0.0
) -> np.ndarray:
    r"""Lorentzian distribution function

    .. math::
        L(x,\gamma,x_0) = \frac1\pi\frac{\gamma}{(x-x_0)^2 + \gamma^2}

    Parameters
    ----------
    x :
        points at which the Lorentzian distribution is calculated
    gamma :
        spread of the Lorentzian
    x0 :
        maximum position of the Lorentzian

    Returns
    -------
    numpy.ndarray
        the Lorentzian distribution, same length as `x`
    """
    return (gamma / _pi) / ((x - x0) ** 2 + gamma * gamma)


@set_module("sisl.physics")
def fermi_dirac(
    E: npt.ArrayLike, kT: float = 0.1, mu: Union[float, npt.ArrayLike] = 0.0
) -> np.ndarray:
    r"""Fermi-Dirac distribution function

    .. math::
        n_F(E,k_BT,\mu) = \frac1{\exp\Big[\frac{E - \mu}{k_BT}\Big] + 1}

    Parameters
    ----------
    E :
        energy evaluation points
    kT :
        temperature broadening
    mu :
        chemical potential

    Returns
    -------
    numpy.ndarray
        the Fermi-Dirac distribution, same length as `E`
    """
    return 1.0 / (expm1((E - mu) / kT) + 2.0)


@set_module("sisl.physics")
def bose_einstein(
    E: npt.ArrayLike, kT: float = 0.1, mu: Union[float, npt.ArrayLike] = 0.0
) -> np.ndarray:
    r"""Bose-Einstein distribution function

    .. math::
        n_B(E,k_BT,\mu) = \frac1{\exp\Big[\frac{E - \mu}{k_BT}\Big] - 1}

    Parameters
    ----------
    E :
        energy evaluation points
    kT :
        temperature broadening
    mu :
        chemical potential

    Returns
    -------
    numpy.ndarray
        the Bose-Einstein distribution, same length as `E`
    """
    return 1.0 / expm1((E - mu) / kT)


@set_module("sisl.physics")
def cold(
    E: npt.ArrayLike, kT: float = 0.1, mu: Union[float, npt.ArrayLike] = 0.0
) -> np.ndarray:
    r""" Cold smearing function

    For more details see :cite:`Marzari1999`.

    .. math::
        C(E,k_BT,\mu) = \frac12 &+ \mathrm{erf}\Big(-\frac{E-\mu}{k_BT}-\frac1{\sqrt2}\Big)
        \\
        &+ \frac1{\sqrt{2\pi}} \exp\Bigg\{-\Big[\frac{E-\mu}{k_BT}+\frac1{\sqrt2}\Big]^2\Bigg\}

    Parameters
    ----------
    E :
        energy evaluation points
    kT :
        temperature broadening
    mu :
        chemical potential

    Returns
    -------
    numpy.ndarray
        the Cold smearing distribution function, same length as `E`
    """
    x = -(E - mu) / kT - 1 / 2**0.5
    return 0.5 + 0.5 * erf(x) + exp(-x * x) / _sqrt_2pi


@set_module("sisl.physics")
def heaviside(x: npt.ArrayLike, x0: Union[float, npt.ArrayLike] = 0.0) -> np.ndarray:
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
    x :
        points at which the Heaviside step distribution is calculated
    x0 :
        step position

    Returns
    -------
    numpy.ndarray
        the Heaviside step function distribution, same length as `x`
    """
    x = np.asarray(x)
    shape = np.broadcast_shapes(x.shape, np.asarray(x0).shape)
    H = np.zeros_like(x, shape=shape)
    H[x == x0] = 0.5
    H[x > x0] = 1.0
    return H


@set_module("sisl.physics")
def step_function(
    x: npt.ArrayLike, x0: Union[float, npt.ArrayLike] = 0.0
) -> np.ndarray:
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
    x :
        points at which the step distribution is calculated
    x0 :
        step position

    Returns
    -------
    numpy.ndarray
        the step function distribution, same length as `x`
    """
    x = np.asarray(x)
    shape = np.broadcast_shapes(x.shape, np.asarray(x0).shape)
    s = np.ones_like(x, shape=shape)
    s[x == x0] = 0.5
    s[x > x0] = 0.0
    return s
