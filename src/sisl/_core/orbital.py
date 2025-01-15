# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections import namedtuple
from collections.abc import Callable
from functools import partial
from math import factorial as fact
from math import pi
from math import sqrt as msqrt
from numbers import Integral, Real
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from numpy import cos, sin, take
from scipy.special import eval_genlaguerre, factorial, lpmv

try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:
    from scipy.integrate import cumtrapz as cumulative_trapezoid

from scipy.interpolate import UnivariateSpline

import sisl._array as _a
from sisl._internal import set_module
from sisl.constant import a0
from sisl.messages import warn
from sisl.shape import Sphere
from sisl.utils.mathematics import cart2spher, close

__all__ = [
    "Orbital",
    "SphericalOrbital",
    "AtomicOrbital",
    "HydrogenicOrbital",
    "GTOrbital",
    "STOrbital",
    "radial_minimize_range",
]


# Create the factor table for the real spherical harmonics
def _rfact(l, m):
    pi4 = 4 * pi
    if m == 0:
        return msqrt((2 * l + 1) / pi4)
    elif m < 0:
        return -msqrt(2 * (2 * l + 1) / pi4 * fact(l - m) / fact(l + m)) * (-1) ** m
    return msqrt(2 * (2 * l + 1) / pi4 * fact(l - m) / fact(l + m))


# This is a tuple of dicts
#  [0]{0} is l==0, m==0
#  [1]{-1} is l==1, m==-1
#  [1]{1} is l==1, m==1
# and so on.
# Calculate it up to l == 7 which is the j shell
# It will never be used, but in case somebody wishes to play with spherical harmonics
# then why not ;)
_rspher_harm_fact = tuple({m: _rfact(l, m) for m in range(-l, l + 1)} for l in range(8))
# Clean-up
del _rfact


def _rspherical_harm(m, l, theta, cos_phi):
    r""" Calculates the real spherical harmonics using :math:`Y_l^m(\theta, \varphi)` with :math:`\mathbf r\to \{r, \theta, \varphi\}`.

    These real spherical harmonics are via these equations:

    .. math::
        Y^m_l(\theta,\varphi) &= -(-1)^m\sqrt{2\frac{2l+1}{4\pi} \frac{(l-m)!}{(l+m)!}}
           P^{m}_l (\cos(\varphi)) \sin(m \theta) & m < 0\\
        Y^m_l(\theta,\varphi) &= \sqrt{\frac{2l+1}{4\pi}} P^{m}_l (\cos(\varphi)) & m = 0\\
        Y^m_l(\theta,\varphi) &= \sqrt{2\frac{2l+1}{4\pi} \frac{(l-m)!}{(l+m)!}}
           P^{m}_l (\cos(\varphi)) \cos(m \theta) & m > 0

    Parameters
    ----------
    m : int
       order of the spherical harmonics
    l : int
       degree of the spherical harmonics
    theta : array_like
       angle in :math:`xy` plane (azimuthal)
    cos_phi : array_like
       cos(phi) to angle from :math:`z` axis (polar)
    """
    # Calculate the associated Legendre polynomial
    # Since the real spherical harmonics has slight differences
    # for positive and negative m, we have to implement them individually.
    # Currently this is a re-write of what Inelastica does and a combination of
    # learned lessons from Denchar.
    # As such the choice of these real spherical harmonics is that of Siesta.
    if m == 0:
        return _rspher_harm_fact[l][m] * lpmv(m, l, cos_phi)
    elif m < 0:
        return _rspher_harm_fact[l][m] * (lpmv(m, l, cos_phi) * sin(m * theta))
    return _rspher_harm_fact[l][m] * (lpmv(m, l, cos_phi) * cos(m * theta))


@set_module("sisl")
class Orbital:
    r"""Base class for orbital information.

    The orbital class is still in an experimental stage and will probably evolve over some time.

    Parameters
    ----------
    R :
        maximum radius of interaction.
        In case of a dict the values will be passed to the `radial_minimize_range`
        method.
        Currently allowed arguments are:

        - ``contains``: R will be selected such that the integrated function ``func``
            will contain this percentage of the full integral (determined at ``maxR``
        - ``maxR``: maximum R to search in, default to 100 Ang
        - ``func``: the function that will be integrated and checked for ``contains``

        See examples for details.
        If None the default will be ``{'contains': 0.9999}``.
        If a negative number is passed, it will be converted to ``{'contains':-R}``
        A dictionary will only make sense if the class has the ``_radial`` function
        associated.
    q0 :
        initial charge
    tag :
        user defined tag

    Examples
    --------
    >>> orb = Orbital(1)
    >>> orb_tag = Orbital(2, tag="range=2")
    >>> orb.R == orb_tag.R / 2
    True
    >>> orbq = Orbital(2, 1)
    >>> orbq.q0
    1.

    Optimizing the R range for the radial function integral :math:`\int\mathrm dr radial(r)^2 r ^2`
    >>> R = {
    ...    "contains": 0.9999,
    ...    "func": lambda radial, r: (radial(r) * r)**2,
    ...    "maxR": 100
    ... }
    >>> orb = Orbital(R)

    The default dictionary if none is passed will be:
    ``dict(contains=0.9999, func=lambda radial, r: abs(radial(r)), maxR=100)``
    The optimization problem depends heavily on the ``func`` since the tails are
    important for real-space quantities.

    See also
    --------
    SphericalOrbital : orbitals with a spherical basis set
    AtomicOrbital : specification of n, m, l quantum numbers + a spherical basis set
    HydrogenicOrbital : simplistic orbital model of Hydrogenic-like basis sets
    GTOrbital : Gaussian-type orbitals
    STOrbital : Slater-type orbitals
    """

    __slots__ = ("_R", "_tag", "_q0")

    def __init__(self, R: Optional[Union[float, dict]], q0: float = 0.0, tag: str = ""):
        """Initialize orbital object"""
        # Determine if the orbital has a radial function
        # In which case we can apply the radial discovery
        if R is None:
            R = -0.9999
        if hasattr(self, "_radial"):
            if isinstance(R, dict):
                pass
            elif R < 0:
                # change to a dict
                R = {"contains": -R}

            if isinstance(R, dict):
                R = radial_minimize_range(self._radial, **R)

        elif isinstance(R, dict):
            warn(
                f"{self.__class__.__name__} cannot optimize R without a radial function."
            )
            R = R.get("contains", -0.9999)

        self._R = float(R)
        self._q0 = float(q0)
        self._tag = tag

    def __hash__(self):
        return hash((self._R, self._q0, self._tag))

    @property
    def R(self):
        """Maxmimum radius of orbital"""
        return self._R

    @property
    def q0(self):
        """Initial charge"""
        return self._q0

    @property
    def tag(self):
        """Named tag of orbital"""
        return self._tag

    def __str__(self):
        """A string representation of the object"""
        if self.tag:
            return f"{self.__class__.__name__}{{R: {self.R:.5f}, q0: {self.q0}, tag: {self.tag}}}"
        return f"{self.__class__.__name__}{{R: {self.R:.5f}, q0: {self.q0}}}"

    def __repr__(self):
        if self.tag:
            return f"<{self.__module__}.{self.__class__.__name__} R={self.R:.3f}, q0={self.q0}, tag={self.tag}>"
        return f"<{self.__module__}.{self.__class__.__name__} R={self.R:.3f}, q0={self.q0}>"

    def name(self, tex=False):
        """Return a named specification of the orbital (`tag`)"""
        return self.tag

    def psi(self, r, *args, **kwargs):
        r"""Calculate :math:`\phi(\mathbf r)` for Cartesian coordinates"""
        raise NotImplementedError

    def toSphere(self, center=None):
        """Return a sphere with radius equal to the orbital size

        Returns
        -------
        ~sisl.shape.Sphere
            sphere with a radius equal to the radius of this orbital
        """
        return Sphere(self.R, center)

    def equal(self, other, psi: bool = False, radial: bool = False):
        """Compare two orbitals by comparing their radius, and possibly the radial and psi functions

        When comparing two orbital radius they are considered *equal* with a precision of 1e-4 Ang.

        Parameters
        ----------
        other : Orbital
           comparison orbital
        psi :
           also compare that the full psi are the same
        radial :
           also compare that the radial parts are the same
        """
        if isinstance(other, str):
            # just check for the same name
            return self.name == other

        elif not isinstance(other, Orbital):
            return False

        same = self.tag == other.tag
        same &= close(self.R, other.R, atol=1e-4)
        same &= close(self.q0, other.q0, atol=1e-4)
        if not same:
            # Quick return
            return False

        if same and radial:
            # Ensure they also have the same fill-values
            r = np.linspace(0, self.R + 1, 500)
            same &= np.allclose(self.radial(r), other.radial(r))

        if same and psi:
            xyz = np.linspace(0, self.R * 2, 999).reshape(-1, 3)
            same &= np.allclose(self.psi(xyz), other.psi(xyz))

        return same

    def __eq__(self, other):
        return self.equal(other)

    def toGrid(
        self, precision: float = 0.05, c: float = 1.0, R=None, dtype=np.float64, atom=1
    ):
        """Create a Grid with *only* this orbital wavefunction on it

        Parameters
        ----------
        precision : float, optional
           used separation in the `Grid` between voxels (in Ang)
        c : float or complex, optional
           coefficient for the orbital
        R : float, optional
            box size of the grid (default to the orbital range)
        dtype : numpy.dtype, optional
            the used separation in the `Grid` between voxels
        atom : optional
            atom associated with the grid; either an atom instance or
            something that ``Atom(atom)`` would convert to a proper atom.
        """
        if R is None:
            R = self.R
        if R < 0:
            raise ValueError(
                f"{self.__class__.__name__}.toGrid was unable to create "
                "the orbital grid for plotting, the box size is negative."
            )

        # Since all these things depend on other elements
        # we will simply import them here.
        from sisl.physics.electron import wavefunction

        from .atom import Atom
        from .geometry import Geometry
        from .grid import Grid
        from .lattice import Lattice

        lattice = Lattice(R * 2, origin=[-R] * 3)
        if isinstance(atom, Atom):
            atom = atom.copy(orbitals=self)
        else:
            atom = Atom(atom, self)
        g = Geometry([0] * 3, atom, lattice=lattice)
        G = Grid(precision, dtype=dtype, geometry=g)
        wavefunction(np.full(1, c), G, geometry=g)
        return G

    def __getstate__(self):
        """Return the state of this object"""
        return {"R": self.R, "q0": self.q0, "tag": self.tag}

    def __setstate__(self, d):
        """Re-create the state of this object"""
        self.__init__(d["R"], q0=d["q0"], tag=d["tag"])


RadialFuncT = Callable[[npt.ArrayLike], npt.NDArray]


def radial_minimize_range(
    radial_func: Callable[[RadialFuncT], npt.NDArray],
    contains: float,
    dr: tuple[float, float] = (0.01, 0.0001),
    maxR: float = 100,
    func: Optional[Callable[[RadialFuncT, npt.ArrayLike], npt.NDArray]] = None,
) -> float:
    """Minimize the maximum radius such that the integrated function `radial_func**2*r**3` contains `contains` of the integrand

    Parameters
    ----------
    radial_func : callable
       the function that returns the radial part
    contains : float
       how much of a percentage the squared function should contain @ R
    dr : tuple of float, optional
       the precision of the integral. First number is the coarse integral.
       The second number determines the fine-integral to exactly determine R between
       coarser points.
    maxR : float, optional
       maximally searched ``R``, in case there is no cross-over of the integrand
       containing `contains` in this range a ``-contains`` will be returned to
       signal it could not be found
    func : callable, optional
        function that is evaluated when doing the `contains` check.
        I.e. ``trapz(func(radial_func, r)) >= contains``.
    """
    # Determine the maximum R
    # We should never expect a radial components above
    assert maxR > 0.05, "maxR too small (> 0.05)"
    assert contains > 0, "contains too small (> 0)"
    assert len(dr) == 2, "number of sub-divisions is not 2: dr argument"

    def func_base(func, r):
        # finding the best integral function for locating max
        # R is difficult.
        # For instance the exact integral of a radial function
        # is: (f(r) * r)**2
        # However, locating R that takes 99.99% of the integrand
        # tends to yield a too low R.
        # This is send by evaluating f(R) which tends to be 1% of
        # the maximum f(:). Hence when expanding individual points
        # in the real space grid one finds non-negligeble points
        # that are left out. Hence we cannot limit these integration
        # points.
        # Instead we use the absolute radial function to better capture
        # long tails.
        # Tried functions:
        # 1. f(r)  ->  problematic when f turns negative
        # 2. f(r) ** 2 -> yields somewhat short R
        # 3. f(r) * r -> problematic when f turns negative
        # 4. (f(r) * r)**2 -> yields too short R
        # 5. f(r)**2 * r**3 -> much better
        # 6. abs(f(r)) -> yields a pretty long tail, but should be fine
        return abs(func(r))

    if func is None:
        func = func_base

    def loc(intf, integrand):
        # get index location of the boolean index where
        # all subsequent indices are also of the same type
        # first we find placements below the integrand, and
        # then only select ones above the max placement
        idx = (intf < integrand).nonzero()[0]
        if len(idx) > 0:
            idx = idx.max()
        else:
            idx = 0
        return idx + (intf[idx:] >= integrand).nonzero()[0]

    r = np.arange(0.0, maxR + dr[0] / 2, dr[0])
    f = func(radial_func, r)
    intf = cumulative_trapezoid(f, dx=dr[0], initial=0)
    integrand = intf[-1] * contains

    # we'll accept a containment of 99.99% of the integrand
    loc(intf, integrand)
    idx = loc(intf, integrand)
    if len(idx) > 0 and idx.min() > 0:
        idx = idx.min()

        # in the trapezoid integration each point is half contributed
        # to the previous point and half to the following point.
        # Here intf[idx-1] is the closed integral from 0:r[idx-1]
        idxm_integrand = intf[idx - 1]

        # Preset R
        R = r[idx]

        r = np.arange(R - dr[0], min(R + dr[0] * 2, maxR) + dr[1] / 2, dr[1])
        f = func(radial_func, r)
        intf = cumulative_trapezoid(f, dx=dr[1], initial=0) + idxm_integrand

        # Find minimum R and focus around this point
        idx = loc(intf, integrand)
        if len(idx) > 0:
            R = r[idx.min()]
        return R

    try:
        func_name = radial_func.__class__.__name__
    except AttributeError:
        func_name = radial_func.__name__

    warn(
        f"{func_name} failed to detect a proper radius for integration purposes, retaining R=-{contains}"
    )
    return -contains


def _set_radial(self, *args, **kwargs) -> None:
    r"""Update the internal radial function used as a :math:`f(|\mathbf r|)`

    This can be called in several ways:

          set_radial(r, f)
                which uses ``scipy.interpolate.UnivariateSpline(r, f, k=3, s=0, ext=1, check_finite=False)``
                to define the interpolation function (see `interp` keyword).
                Here the maximum radius of the orbital is the maximum `r` value,
                regardless of ``f(r)`` is zero for smaller `r`.

          set_radial(func)
                which sets the interpolation function directly.
                The maximum orbital range is determined automatically to a precision
                of 0.0001 AA.

    Parameters
    ----------
    r, f : numpy.ndarray
        the radial positions and the radial function values at `r`.
    func : callable
        a function which enables evaluation of the radial function. The function should
        accept a single array and return a single array.
    interp : callable, optional
        When two non-keyword arguments are passed this keyword will be used.
        It is the interpolation function which should return the equivalent of
        `func`. By using this one can define a custom interpolation routine.
        It should accept two arguments, ``interp(r, f)`` and return a callable
        that returns interpolation values.
        See examples for different interpolation routines.

    Examples
    --------
    >>> from scipy import interpolate as interp
    >>> o = SphericalOrbital(1, lambda x:x)
    >>> r = np.linspace(0, 4, 300)
    >>> f = np.exp(-r)
    >>> def i_univariate(r, f):
        ...    return interp.UnivariateSpline(r, f, k=3, s=0, ext=1, check_finite=False)
    >>> def i_interp1d(r, f):
        ...    return interp.interp1d(r, f, kind="cubic", fill_value=(f[0], 0.), bounds_error=False)
    >>> def i_spline(r, f):
        ...    from functools import partial
    ...    tck = interp.splrep(r, f, k=3, s=0)
    ...    return partial(interp.splev, tck=tck, der=0, ext=1)
    >>> R = np.linspace(0, 4, 400)
    >>> o.set_radial(r, f, interp=i_univariate)
    >>> f_univariate = o.radial(R)
    >>> o.set_radial(r, f, interp=i_interp1d)
    >>> f_interp1d = o.radial(R)
    >>> o.set_radial(r, f, interp=i_spline)
    >>> f_spline = o.radial(R)
    >>> np.allclose(f_univariate, f_interp1d)
    True
    >>> np.allclose(f_univariate, f_spline)
    True
    """
    if len(args) == 0:

        def f0(r):
            """Wrapper for returning 0s"""
            return np.zeros_like(r)

        self._radial = f0
        # we cannot set R since it will always give the largest distance

    elif len(args) == 1 and callable(args[0]):
        self._radial = args[0]

    elif len(args) > 1:
        # A radial and function component has been passed
        r = _a.asarrayd(args[0])
        f = _a.asarrayd(args[1])
        # Sort r and f
        idx = np.argsort(r)
        r = r[idx]
        f = f[idx]

        # k = 3 == cubic spline
        # ext = 1 == return zero outside of bounds.
        # s, smoothing factor. If 0, smooth through all points
        # I can see that this function is *much* faster than
        # interp1d, AND it yields same results with these arguments.
        interp = partial(UnivariateSpline, k=3, s=0, ext=1, check_finite=False)
        interp = kwargs.pop("interp", interp)(r, f)

        # this will defer the actual R designation (whether it should be set or not)
        self._radial = interp
    else:
        raise ValueError(
            f"{self.__class__.__name__}.set_radial could not determine the arguments, please correct."
        )


def _radial(self, r, *args, **kwargs) -> np.ndarray:
    r"""Calculate the radial part of spherical orbital :math:`R(\mathbf r)`

    The position `r` is a vector from the origin of this orbital.

    Parameters
    -----------
    r : array_like
       radius from the orbital origin
    *args :
       arguments passed to the radial function
    **args :
       keyword arguments passed to the radial function

    Returns
    -------
    numpy.ndarray
        radial orbital value at point `r`
    """
    r = _a.asarray(r)
    p = _a.zerosd(r.shape)

    # Only calculate where it makes sense, all other points are removed and set to zero
    idx = (r <= self.R).nonzero()

    # Reduce memory immediately
    r = take(r, idx)

    if len(idx) > 0:
        p[idx] = self._radial(r, *args, **kwargs)

    return p


RadialFuncType = Union[
    tuple[npt.ArrayLike, npt.ArrayLike], Callable[[npt.ArrayLike], npt.NDArray]
]


@set_module("sisl")
class SphericalOrbital(Orbital):
    r"""An *arbitrary* orbital class which only contains the harmonical part of the wavefunction  where :math:`\phi(\mathbf r)=f(|\mathbf r|)Y_l^m(\theta,\varphi)`

    Note that in this case the used spherical harmonics is:

    .. math::
        Y^m_l(\theta,\varphi) = (-1)^m\sqrt{\frac{2l+1}{4\pi} \frac{(l-m)!}{(l+m)!}}
             e^{i m \theta} P^m_l(\cos(\varphi))

    The resulting orbital is

    .. math::
        \phi_{lmn}(\mathbf r) = f(|\mathbf r|) Y^m_l(\theta, \varphi)

    where typically :math:`f(|\mathbf r|)\equiv\phi_{ln}(|\mathbf r|)`. The above equation
    clarifies that this class is only intended for each :math:`l`, and that subsequent
    :math:`m` orders may be extracted by altering the spherical harmonic. Also, the quantum
    number :math:`n` is not necessary as that value is implicit in the
    :math:`\phi_{ln}(|\mathbf r|)` function.


    Parameters
    ----------
    l :
       azimuthal quantum number
    rf_or_func :
       radial components as a tuple/list, or the function which can interpolate to any R
       See `set_radial` for details.
    R :
       See `Orbital` for details.
    q0 :
       initial charge
    tag :
       user defined tag
    **kwargs:
       arguments passed directly to ``set_radial(rf_or_func, **kwargs)``

    Attributes
    ----------
    f : func
        interpolation function that returns `f(r)` for a given radius

    Examples
    --------
    >>> from scipy.interpolate import interp1d
    >>> orb = SphericalOrbital(1, (np.arange(10.), np.arange(10.)))
    >>> orb.equal(SphericalOrbital(1, interp1d(np.arange(10.), np.arange(10.),
    ...       fill_value=(0., 0.), kind="cubic", bounds_error=False)))
    True
    """

    # Additional slots (inherited classes retain the same slots)
    __slots__ = ("_l", "_radial")

    def __init__(
        self,
        l: int,
        rf_or_func: Optional[RadialFuncType] = None,
        q0: float = 0.0,
        tag: str = "",
        **kwargs,
    ):
        """Initialize spherical orbital object"""
        self._l = l

        # Set the internal function
        if rf_or_func is None:
            args = []
        elif callable(rf_or_func):
            args = [rf_or_func]
        else:
            args = rf_or_func

        self.set_radial(*args, **kwargs)

        # ensure we pass an R value (default None)
        R = kwargs.get("R")

        # Initialize R and tag through the parent
        # Note that the maximum range of the orbital will be the
        # maximum value in r.
        super().__init__(R, q0, tag)

    @property
    def l(self):
        r""":math:`l` quantum number"""
        return self._l

    def __hash__(self):
        return hash((super(Orbital, self), self._l, self._radial))

    set_radial = _set_radial
    radial = _radial

    def spher(self, theta, phi, m: int = 0, cos_phi: bool = False):
        r"""Calculate the spherical harmonics of this orbital at a given point (in spherical coordinates)

        Parameters
        -----------
        theta : array_like
            azimuthal angle in the :math:`xy` plane (from :math:`x`)
        phi : array_like
            polar angle from :math:`z` axis
        m :
            magnetic quantum number, must be in range ``-self.l <= m <= self.l``
        cos_phi :
            whether `phi` is actually :math:`cos(\phi)` which will be faster because
            `cos` is not necessary to call.

        Returns
        -------
        numpy.ndarray
           spherical harmonics at angles :math:`\theta` and :math:`\phi` and given quantum number `m`
        """
        if cos_phi:
            return _rspherical_harm(m, self.l, theta, phi)
        return _rspherical_harm(m, self.l, theta, cos(phi))

    def psi(self, r, m: int = 0):
        r"""Calculate :math:`\phi(\mathbf r)` at a given point (or more points)

        The position `r` is a vector from the origin of this orbital.

        Parameters
        -----------
        r : array_like of (:, 3)
           vector from the orbital origin
        m :
           magnetic quantum number, must be in range ``-self.l <= m <= self.l``

        Returns
        -------
        numpy.ndarray
            basis function value at point `r`
        """
        r = _a.asarray(r)
        s = r.shape[:-1]
        # Convert to spherical coordinates
        idx, r, theta, phi = cart2spher(r, theta=m != 0, cos_phi=True, maxR=self.R)
        p = _a.zerosd(s)
        if len(idx) > 0:
            p[idx] = self.psi_spher(r, theta, phi, m, cos_phi=True)
            # Reduce memory immediately
            del idx, r, theta, phi
        p.shape = s
        return p

    def psi_spher(self, r, theta, phi, m: int = 0, cos_phi: bool = False):
        r"""Calculate :math:`\phi(|\mathbf r|, \theta, \phi)` at a given point (in spherical coordinates)

        This is equivalent to `psi` however, the input is given in spherical coordinates.

        Parameters
        -----------
        r : array_like
           the radius from the orbital origin
        theta : array_like
           azimuthal angle in the :math:`xy` plane (from :math:`x`)
        phi : array_like
           polar angle from :math:`z` axis
        m :
           magnetic quantum number, must be in range ``-self.l <= m <= self.l``
        cos_phi :
           whether `phi` is actually :math:`cos(\phi)` which will be faster because
           `cos` is not necessary to call.

        Returns
        -------
        numpy.ndarray
            basis function value at point `r`
        """
        return self.radial(r) * self.spher(theta, phi, m, cos_phi)

    def equal(self, other, psi: bool = False, radial: bool = False):
        """Compare two orbitals by comparing their radius, and possibly the radial and psi functions

        Parameters
        ----------
        other : Orbital
           comparison orbital
        psi : bool, optional
           also compare that the full psi are the same
        radial : bool, optional
           also compare that the radial parts are the same
        """
        same = super().equal(other, psi, radial)
        if not same:
            return False
        if isinstance(other, SphericalOrbital):
            same &= self.l == other.l
        return same

    def __str__(self):
        """A string representation of the object"""
        if self.tag:
            return f"{self.__class__.__name__}{{l: {self.l}, R: {self.R}, q0: {self.q0}, tag: {self.tag}}}"
        return f"{self.__class__.__name__}{{l: {self.l}, R: {self.R}, q0: {self.q0}}}"

    def __repr__(self):
        if self.tag:
            return f"<{self.__module__}.{self.__class__.__name__} l={self.l}, R={self.R:.3f}, q0={self.q0}, tag={self.tag}>"
        return f"<{self.__module__}.{self.__class__.__name__} l={self.l}, R={self.R:.3f}, q0={self.q0}>"

    def toAtomicOrbital(
        self,
        m=None,
        n: Optional[int] = None,
        zeta: int = 1,
        P: bool = False,
        q0: Optional[float] = None,
    ):
        r"""Create a list of `AtomicOrbital` objects

        This defaults to create a list of `AtomicOrbital` objects for every `m` (for m in -l:l).
        One may optionally specify the sub-set of `m` to retrieve.

        Parameters
        ----------
        m : int or list or None
           if ``None`` it defaults to ``-l:l``, else only for the requested `m`
        zeta :
           the specified :math:`\zeta`-shell
        n :
           specify the :math:`n` quantum number
        P :
           whether the orbitals are polarized.
        q0 :
           the initial charge per orbital, initially :math:`q_0 / (2l+1)` with :math:`q_0` from this object

        Returns
        -------
        AtomicOrbital : for passed `m` an atomic orbital will be returned
        list of AtomicOrbital : for each :math:`m\in[-l;l]` an atomic orbital will be returned in the list
        """
        # Initial charge
        if q0 is None:
            q0 = self.q0 / (2 * self.l + 1)
        if m is None:
            m = range(-self.l, self.l + 1)
        elif isinstance(m, Integral):
            return AtomicOrbital(
                n=n, l=self.l, m=m, zeta=zeta, P=P, spherical=self, q0=q0, R=self.R
            )
        return [
            AtomicOrbital(
                n=n, l=self.l, m=mm, zeta=zeta, P=P, spherical=self, q0=q0, R=self.R
            )
            for mm in m
        ]

    def __getstate__(self):
        """Return the state of this object"""
        # A function is not necessarily pickable, so we store interpolated
        # data which *should* ensure the correct pickable state (to close agreement)
        r = np.linspace(0, self.R, 1000)
        f = self.radial(r)
        return {"l": self.l, "r": r, "f": f, "q0": self.q0, "tag": self.tag}

    def __setstate__(self, d):
        """Re-create the state of this object"""
        self.__init__(d["l"], (d["r"], d["f"]), q0=d["q0"], tag=d["tag"])


@set_module("sisl")
class AtomicOrbital(Orbital):
    r""" A projected atomic orbital consisting of real harmonics

    The `AtomicOrbital` is a specification of the `SphericalOrbital` by
    assigning the magnetic quantum number :math:`m` to the object.

    `AtomicOrbital` should always be preferred over the
    `SphericalOrbital` because it explicitly contains *all* quantum numbers.

    The atomic orbital has a radial part defined by an external function; this
    is then expanded using spherical harmonics

    .. math::
        Y^m_l(\theta,\varphi) &= (-1)^m\sqrt{\frac{2l+1}{4\pi} \frac{(l-m)!}{(l+m)!}}
             e^{i m \theta} P^m_l(\cos(\varphi))
        \\
                \phi_{lmn}(\mathbf r) &= R(|\mathbf r|) Y^m_l(\theta, \varphi)

    where the function :math:`R(|\mathbf r|)` is user-defined.

    Parameters
    ----------
    *args : list of arguments
        list of arguments can be in different input options
    R :
       See `Orbital` for details.
    q0 :
        initial charge
    tag :
        user defined tag

    Examples
    --------
    >>> r = np.linspace(0, 5, 50)
    >>> f = np.exp(-r)
    >>> #                    n, l, m, [zeta, [P]]
    >>> orb1 = AtomicOrbital(2, 1, 0, 1, (r, f))
    >>> orb2 = AtomicOrbital(n=2, l=1, m=0, zeta=1, (r, f))
    >>> orb3 = AtomicOrbital("2pzZ", (r, f))
    >>> orb4 = AtomicOrbital("2pzZ1", (r, f))
    >>> orb5 = AtomicOrbital("pz", (r, f))
    >>> orb2 == orb3
    True
    >>> orb2 == orb4
    True
    >>> orb2 == orb5
    True
    """

    # All of these follow standard notation:
    #   n = principal quantum number
    #   l = azimuthal quantum number
    #   m = magnetic quantum number
    #   Z = zeta shell
    #   P = polarization shell or not
    # orb is the SphericalOrbital class that retains the radial
    # grid and enables to calculate psi(r)
    __slots__ = ("_n", "_l", "_m", "_zeta", "_P", "_orb")

    def __init__(self, *args, **kwargs):
        """Initialize atomic orbital object"""

        # Ensure args is a list (to be able to pop)
        args = list(args)
        self._orb = None

        # Extract shell information
        n = kwargs.get("n", None)
        l = kwargs.get("l", None)
        m = kwargs.get("m", None)
        zeta = kwargs.get("zeta", 1)
        P = kwargs.get("P", False)

        if len(args) > 0:
            if isinstance(args[0], str):
                # String specification of the atomic orbital
                s = args.pop(0)

                _n = {"s": 1, "p": 2, "d": 3, "f": 4, "g": 5}
                _l = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}
                _m = {
                    "s": 0,
                    "pz": 0,
                    "px": 1,
                    "py": -1,
                    "dxy": -2,
                    "dyz": -1,
                    "dz2": 0,
                    "dxz": 1,
                    "dx2-y2": 2,
                    "fy(3x2-y2)": -3,
                    "fxyz": -2,
                    "fz2y": -1,
                    "fyz2": -1,
                    "fz3": 0,
                    "fz2x": 1,
                    "fxz2": 1,
                    "fz(x2-y2)": 2,
                    "fx(x2-3y2)": 3,
                    "gxy(x2-y2)": -4,
                    "gyx(x2-y2)": -4,
                    "gzy(3x2-y2)": -3,
                    "gyz(3x2-y2)": -3,
                    "gz2xy": -2,
                    "gxyz2": -2,
                    "gyxz2": -2,
                    "gz3y": -1,
                    "gyz3": -1,
                    "gz4": 0,
                    "gz3x": 1,
                    "gxz3": 1,
                    "gz2(x2-y2)": 2,
                    "gzx(x2-3y2)": 3,
                    "gxz(x2-3y2)": 3,
                    "gx4+y4": 4,
                }

                # First remove a P for polarization
                P = "P" in s
                s = s.replace("P", "")

                # Try and figure out the input
                #   2s => n=2, l=0, m=0, z=1, P=False
                #   2sZ2P => n=2, l=0, m=0, z=2, P=True
                #   2pxZ2P => n=2, l=0, m=0, z=2, P=True
                # By default a non-"n" specification takes the lowest value allowed
                #    s => n=1
                #    p => n=2
                #    ...
                try:
                    n = int(s[0])
                    # Remove n specification
                    s = s[1:]
                except Exception:
                    n = _n.get(s[0], n)

                # Get l
                l = _l.get(s[0], l)

                # Get number of zeta shell
                iZ = s.find("Z")
                if iZ >= 0:
                    # Currently we know that we are limited to 9 zeta shells.
                    # However, for now we assume this is enough (could easily
                    # be extended by a reg-exp)
                    try:
                        zeta = int(s[iZ + 1])
                        # Remove Z + int
                        s = s[:iZ] + s[iZ + 2 :]
                    except Exception:
                        zeta = 1
                        s = s[:iZ] + s[iZ + 1 :]

                # We should be left with m specification
                m = _m.get(s, m)

            else:
                # Arguments *have* to be
                # n, l, [m (only for l>0)] [, zeta [, P]]
                if n is None and len(args) > 0:
                    n = args.pop(0)
                if l is None and len(args) > 0:
                    l = args.pop(0)
                if m is None and len(args) > 0:
                    m = args.pop(0)

                # Now we need to figure out if they are shell
                # information or radial functions
                if len(args) > 0:
                    if isinstance(args[0], Integral):
                        zeta = args.pop(0)
                if len(args) > 0:
                    if isinstance(args[0], bool):
                        P = args.pop(0)

        if l is None:
            raise ValueError(f"{self.__class__.__name__} l is not defined")

        # Still if n is None, we assign the default (lowest) quantum number
        if n is None:
            n = l + 1
        # Still if m is None, we assign the default value of 0
        if m is None:
            m = 0

        # Copy over information
        self._n = n
        self._l = l
        self._m = m
        self._zeta = zeta
        self._P = P

        if n <= 0:
            raise ValueError(f"{self.__class__.__name__} n must be >= 1")

        if zeta <= 0:
            raise ValueError(f"{self.__class__.__name__} zeta must be >= 1")

        if self.l >= len(_rspher_harm_fact):
            raise ValueError(
                f"{self.__class__.__name__} does not implement shells l>={len(_rspher_harm_fact)}!"
            )
        if abs(self.m) > self.l:
            raise ValueError(f"{self.__class__.__name__} requires |m| <= l.")

        # Now we should figure out how the spherical orbital
        # has been passed.
        # There are two options:
        #  1. The radial function is passed as two arrays: r, f
        #  2. The SphericalOrbital-class is passed which already contains
        #     the relevant information.
        # Figure out if it is a sphericalorbital
        if len(args) > 0:
            s = args.pop(0)
            if "spherical" in kwargs:
                raise ValueError(
                    f"{self.__class__.__name__} multiple values for the spherical "
                    "orbital is present, 1) argument, 2) spherical=. Only supply one of them."
                )

        else:
            # in case the class has its own radial implementation, we might as well rely on that one
            s = kwargs.get("spherical", getattr(self, "_radial", None))

        # Get the radius requested
        R = kwargs.get("R")
        q0 = kwargs.get("q0", 0.0)

        if s is None:
            self._orb = Orbital(R, q0=q0)
        elif isinstance(s, Orbital):
            self._orb = s
        else:
            # Determine the correct R if requested a sub-set
            self._orb = SphericalOrbital(l, s, q0=q0, R=R)

        if isinstance(self._orb, SphericalOrbital):
            if self._orb.l != self.l:
                raise ValueError(
                    f"{self.__class__.__name__} got a spherical argument with l={self._orb.l} which is different from this objects l={self.l}."
                )

        super().__init__(self._orb.R, q0=q0, tag=kwargs.get("tag", ""))

    def __hash__(self):
        return hash(
            (
                super(Orbital, self),
                self._l,
                self._n,
                self._m,
                self._zeta,
                self._P,
                self._orb,
            )
        )

    @property
    def n(self):
        r""":math:`n` shell"""
        return self._n

    @property
    def l(self):
        r""":math:`l` quantum number"""
        return self._l

    @property
    def m(self):
        r""":math:`m` quantum number"""
        return self._m

    @property
    def zeta(self):
        r""":math:`\zeta` shell"""
        return self._zeta

    @property
    def P(self):
        r"""Whether this is polarized shell or not"""
        return self._P

    @property
    def orb(self):
        r"""Orbital with radial part"""
        return self._orb

    def equal(self, other, psi: bool = False, radial: bool = False):
        """Compare two orbitals by comparing their radius, and possibly the radial and psi functions

        Parameters
        ----------
        other : Orbital
           comparison orbital
        psi :
           also compare that the full psi are the same
        radial :
           also compare that the radial parts are the same
        """
        if isinstance(other, AtomicOrbital):
            same = self.orb.equal(other.orb, psi, radial)
            same &= self.n == other.n
            same &= self.l == other.l
            same &= self.m == other.m
            same &= self.zeta == other.zeta
            same &= self.P == other.P
        elif isinstance(other, Orbital):
            same = self.orb.equal(other)
        else:
            return False
        return same

    def name(self, tex=False):
        """Return named specification of the atomic orbital"""
        if tex:
            name = "{}{}".format(self.n, "spdfghij"[self.l])
            if self.l == 1:
                name += ("_y", "_z", "_x")[self.m + 1]
            elif self.l == 2:
                name += ("_{xy}", "_{yz}", "_{z^2}", "_{xz}", "_{x^2-y^2}")[self.m + 2]
            elif self.l == 3:
                name += (
                    "_{y(3x^2-y^2)}",
                    "_{xyz}",
                    "_{z^2y}",
                    "_{z^3}",
                    "_{z^2x}",
                    "_{z(x^2-y^2)}",
                    "_{x(x^2-3y^2)}",
                )[self.m + 3]
            elif self.l == 4:
                name += (
                    "_{_{xy(x^2-y^2)}}",
                    "_{zy(3x^2-y^2)}",
                    "_{z^2xy}",
                    "_{z^3y}",
                    "_{z^4}",
                    "_{z^3x}",
                    "_{z^2(x^2-y^2)}",
                    "_{zx(x^2-3y^2)}",
                    "_{x^4+y^4}",
                )[self.m + 4]
            elif self.l >= 5:
                name = f"{name}_{{m={self.m}}}"
            if self.P:
                return name + rf"\zeta^{self.zeta}\mathrm{{P}}"
            return name + rf"\zeta^{self.zeta}"
        name = "{}{}".format(self.n, "spdfghij"[self.l])
        if self.l == 1:
            name += ("y", "z", "x")[self.m + 1]
        elif self.l == 2:
            name += ("xy", "yz", "z2", "xz", "x2-y2")[self.m + 2]
        elif self.l == 3:
            name += ("y(3x2-y2)", "xyz", "z2y", "z3", "z2x", "z(x2-y2)", "x(x2-3y2)")[
                self.m + 3
            ]
        elif self.l == 4:
            name += (
                "xy(x2-y2)",
                "zy(3x2-y2)",
                "z2xy",
                "z3y",
                "z4",
                "z3x",
                "z2(x2-y2)",
                "zx(x2-3y2)",
                "x4+y4",
            )[self.m + 4]
        elif self.l >= 5:
            name = f"{name}(m={self.m})"
        if self.P:
            return name + f"Z{self.zeta}P"
        return name + f"Z{self.zeta}"

    def __str__(self):
        """A string representation of the object"""
        if self.tag:
            return f"{self.__class__.__name__}{{{self.name()}, q0: {self.q0}, tag: {self.tag}, {self.orb!s}}}"
        return (
            f"{self.__class__.__name__}{{{self.name()}, q0: {self.q0}, {self.orb!s}}}"
        )

    def __repr__(self):
        if self.tag:
            return f"<{self.__module__}.{self.__class__.__name__} {self.name()} q0={self.q0}, tag={self.tag}>"
        return (
            f"<{self.__module__}.{self.__class__.__name__} {self.name()} q0={self.q0}>"
        )

    def set_radial(self, *args, **kwargs):
        r"""Update the internal radial function used as a :math:`f(|\mathbf r|)`

        See `SphericalOrbital.set_radial` where these arguments are passed to.
        """
        return self.orb.set_radial(*args, **kwargs)

    def radial(self, r, *args, **kwargs):
        r"""Calculate the radial part of the wavefunction :math:`f(\mathbf r)`

        The position `r` is a vector from the origin of this orbital.

        Parameters
        -----------
        r : array_like
           radius from the orbital origin

        Returns
        -------
        numpy.ndarray
            radial orbital value at point `r`
        """
        return self.orb.radial(r, *args, **kwargs)

    def psi(self, r):
        r"""Calculate :math:`\phi(\mathbf r)` at a given point (or more points)

        The position `r` is a vector from the origin of this orbital.

        Parameters
        -----------
        r : array_like
           the vector from the orbital origin

        Returns
        -------
        numpy.ndarray
             basis function value at point `r`
        """
        return self.orb.psi(r, self.m)

    def spher(self, theta, phi, cos_phi: bool = False):
        r"""Calculate the spherical harmonics of this orbital at a given point (in spherical coordinates)

        Parameters
        -----------
        theta : array_like
           azimuthal angle in the :math:`xy` plane (from :math:`x`)
        phi : array_like
           polar angle from :math:`z` axis
        cos_phi :
           whether `phi` is actually :math:`cos(\phi)` which will be faster because
           `cos` is not necessary to call.

        Returns
        -------
        numpy.ndarray
            spherical harmonics at angles :math:`\theta` and :math:`\phi`
        """
        return self.orb.spher(theta, phi, self.m, cos_phi)

    def psi_spher(self, r, theta, phi, cos_phi: bool = False):
        r"""Calculate :math:`\phi(|\mathbf r|, \theta, \phi)` at a given point (in spherical coordinates)

        This is equivalent to `psi` however, the input is given in spherical coordinates.

        Parameters
        -----------
        r : array_like
           the radius from the orbital origin
        theta : array_like
           azimuthal angle in the :math:`xy` plane (from :math:`x`)
        phi : array_like
           polar angle from :math:`z` axis
        cos_phi :
           whether `phi` is actually :math:`cos(\phi)` which will be faster because
           `cos` is not necessary to call.

        Returns
        -------
        numpy.ndarray
             basis function value at point `r`
        """
        return self.orb.psi_spher(r, theta, phi, self.m, cos_phi)

    def __getstate__(self):
        """Return the state of this object"""
        # A function is not necessarily pickable, so we store interpolated
        # data which *should* ensure the correct pickable state (to close agreement)
        try:
            # this will tricker the AttributeError
            # before we create the data-array
            r = np.linspace(0, self.R, 1000)
            f = self.radial(r)
        except AttributeError:
            r, f = None, None
        return {"name": self.name(), "r": r, "f": f, "q0": self.q0, "tag": self.tag}

    def __setstate__(self, d):
        """Re-create the state of this object"""
        if d["r"] is None:
            self.__init__(d["name"], q0=d["q0"], tag=d["tag"])
        else:
            self.__init__(d["name"], (d["r"], d["f"]), q0=d["q0"], tag=d["tag"])


@set_module("sisl")
class HydrogenicOrbital(AtomicOrbital):
    r""" A hydrogen-like atomic orbital defined by an effective atomic number Z in addition to the usual quantum numbers (n, l, m).

    A hydrogenic atom (Hydrogen-like) is an atom with a single valence electron.

    The returned orbital is properly normalized, see [HydrogenicO]_ for details.

    The orbital has the familiar spherical shape

    .. math::
        Y^m_l(\theta,\varphi) &= (-1)^m\sqrt{\frac{2l+1}{4\pi} \frac{(l-m)!}{(l+m)!}}
             e^{i m \theta} P^m_l(\cos(\varphi))
        \\
        \phi_{lmn}(\mathbf r) &= R_{nl}(|\mathbf r|) Y^m_l(\theta, \varphi)
        \\
        R_{nl}(|\mathbf r|) &= -\sqrt{\big(\frac{2Z}{na_0}\big)^3 \frac{(n-l-1)!}{2n(n+l)!}}
           e^{-Zr/(na_0)} \big( \frac{2Zr}{na_0} \big)^l L_{n-l-1}^{(2l+1)}
           \big( \frac{2Zr}{na_0} \big)

    With :math:`L_{n-l-1}^{(2l+1)}` is the generalized Laguerre polynomials.


    References
    ----------
    .. [HydrogenicO] https://en.wikipedia.org/wiki/Hydrogen-like_atom


    Parameters
    ----------
    n :
        principal quantum number
    l :
        angular momentum quantum number
    m :
        magnetic quantum number
    Z :
        effective atomic number
    **kwargs :
        See `Orbital` for details.

    Examples
    --------
    >>> carbon_pz = HydrogenicOrbital(2, 1, 0, 3.2)

    """

    def __init__(self, n: int, l: int, m: int, Z: float, **kwargs):
        self._Z = Z

        Helper = namedtuple("Helper", ["Z", "prefactor"])
        z = 2 * Z / (n * a0("Ang"))
        pref = (z**3 * factorial(n - l - 1) / (2 * n * factorial(n + l))) ** 0.5
        self._radial_helper = Helper(z, pref)

        super().__init__(n, l, m, **kwargs)

    def _radial(self, r):
        r"""Radial functional for the Hydrogenic orbital"""
        H = self._radial_helper
        n = self.n
        l = self.l
        zr = H.Z * r
        L = H.prefactor * eval_genlaguerre(n - l - 1, 2 * l + 1, zr)
        return np.exp(-zr * 0.5) * zr**l * L

    def __getstate__(self):
        """Return the state of this object"""
        return {
            "n": self.n,
            "l": self.l,
            "m": self.m,
            "Z": self._Z,
            "R": self.R,
            "q0": self.q0,
            "tag": self.tag,
        }

    def __setstate__(self, d):
        """Re-create the state of this object"""
        self.__init__(
            d["n"], d["l"], d["m"], d["Z"], R=d["R"], q0=d["q0"], tag=d["tag"]
        )


class _ExponentialOrbital(Orbital):
    r"""Inheritable class for different exponential spherical orbitals

    All exponential spherical orbitals are defined using:

    .. math::
        Y^m_l(\theta,\varphi) = (-1)^m\sqrt{\frac{2l+1}{4\pi} \frac{(l-m)!}{(l+m)!}}
             e^{i m \theta} P^m_l(\cos(\varphi))

    The resulting orbital is

    .. math::
        \phi_{lmn}(\mathbf r) = R_l(|\mathbf r|) Y^m_l(\theta, \varphi)

    And :math:`R_l` is some exponential function with suitable parameters
    that are to be defined in the subclass.
    """

    __slots__ = ("_n", "_l", "_m", "_alpha", "_coeff")

    def __init__(self, *args, **kwargs):
        # Ensure args is a list (to be able to pop)
        args = list(args)

        # Extract shell information
        n = kwargs.pop("n", None)
        l = kwargs.pop("l", None)
        m = kwargs.pop("m", None)
        alpha = kwargs.pop("alpha", None)
        coeff = kwargs.pop("coeff", None)

        # Arguments *have* to be
        # n, l, [m (only for l>0)], alpha, coeff

        if n is None and len(args) > 0:
            n = args.pop(0)

        if l is None and len(args) > 0:
            l = args.pop(0)
        if l is None:
            raise ValueError(f"{self.__class__.__name__} l is not defined")

        if m is None and len(args) > 0 and l > 0:
            m = args.pop(0)
        if alpha is None and len(args) > 0:
            alpha = args.pop(0)
        if coeff is None and len(args) > 0:
            coeff = args.pop(0)

        if m is None:
            # default to 0
            m = 0

        if n is None:
            n = l + 1

        if n <= 0:
            raise ValueError(f"{self.__class__.__name__} n must be >= 1")

        if coeff is None:
            raise ValueError(f"{self.__class__.__name__} coeff is not defined")

        if alpha is None:
            raise ValueError(f"{self.__class__.__name__} alpha is not defined")

        # Copy over information
        self._n = n
        self._l = l
        self._m = m
        if isinstance(alpha, Real):
            alpha = (alpha,)
        self._alpha = tuple(alpha)

        if isinstance(coeff, Real):
            coeff = (coeff,)
        self._coeff = tuple(coeff)

        assert len(self.alpha) == len(
            self.coeff
        ), "Contraction factors and exponents needs to have same length"

        if self.l >= len(_rspher_harm_fact):
            raise ValueError(
                f"{self.__class__.__name__} does not implement shells l>={len(_rspher_harm_fact)}!"
            )
        if abs(self.m) > self.l:
            raise ValueError(f"{self.__class__.__name__} requires |m| <= l.")

        # update R in case the user did not specify it
        R = kwargs.pop("R", None)
        super().__init__(*args, R=R, **kwargs)

    def __str__(self):
        """A string representation of the object"""
        if self.tag:
            s = f"{self.__class__.__name__}{{n: {self.n}, l: {self.l}, m: {self.m}, R: {self.R}, q0: {self.q0}, tag: {self.tag}"
        else:
            s = f"{self.__class__.__name__}{{n: {self.n}, l: {self.l}, m: {self.m}, R: {self.R}, q0: {self.q0}"
        orbs = ",\n c, a:".join(
            [f"{c:.4f} , {a:.5f}" for c, a in zip(self.alpha, self.coeff)]
        )
        return f"{s}{orbs}\n}}"

    def __repr__(self):
        if self.tag:
            return f"<{self.__module__}.{self.__class__.__name__} n={self.n}, l={self.l}, m={self.m}, no={len(self.alpha)}, R={self.R:.3f}, q0={self.q0}, tag={self.tag}>"
        return f"<{self.__module__}.{self.__class__.__name__} n={self.n}, l={self.l}, m={self.m}, no={len(self.alpha)}, R={self.R:.3f}, q0={self.q0}>"

    def __hash__(self):
        return hash(
            (super(Orbital, self), self.n, self.l, self.m, self.coeff, self.alpha)
        )

    @property
    def n(self):
        r""":math:`n` quantum number"""
        return self._n

    @property
    def l(self):
        r""":math:`l` quantum number"""
        return self._l

    @property
    def m(self):
        r""":math:`m` quantum number"""
        return self._m

    @property
    def alpha(self):
        r""":math:`\alpha` factors"""
        return self._alpha

    @property
    def coeff(self):
        r""":math:`c` contraction factors"""
        return self._coeff

    def psi(self, r):
        r"""Calculate :math:`\phi(\mathbf r)` at a given point (or more points)

        The position `r` is a vector from the origin of this orbital.

        Parameters
        -----------
        r : array_like
           the vector from the orbital origin

        Returns
        -------
        numpy.ndarray
             basis function value at point `r`
        """
        r = _a.asarray(r)
        s = r.shape[:-1]
        # Convert to spherical coordinates
        idx, r, theta, phi = cart2spher(r, theta=self.m != 0, cos_phi=True, maxR=self.R)
        p = _a.zerosd(s)
        if len(idx) > 0:
            p[idx] = self.psi_spher(r, theta, phi, cos_phi=True)
            # Reduce memory immediately
            del idx, r, theta, phi
        p.shape = s
        return p

    def spher(self, theta, phi, cos_phi: bool = False):
        r"""Calculate the spherical harmonics of this orbital at a given point (in spherical coordinates)

        Parameters
        -----------
        theta : array_like
           azimuthal angle in the :math:`xy` plane (from :math:`x`)
        phi : array_like
           polar angle from :math:`z` axis
        cos_phi :
           whether `phi` is actually :math:`cos(\phi)` which will be faster because
           `cos` is not necessary to call.

        Returns
        -------
        numpy.ndarray
            spherical harmonics at angles :math:`\theta` and :math:`\phi`
        """
        if cos_phi:
            return _rspherical_harm(self.m, self.l, theta, phi)
        return _rspherical_harm(self.m, self.l, theta, cos(phi))

    def psi_spher(self, r, theta, phi, cos_phi: bool = False):
        r"""Calculate :math:`\phi(|\mathbf r|, \theta, \phi)` at a given point (in spherical coordinates)

        This is equivalent to `psi` however, the input is given in spherical coordinates.

        Parameters
        -----------
        r : array_like
           the radius from the orbital origin
        theta : array_like
           azimuthal angle in the :math:`xy` plane (from :math:`x`)
        phi : array_like
           polar angle from :math:`z` axis
        cos_phi :
           whether `phi` is actually :math:`cos(\phi)` which will be faster because
           `cos` is not necessary to call.

        Returns
        -------
        numpy.ndarray
             basis function value at point `r`
        """
        return self.radial(r) * self.spher(theta, phi, cos_phi)


class GTOrbital(_ExponentialOrbital):
    r""" Gaussian type orbital

    The `GTOrbital` uses contraction factors and coefficients.

    The Gaussian type orbital consists of a gaussian radial part and a spherical
    harmonic part that only depends on angles.

    .. math::
        Y^m_l(\theta,\varphi) &= (-1)^m\sqrt{\frac{2l+1}{4\pi} \frac{(l-m)!}{(l+m)!}}
             e^{i m \theta} P^m_l(\cos(\varphi))
        \\
        \phi_{lmn}(\mathbf r) &= R_l(|\mathbf r|) Y^m_l(\theta, \varphi)
        \\
        R_l(|\mathbf r|) &= \sum c_i e^{-\alpha_i r^2}

    Notes
    -----
    This class is opted for significant changes based on user feedback. If you use it,
    please give feedback.

    Parameters
    ----------
    n : int, optional
       principal quantum number, default to ``l + 1``
    l : int
       azimuthal quantum number
    m : int, optional for l == 0
       magnetic quantum number
    alpha : float or array_like
       coefficients for the exponential (in 1/Ang^2)
       Generally the coefficients are given in atomic units, so
       a conversion from online tables is necessary.
    coeff : float or array_like
       contraction factors
    R :
        See `Orbital` for details.
    q0 : float, optional
        initial charge
    tag : str, optional
        user defined tag
    """

    __slots__ = ()

    radial = _radial

    def _radial(self, r):
        r"""Radial function"""
        r2 = np.square(r)
        coeff = self.coeff
        alpha = self.alpha
        v = coeff[0] * np.exp(-alpha[0] * r2)
        for c, a in zip(coeff[1:], alpha[1:]):
            v += c * np.exp(-a * r2)
        if self.l == 0:
            return v
        return r**self.l * v


class STOrbital(_ExponentialOrbital):
    r""" Slater type orbital

    The `STOrbital` uses contraction factors and coefficients.

    The Slater type orbital consists of an exponential radial part and a spherical
    harmonic part that only depends on angles.

    .. math::
        Y^m_l(\theta,\varphi) &= (-1)^m\sqrt{\frac{2l+1}{4\pi} \frac{(l-m)!}{(l+m)!}}
             e^{i m \theta} P^m_l(\cos(\varphi))
        \\
        \phi_{lmn}(\mathbf r) &= R_n(|\mathbf r|) Y^m_l(\theta, \varphi)
        \\
        R_n(|\mathbf r|) &= r^{n-1} \sum c_i e^{-\alpha_i r}

    Notes
    -----
    This class is opted for significant changes based on user feedback. If you use it,
    please give feedback.

    Parameters
    ----------
    n : int
       principal quantum number
    l : int
       azimuthal quantum number
    m : int, optional for l == 0
       magnetic quantum number
    alpha : float or array_like
       coefficients for the exponential (in 1/Ang)
       Generally the coefficients are given in atomic units, so
       a conversion from online tables is necessary.
    coeff : float or array_like
       contraction factors
    R :
        See `Orbital` for details.
    q0 : float, optional
        initial charge
    tag : str, optional
        user defined tag
    """

    __slots__ = ()

    radial = _radial

    def _radial(self, r):
        r"""Radial function"""
        coeff = self.coeff
        alpha = self.alpha
        v = coeff[0] * np.exp(-alpha[0] * r)
        for c, a in zip(coeff[1:], alpha[1:]):
            v += c * np.exp(-a * r)
        if self.n == 1:
            return v
        return r ** (self.n - 1) * v
