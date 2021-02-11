from functools import partial
from numbers import Integral
from math import pi
from math import sqrt as msqrt
from math import factorial as fact

import numpy as np
from numpy import cos, sin
from numpy import take, sqrt, square
from scipy.special import lpmv
from scipy.interpolate import UnivariateSpline

from ._internal import set_module
from . import _plot as plt
from . import _array as _a
from .messages import deprecate, deprecate_method
from .shape import Sphere
from .utils.mathematics import cart2spher


__all__ = ['Orbital', 'SphericalOrbital', 'AtomicOrbital']


# Create the factor table for the real spherical harmonics
def _rfact(l, m):
    pi4 = 4 * pi
    if m == 0:
        return msqrt((2*l + 1)/pi4)
    elif m < 0:
        return -msqrt(2*(2*l + 1)/pi4 * fact(l-m)/fact(l+m)) * (-1) ** m
    return msqrt(2*(2*l + 1)/pi4 * fact(l-m)/fact(l+m))

# This is a list of dict
#  [0]{0} is l==0, m==0
#  [1]{-1} is l==1, m==-1
#  [1]{1} is l==1, m==1
# and so on.
# Calculate it up to l == 5 which is the h shell
_rspher_harm_fact = [{m: _rfact(l, m) for m in range(-l, l+1)} for l in range(6)]
# Clean-up
del _rfact


def _rspherical_harm(m, l, theta, cos_phi):
    r""" Calculates the real spherical harmonics using :math:`Y_l^m(\theta, \varphi)` with :math:`\mathbf R\to \{r, \theta, \varphi\}`.

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
       angle in :math:`x-y` plane (azimuthal)
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
        return _rspher_harm_fact[l][m] * (lpmv(m, l, cos_phi) * sin(m*theta))
    return _rspher_harm_fact[l][m] * (lpmv(m, l, cos_phi) * cos(m*theta))


@set_module("sisl")
class Orbital:
    """ Base class for orbital information.

    The orbital class is still in an experimental stage and will probably evolve over some time.

    Parameters
    ----------
    R : float
        maximum radius
    q0 : float, optional
        initial charge
    tag : str, optional
        user defined tag

    Examples
    --------
    >>> orb = Orbital(1)
    >>> orb_tag = Orbital(2, tag='range=2')
    >>> orb.R == orb_tag.R / 2
    True
    >>> orbq = Orbital(2, 1)
    >>> orbq.q0
    1.
    """
    __slots__ = ['_R', '_tag', '_q0']

    def __init__(self, R, q0=0., tag=''):
        """ Initialize orbital object """
        self._R = float(R)
        self._q0 = float(q0)
        self._tag = tag

    @property
    def R(self):
        """ Maxmimum radius of orbital """
        return self._R

    @property
    def q0(self):
        """ Initial charge """
        return self._q0

    @property
    def tag(self):
        """ Named tag of orbital """
        return self._tag

    def __str__(self):
        """ A string representation of the object """
        if len(self.tag) > 0:
            return f'{self.__class__.__name__}{{R: {self.R:.5f}, q0: {self.q0}, tag: {self.tag}}}'
        return f'{self.__class__.__name__}{{R: {self.R:.5f}, q0: {self.q0}}}'

    def __repr__(self):
        if self.tag:
            return f"<{self.__module__}.{self.__class__.__name__} R={self.R:.3f}, q0={self.q0}, tag={self.tag}>"
        return f"<{self.__module__}.{self.__class__.__name__} R={self.R:.3f}, q0={self.q0}>"

    def name(self, tex=False):
        """ Return a named specification of the orbital (`tag`) """
        return self.tag

    def toSphere(self, center=None):
        """ Return a sphere with radius equal to the orbital size

        Returns
        -------
        ~sisl.shape.Sphere
            sphere with a radius equal to the radius of this orbital
        """
        return Sphere(self.R, center)

    def equal(self, other, psi=False, radial=False):
        """ Compare two orbitals by comparing their radius, and possibly the radial and psi functions

        When comparing two orbital radius they are considered *equal* with a precision of 1e-4 Ang.

        Parameters
        ----------
        other : Orbital
           comparison orbital
        psi : bool, optional
           also compare that the full psi are the same
        radial : bool, optional
           also compare that the radial parts are the same
        """
        if not isinstance(other, Orbital):
            return False
        same = abs(self.R - other.R) <= 1e-4 and abs(self.q0 - other.q0) < 1e-4
        if not same:
            # Quick return
            return False
        if same and radial:
            # Ensure they also have the same fill-values
            r = np.linspace(0, self.R * 2, 500)
            same &= np.allclose(self.radial(r), other.radial(r))
        if same and psi:
            xyz = np.linspace(0, self.R * 2, 999).reshape(-1, 3)
            same &= np.allclose(self.psi(xyz), other.psi(xyz))
        return same and self.tag == other.tag

    def copy(self):
        """ Create an exact copy of this object """
        return self.__class__(self.R, self.q0, self.tag)

    def scale(self, scale):
        """ Scale the orbital by extending R by `scale` """
        R = self.R * scale
        if R < 0:
            R = -1.
        return self.__class__(R, self.q0, self.tag)

    def __eq__(self, other):
        return self.equal(other)

    def radial(self, r, *args, **kwargs):
        r""" Calculate the radial part of the wavefunction :math:`f(\mathbf R)` """
        raise NotImplementedError

    def spher(self, theta, phi, *args, **kwargs):
        r""" Calculate the spherical harmonics of this orbital at a given point (in spherical coordinates) """
        raise NotImplementedError

    def psi(self, r, *args, **kwargs):
        r""" Calculate :math:`\phi(\mathbf R)` for Cartesian coordinates """
        raise NotImplementedError

    def psi_spher(self, r, theta, phi, *args, **kwargs):
        r""" Calculate :math:`\phi(|\mathbf R|, \theta, \phi)` for spherical coordinates """
        raise NotImplementedError

    def __plot__(self, harmonics=False, axes=False, *args, **kwargs):
        """ Plot the orbital radial/spherical harmonics

        Parameters
        ----------
        harmonics : bool, optional
           if `True` the spherical harmonics will be plotted in a 3D only plot a subset of the axis, defaults to all axis
        axes : bool or matplotlib.Axes, optional
           the figure axes to plot in (if ``matplotlib.Axes`` object).
           If ``True`` it will create a new figure to plot in.
           If ``False`` it will try and grap the current figure and the current axes.
        """
        d = dict()

        if harmonics:
            # We are plotting the harmonic part
            d['projection'] = 'polar'

        axes = plt.get_axes(axes, **d)

        # Add plots
        if harmonics:

            # Calculate the spherical harmonics
            theta, phi = np.meshgrid(np.arange(360), np.arange(180) - 90)
            s = self.spher(np.radians(theta), np.radians(phi))

            # Plot data
            cax = axes.contourf(theta, phi, s, *args, **kwargs)
            cax.set_clim(s.min(), s.max())
            cab = axes.get_figure().colorbar(cax)
            axes.set_title(r'${}$'.format(self.name(True)))
            # I don't know how exactly to handle this...
            #axes.set_xlabel(r'Azimuthal angle $\theta$')
            #axes.set_ylabel(r'Polar angle $\phi$')

        else:
            # Plot the radial function and 5% above 0 value
            r = np.linspace(0, self.R * 1.05, 1000)
            f = self.radial(r)
            axes.plot(r, f, *args, **kwargs)
            axes.set_xlim(left=0)
            axes.set_xlabel('Radius [Ang]')
            axes.set_ylabel(r'$f(r)$ [1/Ang$^{3/2}$]')

        return axes

    def toGrid(self, precision=0.05, c=1., R=None, dtype=np.float64, atom=1):
        """ Create a Grid with *only* this orbital wavefunction on it

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
            raise ValueError(f"{self.__class__.__name__}.toGrid was unable to create "
                             "the orbital grid for plotting, the box size is negative.")

        # Since all these things depend on other elements
        # we will simply import them here.
        from .supercell import SuperCell
        from .geometry import Geometry
        from .grid import Grid
        from .atom import Atom
        from .physics.electron import wavefunction
        sc = SuperCell(R*2, origo=[-R] * 3)
        if isinstance(atom, Atom):
            atom = atom.copy(orbitals=self)
        else:
            atom = Atom(atom, self)
        g = Geometry([0] * 3, atom, sc=sc)
        G = Grid(precision, dtype=dtype, geometry=g)
        wavefunction(np.full(1, c), G, geometry=g)
        return G

    def __getstate__(self):
        """ Return the state of this object """
        return {'R': self.R, 'q0': self.q0, 'tag': self.tag}

    def __setstate__(self, d):
        """ Re-create the state of this object """
        self.__init__(d['R'], q0=d['q0'], tag=d['tag'])


@set_module("sisl")
class SphericalOrbital(Orbital):
    r""" An *arbitrary* orbital class where :math:`\phi(\mathbf r)=f(|\mathbf r|)Y_l^m(\theta,\varphi)`

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
    l : int
       azimuthal quantum number
    rf_or_func : tuple of (r, f) or func
       radial components as a tuple/list, or the function which can interpolate to any R
       See `set_radial` for details.
    q0 : float, optional
       initial charge
    tag : str, optional
       user defined tag

    Attributes
    ----------
    f : func
        interpolation function that returns `f(r)` for a given radius

    Examples
    --------
    >>> from scipy.interpolate import interp1d
    >>> orb = SphericalOrbital(1, (np.arange(10.), np.arange(10.)))
    >>> orb.equal(SphericalOrbital(1, interp1d(np.arange(10.), np.arange(10.),
    ...       fill_value=(0., 0.), kind='cubic', bounds_error=False)))
    True
    """
    # Additional slots (inherited classes retain the same slots)
    __slots__ = ['_l', 'f']

    def __init__(self, l, rf_or_func, q0=0., tag='', **kwargs):
        """ Initialize spherical orbital object """
        self._l = l

        # Set the internal function
        if callable(rf_or_func):
            self.set_radial(rf_or_func, **kwargs)
        elif rf_or_func is None:
            # We don't do anything
            self.f = NotImplemented
            self._R = -1.
        else:
            # it must be two arguments
            self.set_radial(rf_or_func[0], rf_or_func[1], **kwargs)

        # Initialize R and tag through the parent
        # Note that the maximum range of the orbital will be the
        # maximum value in r.
        super().__init__(self.R, q0, tag)

    @property
    def l(self):
        r""" :math:`l` quantum number """
        return self._l

    def copy(self):
        """ Create an exact copy of this object """
        return self.__class__(self.l, self.f, self.q0, self.tag)

    def equal(self, other, psi=False, radial=False):
        """ Compare two orbitals by comparing their radius, and possibly the radial and psi functions

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

    def set_radial(self, *args, **kwargs):
        r""" Update the internal radial function used as a :math:`f(|\mathbf r|)`

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
        ...    return interp.interp1d(r, f, kind='cubic', fill_value=(f[0], 0.), bounds_error=False)
        >>> def i_spline(r, f):
        ...    from functools import partial
        ...    tck = interp.splrep(r, f, k=3, s=0)
        ...    return partial(interp.splev, tck=tck, der=0, ext=1)
        >>> R = np.linspace(0, 4, 400)
        >>> o.set_radial(r, f, interp=i_univariate)
        >>> f_univariate = o.f(R)
        >>> o.set_radial(r, f, interp=i_interp1d)
        >>> f_interp1d = o.f(R)
        >>> o.set_radial(r, f, interp=i_spline)
        >>> f_spline = o.f(R)
        >>> np.allclose(f_univariate, f_interp1d)
        True
        >>> np.allclose(f_univariate, f_spline)
        True
        """
        if len(args) == 0:
            # Return immediately
            def f0(R):
                return R * 0.
            self.set_radial(f0)
            if 'R' in kwargs:
                self._R = kwargs['R']
        elif len(args) == 1 and callable(args[0]):
            self.f = args[0]
            # Determine the maximum R
            # We should never expect a radial components above
            # 50 Ang (is this not fine? ;))
            # Precision of 0.05 A
            r = np.linspace(0.05, 50, 1000)
            f = square(self.f(r))
            # Find maximum R and focus around this point
            idx = (f > 0).nonzero()[0]
            if len(idx) > 0:
                idx = idx.max()
                # Assert that we actually hit where there are zeros
                if idx < len(r) - 1:
                    idx += 1
                # Preset R
                self._R = r[idx]
                # This should give us a precision of 0.0001 A
                r = np.linspace(r[idx]-0.055+0.0001, r[idx]+0.055, 1100)
                f = square(self.f(r))
                # Find minimum R and focus around this point
                idx = (f > 0).nonzero()[0]
                if len(idx) > 0:
                    idx = idx.max()
                    if idx < len(r) - 1:
                        idx += 1
                    self._R = r[idx]

            else:
                # The orbital radius
                # Is undefined, no values are above 0 in a range
                # of 50 A
                self._R = -1

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
            interp = kwargs.get('interp', interp)

            self.set_radial(interp(r, f))
        elif 'R' in kwargs:
            self._R = kwargs.get('R')
        else:
            raise ValueError('Arguments for set_radial are in-correct, please see the documentation of SphericalOrbital.set_radial')

    def __str__(self):
        """ A string representation of the object """
        if len(self.tag) > 0:
            return f'{self.__class__.__name__}{{l: {self.l}, R: {self.R}, q0: {self.q0}, tag: {self.tag}}}'
        return f'{self.__class__.__name__}{{l: {self.l}, R: {self.R}, q0: {self.q0}}}'

    def __repr__(self):
        if self.tag:
            return f"<{self.__module__}.{self.__class__.__name__} l={self.l}, R={self.R:.3f}, q0={self.q0}, tag={self.tag}>"
        return f"<{self.__module__}.{self.__class__.__name__} l={self.l}, R={self.R:.3f}, q0={self.q0}>"

    def radial(self, r, is_radius=True):
        r""" Calculate the radial part of the wavefunction :math:`f(\mathbf R)`

        The position `r` is a vector from the origin of this orbital.

        Parameters
        -----------
        r : array_like
           radius from the orbital origin, for ``is_radius=False`` `r` must be vectors
        is_radius : bool, optional
           whether `r` is a vector or the radius

        Returns
        -------
        numpy.ndarray
            radial orbital value at point `r`
        """
        r = _a.asarrayd(r).ravel()
        if is_radius:
            s = r.shape
        else:
            r = sqrt(square(r.reshape(-1, 3)).sum(-1))
            s = r.shape
        r.shape = (-1,)
        n = len(r)
        # Only calculate where it makes sense, all other points are removed and set to zero
        idx = (r <= self.R).nonzero()[0]
        # Reduce memory immediately
        r = take(r, idx)
        p = _a.zerosd(n)
        if len(idx) > 0:
            p[idx] = self.f(r)
        p.shape = s
        return p

    def psi(self, r, m=0):
        r""" Calculate :math:`\phi(\mathbf R)` at a given point (or more points)

        The position `r` is a vector from the origin of this orbital.

        Parameters
        -----------
        r : array_like of (:, 3)
           the vector from the orbital origin
        m : int, optional
           magnetic quantum number, must be in range ``-self.l <= m <= self.l``

        Returns
        -------
        numpy.ndarray
             basis function value at point `r`
        """
        r = _a.asarrayd(r)
        s = r.shape[:-1]
        # Convert to spherical coordinates
        n, idx, r, theta, phi = cart2spher(r, theta=m != 0, cos_phi=True, maxR=self.R)
        p = _a.zerosd(n)
        if len(idx) > 0:
            p[idx] = self.psi_spher(r, theta, phi, m, cos_phi=True)
            # Reduce memory immediately
            del idx, r, theta, phi
        p.shape = s
        return p

    def spher(self, theta, phi, m=0, cos_phi=False):
        r""" Calculate the spherical harmonics of this orbital at a given point (in spherical coordinates)

        Parameters
        -----------
        theta : array_like
           azimuthal angle in the :math:`x-y` plane (from :math:`x`)
        phi : array_like
           polar angle from :math:`z` axis
        m : int, optional
           magnetic quantum number, must be in range ``-self.l <= m <= self.l``
        cos_phi : bool, optional
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

    def psi_spher(self, r, theta, phi, m=0, cos_phi=False):
        r""" Calculate :math:`\phi(|\mathbf R|, \theta, \phi)` at a given point (in spherical coordinates)

        This is equivalent to `psi` however, the input is given in spherical coordinates.

        Parameters
        -----------
        r : array_like
           the radius from the orbital origin
        theta : array_like
           azimuthal angle in the :math:`x-y` plane (from :math:`x`)
        phi : array_like
           polar angle from :math:`z` axis
        m : int, optional
           magnetic quantum number, must be in range ``-self.l <= m <= self.l``
        cos_phi : bool, optional
           whether `phi` is actually :math:`cos(\phi)` which will be faster because
           `cos` is not necessary to call.

        Returns
        -------
        numpy.ndarray
             basis function value at point `r`
        """
        return self.f(r) * self.spher(theta, phi, m, cos_phi)

    def toAtomicOrbital(self, m=None, n=None, zeta=1, P=False, q0=None):
        r""" Create a list of `AtomicOrbital` objects

        This defaults to create a list of `AtomicOrbital` objects for every `m` (for m in -l:l).
        One may optionally specify the sub-set of `m` to retrieve.

        Parameters
        ----------
        m : int or list or None
           if ``None`` it defaults to ``-l:l``, else only for the requested `m`
        zeta : int, optional
           the specified zeta-shell
        P : bool, optional
           whether the orbitals are polarized.
        q0 : float, optional
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
            return AtomicOrbital(n=n, l=self.l, m=m, zeta=zeta, P=P, spherical=self, q0=q0)
        return [AtomicOrbital(n=n, l=self.l, m=mm, zeta=zeta, P=P, spherical=self, q0=q0) for mm in m]

    def __getstate__(self):
        """ Return the state of this object """
        # A function is not necessarily pickable, so we store interpolated
        # data which *should* ensure the correct pickable state (to close agreement)
        r = np.linspace(0, self.R, 1000)
        f = self.f(r)
        return {'l': self.l, 'r': r, 'f': f, 'q0': self.q0, 'tag': self.tag}

    def __setstate__(self, d):
        """ Re-create the state of this object """
        self.__init__(d['l'], (d['r'], d['f']), q0=d['q0'], tag=d['tag'])


@set_module("sisl")
class AtomicOrbital(Orbital):
    r""" A projected atomic orbital consisting of real harmonics

    The `AtomicOrbital` is a specification of the `SphericalOrbital` by
    assigning the magnetic quantum number :math:`m` to the object.

    `AtomicOrbital` should always be preferred over the
    `SphericalOrbital` because it explicitly contains *all* quantum numbers.

    Parameters
    ----------
    *args : list of arguments
        list of arguments can be in different input options
    q0 : float, optional
        initial charge
    tag : str, optional
        user defined tag

    Examples
    --------
    >>> r = np.linspace(0, 5, 50)
    >>> f = np.exp(-r)
    >>> #                    n, l, m, [zeta, [P]]
    >>> orb1 = AtomicOrbital(2, 1, 0, 1, (r, f))
    >>> orb2 = AtomicOrbital(n=2, l=1, m=0, zeta=1, (r, f))
    >>> orb3 = AtomicOrbital('2pzZ', (r, f))
    >>> orb4 = AtomicOrbital('2pzZ1', (r, f))
    >>> orb5 = AtomicOrbital('pz', (r, f))
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
    __slots__ = ['_n', '_l', '_m', '_zeta', '_P', '_orb']

    def __init__(self, *args, **kwargs):
        """ Initialize atomic orbital object """
        super().__init__(kwargs.get('R', 0.), q0=kwargs.get('q0', 0.), tag=kwargs.get('tag', ''))

        # Ensure args is a list (to be able to pop)
        args = list(args)
        self._orb = None

        # Extract shell information
        n = kwargs.get('n', None)
        l = kwargs.get('l', None)
        m = kwargs.get('m', None)
        if 'Z' in kwargs:
            deprecate(f"{self.__class__.__name__}(Z=) is deprecated, please use (zeta=) instead")
        zeta = kwargs.get('zeta', kwargs.get('Z', 1))
        P = kwargs.get('P', False)

        if len(args) > 0:
            if isinstance(args[0], str):
                # String specification of the atomic orbital
                s = args.pop(0)

                _n = {'s': 1, 'p': 2, 'd': 3, 'f': 4, 'g': 5}
                _l = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4}
                _m = {'s': 0,
                      'pz': 0, 'px': 1, 'py': -1,
                      'dxy': -2, 'dyz': -1, 'dz2': 0, 'dxz': 1, 'dx2-y2': 2,
                      'fy(3x2-y2)': -3, 'fxyz': -2, 'fz2y': -1, 'fz3': 0,
                      'fz2x': 1, 'fz(x2-y2)': 2, 'fx(x2-3y2)': 3,
                      'gxy(x2-y2)': -4, 'gzy(3x2-y2)': -3, 'gz2xy': -2, 'gz3y': -1, 'gz4': 0,
                      'gz3x': 1, 'gz2(x2-y2)': 2, 'gzx(x2-3y2)': 3, 'gx4+y4': 4,
                }

                # First remove a P for polarization
                P = 'P' in s
                s = s.replace('P', '')

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
                except:
                    n = _n.get(s[0])

                # Get l
                l = _l.get(s[0])

                # Get number of zeta shell
                iZ = s.find('Z')
                if iZ >= 0:
                    # Currently we know that we are limited to 9 zeta shells.
                    # However, for now we assume this is enough (could easily
                    # be extended by a reg-exp)
                    try:
                        zeta = int(s[iZ+1])
                        # Remove Z + int
                        s = s[:iZ] + s[iZ+2:]
                    except:
                        zeta = 1
                        s = s[:iZ] + s[iZ+1:]

                # We should be left with m specification
                m = _m.get(s)

                # Now we should figure out how the spherical orbital
                # has been passed.
                # There are two options:
                #  1. The radial function is passed as two arrays: r, f
                #  2. The SphericalOrbital-class is passed which already contains
                #     the relevant information.
                # Figure out if it is a sphericalorbital
                if len(args) > 0:
                    if isinstance(args[0], SphericalOrbital):
                        self._orb = args.pop(0)
                    else:
                        self._orb = SphericalOrbital(l, args.pop(0))
            else:

                # Arguments *have* to be
                # n, l, [m (only for l>0)] [, zeta [, P]]
                if n is None and len(args) > 0:
                    n = args.pop(0)
                if l is None and len(args) > 0:
                    l = args.pop(0)
                if l > 0:
                    if m is None and len(args) > 0:
                        m = args.pop(0)
                else:
                    m = 0

                # Now we need to figure out if they are shell
                # information or radial functions
                if len(args) > 0:
                    if isinstance(args[0], Integral):
                        zeta = args.pop(0)
                if len(args) > 0:
                    if isinstance(args[0], bool):
                        P = args.pop(0)

                # Figure out if it is a sphericalorbital
                if len(args) > 0:
                    if isinstance(args[0], SphericalOrbital):
                        self._orb = args.pop(0)
                    else:
                        self._orb = SphericalOrbital(l, args.pop(0))

        # Still if n is None, we assign the default (lowest) quantum number
        if n is None:
            n = l + 1

        # Copy over information
        self._n = n
        self._l = l
        self._m = m
        self._zeta = zeta
        self._P = P

        if self.l > 4:
            raise ValueError(f'{self.__class__.__name__} does not implement shell h and above!')
        if abs(self.m) > self.l:
            raise ValueError(f'{self.__class__.__name__} requires |m| <= l.')

        # Retrieve user-passed spherical orbital
        s = kwargs.get('spherical', None)

        if s is None:
            # Expect the orbital to already be set
            pass
        elif isinstance(s, Orbital):
            self._orb = s
        else:
            self._orb = SphericalOrbital(l, s)

        if self._orb is None:
            # Default orbital to none, this will not create any radial functions
            # But any use of the orbital will still work
            self._orb = Orbital(self.R)

        self._R = self._orb.R

    @property
    def n(self):
        r""" :math:`n` shell """
        return self._n

    @property
    def l(self):
        r""" :math:`l` quantum number """
        return self._l

    @property
    def m(self):
        r""" :math:`m` quantum number """
        return self._m

    @property
    def zeta(self):
        r""" :math:`\zeta` shell """
        return self._zeta

    @property
    @deprecate_method("AtomicOrbital.Z is deprecated, please use .zeta instead")
    def Z(self):
        r""" :math:`\zeta` shell """
        return self._zeta

    @property
    def P(self):
        r""" Whether this is polarized shell or not """
        return self._P

    @property
    def orb(self):
        r""" Orbital with radial part """
        return self._orb

    @property
    def f(self):
        r""" Radial function """
        return self.orb.f

    def copy(self):
        """ Create an exact copy of this object """
        return self.__class__(n=self.n, l=self.l, m=self.m, zeta=self.zeta, P=self.P, spherical=self.orb.copy(), q0=self.q0, tag=self.tag)

    def equal(self, other, psi=False, radial=False):
        """ Compare two orbitals by comparing their radius, and possibly the radial and psi functions

        Parameters
        ----------
        other : Orbital
           comparison orbital
        psi : bool, optional
           also compare that the full psi are the same
        radial : bool, optional
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
        """ Return named specification of the atomic orbital """
        if tex:
            name = '{}{}'.format(self.n, 'spdfg'[self.l])
            if self.l == 1:
                name += ['_y', '_z', '_x'][self.m+1]
            elif self.l == 2:
                name += ['_{xy}', '_{yz}', '_{z^2}', '_{xz}', '_{x^2-y^2}'][self.m+2]
            elif self.l == 3:
                name += ['_{y(3x^2-y^2)}', '_{xyz}', '_{z^2y}', '_{z^3}', '_{z^2x}', '_{z(x^2-y^2)}', '_{x(x^2-3y^2)}'][self.m+3]
            elif self.l == 4:
                name += ['_{_{xy(x^2-y^2)}}', '_{zy(3x^2-y^2)}', '_{z^2xy}', '_{z^3y}', '_{z^4}',
                         '_{z^3x}', '_{z^2(x^2-y^2)}', '_{zx(x^2-3y^2)}', '_{x^4+y^4}'][self.m+4]
            if self.P:
                return name + fr'\zeta^{self.zeta}\mathrm{{P}}'
            return name + fr'\zeta^{self.zeta}'
        name = '{}{}'.format(self.n, 'spdfg'[self.l])
        if self.l == 1:
            name += ['y', 'z', 'x'][self.m+1]
        elif self.l == 2:
            name += ['xy', 'yz', 'z2', 'xz', 'x2-y2'][self.m+2]
        elif self.l == 3:
            name += ['y(3x2-y2)', 'xyz', 'z2y', 'z3', 'z2x', 'z(x2-y2)', 'x(x2-3y2)'][self.m+3]
        elif self.l == 4:
            name += ['xy(x2-y2)', 'zy(3x2-y2)', 'z2xy', 'z3y', 'z4',
                     'z3x', 'z2(x2-y2)', 'zx(x2-3y2)', 'x4+y4'][self.m+4]
        if self.P:
            return name + f'Z{self.zeta}P'
        return name + f'Z{self.zeta}'

    def __str__(self):
        """ A string representation of the object """
        if len(self.tag) > 0:
            return f'{self.__class__.__name__}{{{self.name()}, q0: {self.q0}, tag: {self.tag}, {str(self.orb)}}}'
        return f'{self.__class__.__name__}{{{self.name()}, q0: {self.q0}, {str(self.orb)}}}'

    def __repr__(self):
        if self.tag:
            return f"<{self.__module__}.{self.__class__.__name__} {self.name()} q0={self.q0}, tag={self.tag}>"
        return f"<{self.__module__}.{self.__class__.__name__} {self.name()} q0={self.q0}>"

    def set_radial(self, *args):
        r""" Update the internal radial function used as a :math:`f(|\mathbf r|)`

        See `SphericalOrbital.set_radial` where these arguments are passed to.
        """
        self.orb.set_radial(*args)
        self._R = self.orb.R

    def radial(self, r, is_radius=True):
        r""" Calculate the radial part of the wavefunction :math:`f(\mathbf R)`

        The position `r` is a vector from the origin of this orbital.

        Parameters
        -----------
        r : array_like
           radius from the orbital origin, for ``is_radius=False`` `r` must be vectors
        is_radius : bool, optional
           whether `r` is a vector or the radius

        Returns
        -------
        numpy.ndarray
            radial orbital value at point `r`
        """
        return self.orb.radial(r, is_radius=is_radius)

    def psi(self, r):
        r""" Calculate :math:`\phi(\mathbf r)` at a given point (or more points)

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

    def spher(self, theta, phi, cos_phi=False):
        r""" Calculate the spherical harmonics of this orbital at a given point (in spherical coordinates)

        Parameters
        -----------
        theta : array_like
           azimuthal angle in the :math:`x-y` plane (from :math:`x`)
        phi : array_like
           polar angle from :math:`z` axis
        cos_phi : bool, optional
           whether `phi` is actually :math:`cos(\phi)` which will be faster because
           `cos` is not necessary to call.

        Returns
        -------
        numpy.ndarray
            spherical harmonics at angles :math:`\theta` and :math:`\phi`
        """
        return self.orb.spher(theta, phi, self.m, cos_phi)

    def psi_spher(self, r, theta, phi, cos_phi=False):
        r""" Calculate :math:`\phi(|\mathbf R|, \theta, \phi)` at a given point (in spherical coordinates)

        This is equivalent to `psi` however, the input is given in spherical coordinates.

        Parameters
        -----------
        r : array_like
           the radius from the orbital origin
        theta : array_like
           azimuthal angle in the :math:`x-y` plane (from :math:`x`)
        phi : array_like
           polar angle from :math:`z` axis
        cos_phi : bool, optional
           whether `phi` is actually :math:`cos(\phi)` which will be faster because
           `cos` is not necessary to call.

        Returns
        -------
        numpy.ndarray
             basis function value at point `r`
        """
        return self.orb.psi_spher(r, theta, phi, self.m, cos_phi)

    def __getstate__(self):
        """ Return the state of this object """
        # A function is not necessarily pickable, so we store interpolated
        # data which *should* ensure the correct pickable state (to close agreement)
        try:
            # this will tricker the AttributeError
            # before we create the data-array
            f = self.orb.f
            r = np.linspace(0, self.R, 1000)
            f = f(r)
        except AttributeError:
            r, f = None, None
        return {'name': self.name(), 'r': r, 'f': f, 'q0': self.q0, 'tag': self.tag}

    def __setstate__(self, d):
        """ Re-create the state of this object """
        if d["r"] is None:
            self.__init__(d['name'], q0=d['q0'], tag=d['tag'])
        else:
            self.__init__(d['name'], (d['r'], d['f']), q0=d['q0'], tag=d['tag'])
