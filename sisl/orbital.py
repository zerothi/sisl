from __future__ import print_function, division

# To check for integers
from numbers import Integral

import numpy as np
from numpy import pi
from scipy.misc import factorial
from scipy.special import lpmv, sph_harm
from scipy.interpolate import interp1d

from sisl._help import ensure_array, _str
import sisl._array as _a


__all__ = ['Orbital', 'SphericalOrbital', 'AtomicOrbital']


def xyz2spher(r):
    r""" Transfer a vector to the spherical coordinates

    Parameters
    ----------
    r : array_like
       the cartesian vectors

    Returns
    -------
    r : numpy.ndarray
       the radius in spherical coordinates
    theta : numpy.ndarray
       the angle in the :math:`x-y` plane from :math:`x`
    phi : numpy.ndarray
       angle from :math:`z` and down towards the :math:`x-y` plane
    """
    r = ensure_array(r, np.float64)
    r.shape = (-1, 3)
    rr = np.sqrt((r ** 2).sum(1))
    phi = np.arctan2(r[:, 1], r[:, 0])
    theta = np.where(rr != 0, np.arccos(r[:, 2] / rr), 0)
    return rr, theta, phi


def spherical_harm(m, l, theta, phi):
    r""" Calculate the spherical harmonics using :math:`Y_l^m(\theta, \varphi)` with :math:`\mathbf R\to \{r, \theta, \varphi\}`.

    .. math::
        Y^m_l(\theta,\varphi) = (-1)^m\sqrt{\frac{2l+1}{4\pi} \frac{(l-m)!}{(l+m)!}}
             e^{i m \theta} P^m_l(\cos(\varphi))

    which is the spherical harmonics with the Condon-Shortley phase.

    Parameters
    ----------
    m : int
       order of the spherical harmonics
    l : int
       degree of the spherical harmonics
    theta : array_like
       angle in :math:`x-y` plane (azimuthal angle)
    phi : array_like
       angle from :math:`z` axis and down (polar angle)
    """
    # Probably same as:
    #return (-1) ** m * ( (2*l+1)/(4*pi) * factorial(l-m) / factorial(l+m) ) ** 0.5 \
    #    * lpmv(m, l, np.cos(theta)) * np.exp(1j * m * phi)
    return sph_harm(m, l, theta, phi) * (-1) ** m


class Orbital(object):
    """ Base class for orbital information. This base class holds nothing but the *tag* and the orbital radius

    The orbital class is still in an experimental stage and will probably evolve over some time.

    Attributes
    ----------
    R : float
        the maximum orbital range, any query on values beyond this range should return 0.
    tag :
        the assigned tag for this orbital
    """
    __slots__ = ['R', 'tag']

    def __init__(self, R, tag=None):
        """ Initialize the orbital class with a radius (`R`) and a tag (`tag`)

        Parameters
        ----------
        R : float
           the orbital radius
        tag : str, optional
           the provided tag of the orbital
        """
        self.R = R
        self.tag = tag

    def __repr__(self):
        """ A string representation of the object """
        tag = self.tag
        if tag is None:
            tag = ''
        if len(tag) > 0:
            return self.__class__.__name__ + '{{{0}, tag: {1}}}'.format(self.name(), tag)
        return self.__class__.__name__ + '{{R: {0}}}'.format(self.R)

    def equal(self, other, radial=False, phi=False):
        """ Compare two orbitals by comparing their radius, and possibly the radial and phi functions

        Parameters
        ----------
        other : Orbital
           comparison orbital
        radial : bool, optional
           also compare that the radial parts are the same
        phi : bool, optional
           also compare that the full phi are the same
        """
        if not isinstance(other, Orbital):
            return False
        same = np.isclose(self.R, other.R)
        if radial:
            r = np.linspace(0, self.R, 500)
            same &= np.allclose(self.radial(r), other.radial(r))
        return same

    def radial(self, r, *args, **kwargs):
        raise NotImplementedError

    def phi(self, r, *args, **kwargs):
        raise NotImplementedError


class SphericalOrbital(Orbital):
    r""" An arbitrary orbital class where :math:`\phi(\mathbf r)=f(|\mathbf r|)Y_l^m(\theta,\varphi)`

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

    Attributes
    ----------
    R : float
        the maximum orbital range
    l : int
        azimuthal quantum number
    f : func
        the interpolation function that returns `f(r)` for the provided data
    """
    # Additional slots (inherited classes retain the same slots)
    __slots__ = ['l', 'f']

    def __init__(self, l, r, f, kind='cubic', tag=None):
        """ Initialize a spherical orbital via a radial grid

        Parameters
        ----------
        l : int
           azimuthal quantum number
        r : array_like
           radii for the evaluated function `f`
        f : array_like
           the function evaluated at `r`
        kind : str, func, optional
           the kind of interpolation used, if this is a string it is the `kind` parameter
           in `scipy.interpolate.interp1d`, otherwise it may be a function which should return
           an interpolation function which only accepts two arguments: ``func = kind(r, f)``
        tag : str, optional
           tag of the orbital
        """
        self.l = l
        if self.l > 4:
            raise ValueError(self.__class__.__name__ + ' does not implement h and shells above!')

        # Initialize R and tag through the parent
        # Note that the maximum range of the orbital will be the
        # maximum value in r.
        r = ensure_array(r, np.float64)
        super(SphericalOrbital, self).__init__(r.max(), tag)

        # Set the internal function
        self.update_f(r, f, kind)

    def update_f(self, r, f, kind='cubic'):
        """ Update the internal radial function used as a :math:`f(|\mathbf r|)`

        Parameters
        ----------
        r : array_like
           radii for the evaluated function `f`
        f : array_like
           the function evaluated at `r`
        kind : str, func, optional
           the kind of interpolation used, if this is a string it is the `kind` parameter
           in `scipy.interpolate.interp1d`, otherwise it may be a function which should return
           an interpolation function which only accepts two arguments: ``func = kind(r, f)``
        """
        r = ensure_array(r, np.float64)
        f = ensure_array(f, np.float64)
        # Sort r and f
        idx = np.argsort(r)
        r = r[idx]
        f = f[idx]
        # Also update R
        self.R = r[-1]

        if isinstance(kind, _str):
            # Now make interpolation extrapolation values
            # fill_value *has* to be a tuple
            self.f = interp1d(r, f, kind=kind, fill_value=(f[0], f[-1]), assume_sorted=True)
        else:
            self.f = kind(r, f)
            # Just to be sure we actually have a working function
            self.f(r[0])

    def __repr__(self):
        """ A string representation of the object """
        tag = self.tag
        if tag is None:
            tag = ''
        if len(tag) > 0:
            return self.__class__.__name__ + '{{l: {0}, R: {1}, tag: {2}}}'.format(self.l, self.R, tag)
        return self.__class__.__name__ + '{{l: {0}, R: {1}}}'.format(self.l, self.R)

    def radial(self, r, is_radius=True):
        r""" Calculate the radial part of the wavefunction :math:`f(\mathbf R)`

        The position `r` is a vector from the origin of this orbital.

        Parameters
        -----------
        r : array_like
           radius from the orbital origin, for `is_radius=False` `r` must be vectors
        is_radius : bool, optional
           whether `r` is a vector or the radius

        Returns
        -------
        f : the orbital value at point `r`
        """
        r = ensure_array(r, np.float64)
        if not is_radius:
            r.shape = (-1, 3)
            r = np.sqrt((r ** 2).sum(1))
        p = _a.zerosd(len(r))
        # Only calculate where it makes sense, all other points are removed and set to zero
        idx = (r <= self.R).nonzero()[0]
        if len(idx) > 0:
            p[idx] = self.f(r[idx])
        return p

    def phi(self, r, m=0):
        r""" Calculate :math:`\phi(\mathbf R)` at a given point (or more points)

        The position `r` is a vector from the origin of this orbital.

        Parameters
        -----------
        r : array_like
           the vector from the orbital origin
        m : int, optional
           magnetic quantum number, must be in range ``-self.l <= m <= self.l``

        Returns
        -------
        phi : the orbital value at point `r`
        """
        # Convert to spherical coordinates
        r, theta, phi = xyz2spher(r)
        # Only calculate where it makes sense, all other points are removed and set to zero
        idx = (r <= self.R).nonzero()[0]
        p = _a.zerosd(len(r))
        if len(idx) > 0:
            p[idx] = self.f(r[idx]) * spherical_harm(m, self.l, r[idx], theta[idx], phi[idx])
        return p

    def toAtomicOrbital(self, m=None, n=None, Z=1, P=False):
        """ Create a list of `AtomicOrbital` objects 

        This defaults to create a list of `AtomicOrbital` objects for every `m` (for m in -l:l).
        One may optionally specify the sub-set of `m` to retrieve.

        Parameters
        ----------
        m : int or list or None
           if ``None`` it defaults to -l:l, else only for the requested `m`
        Z : int, optional
           the specified zeta-shell
        P : bool, optional
           whether the orbitals are polarized.
        """
        if m is None:
            m = range(-self.l, self.l + 1)
        elif isinstance(m, Integral):
            return AtomicOrbital(n=n, l=self.l, m=m, Z=Z, P=P, spherical=self)
        return [AtomicOrbital(n=n, l=self.l, m=mm, Z=Z, P=P, spherical=self) for mm in m]


class AtomicOrbital(Orbital):
    r""" A projected atomic orbital made of real  """

    # All of these follow standard notation:
    #   n = principal quantum number
    #   l = azimuthal quantum number
    #   m = magnetic quantum number
    #   Z = zeta shell
    #   P = polarization shell or not
    # orb is the SphericalOrbital class that retains the radial
    # grid and enables to calculate phi(r)
    __slots__ = ['n', 'l', 'm', 'Z', 'P', 'orb']

    def __init__(self, *args, **kwargs):
        """ Initialize the orbital class with a radius (`R`) and a tag (`tag`)

        Parameters
        ----------
        *args : list of arguments
           the list of arguments can be in different input options
        tag : str, optional
           the provided tag of the orbital
        """
        # Immediately setup R and tag
        super(AtomicOrbital, self).__init__(0., kwargs.get('tag', None))
        # Ensure args is a list (to be able to pop)
        args = list(args)

        # Extract shell information
        n = kwargs.get('n', None)
        l = kwargs.get('l', None)
        m = kwargs.get('m', None)
        Z = kwargs.get('Z', 1)
        P = kwargs.get('P', False)

        if len(args) > 0:
            if isinstance(args[0], _str):
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
                iZ = s.index('Z')
                if iZ >= 0:
                    # Currently we know that we are limited to 9 zeta shells.
                    # However, for now we assume this is enough (could easily
                    # be extended by a reg-exp)
                    Z = int(s[iZ+1])
                    # Remove Z + int
                    s = s[:iZ] + s[iZ+2:]

                # We should be left with m specification
                m = _m.get(s)

                # Now we should figure out how the spherical orbital
                # has been passed.
                # There are two options:
                #  1. The radial function is passed as two arrays: r, f
                #  2. The SphericalOrbital-class is passed which already contains
                #     the relevant information.
                if len(args) > 0:
                    if isinstance(args[0], SphericalOrbital):
                        self.orb = args.pop(0)
                if len(args) > 1:
                    # It could be r, f
                    if isinstance(args[0], (list, tuple, np.ndarray)) and \
                       isinstance(args[1], (list, tuple, np.ndarray)):
                        self.orb = SphericalOrbital(l, args.pop(0), args.pop(0))
            else:

                # Arguments *have* to be
                # n, l, [m (only for l>0)] [, Z [, P]]
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
                        Z = args.pop(0)
                if len(args) > 0:
                    if isinstance(args[0], bool):
                        P = args.pop(0)

                # Figure out if it is a sphericalorbital
                if len(args) > 0:
                    if isinstance(args[0], SphericalOrbital):
                        self.orb = args.pop(0)
                # It could be r, f
                if len(args) > 1:
                    if isinstance(args[0], (list, tuple, np.ndarray)) and \
                       isinstance(args[1], (list, tuple, np.ndarray)):
                        self.orb = SphericalOrbital(l, args.pop(0), args.pop(0))

        # Still if n is None, we assign the default (lowest) quantum number
        if n is None:
            n = l + 1

        # Copy over information
        self.n = n
        self.l = l
        self.m = m
        self.Z = Z
        self.P = P

        if self.l > 4:
            raise ValueError(self.__class__.__name__ + ' does not implement h and shells above!')

        if 'spherical' in kwargs:
            self.orb = kwargs.get('spherical')
        self.R = self.orb.R

    def name(self, tex=False):
        """ Return named specification of the atomic orbital """
        if tex:
            name = '{0}{1}'.format(self.n, {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g'}.get(self.l))
            if self.l == 1:
                name += {0: '_z', 1: '_x', -1: '_y'}.get(self.m)
            elif self.l == 2:
                name += {-2: '_{xy}', -1: '_{yz}', 0: '_{z^2}', 1: '_{xz}', 2: '_{x^2-y^2}'}.get(self.m)
            elif self.l == 3:
                name += {-3: '_{y(3x^2-y^2)}', -2: '_{xyz}', -1: '_{z^2y}', 0: '_{z^3}',
                         1: '_{z^2x}', 2: '_{z(x^2-y^2)}', 3: '_{x(x^2-3y^2)}'}.get(self.m)
            elif self.l == 4:
                name += {-4: '_{_{xy(x^2-y^2)}}', -3: '_{zy(3x^2-y^2)}', -2: '_{z^2xy}', -1: '_{z^3y}', 0: '_{z^4}',
                         1: '_{z^3x}', 2: '_{z^2(x^2-y^2)}', 3: '_{zx(x^2-3y^2)}', 4: '_{x^4+y^4}'}.get(self.m)
            if self.P:
                return name + r'\zeta^{}\mathrm{P}'.format(self.Z)
            return name + r'\zeta^{}'.format(self.Z)
        name = '{0}{1}'.format(self.n, {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g'}.get(self.l))
        if self.l == 1:
            name += {0: 'z', 1: 'x', -1: 'y'}.get(self.m)
        elif self.l == 2:
            name += {-2: 'xy', -1: 'yz', 0: 'z2', 1: 'xz', 2: 'x2-y2'}.get(self.m)
        elif self.l == 3:
            name += {-3: 'y(3x2-y2)', -2: 'xyz', -1: 'z2y', 0: 'z3',
                     1: 'z2x', 2: 'z(x2-y2)', 3: 'x(x2-3y2)'}.get(self.m)
        elif self.l == 4:
            name += {-4: 'xy(x2-y2)', -3: 'zy(3x2-y2)', -2: 'z2xy', -1: 'z3y', 0: 'z4',
                     1: 'z3x', 2: 'z2(x2-y2)', 3: 'zx(x2-3y2)', 4: 'x4+y4'}.get(self.m)
        if self.P:
            return name + 'Z{}P'.format(self.Z)
        return name + 'Z{}'.format(self.Z)

    def __repr__(self):
        """ A string representation of the object """
        tag = self.tag
        if tag is None:
            tag = ''
        if len(tag) > 0:
            return self.__class__.__name__ + '{{{0}, tag: {1},\n  {2}\n}}'.format(self.name(), tag, repr(self.orb))
        return self.__class__.__name__ + '{{{0},\n  {1}\n}}'.format(self.name(), repr(self.orb))

    def update_f(self, r, f, kind='cubic'):
        """ Update the internal radial function used as a :math:`f(|\mathbf r|)`

        Parameters
        ----------
        r : array_like
           radii for the evaluated function `f`
        f : array_like
           the function evaluated at `r`
        kind : str, func, optional
           the kind of interpolation used, if this is a string it is the `kind` parameter
           in `scipy.interpolate.interp1d`, otherwise it may be a function which should return
           an interpolation function which only accepts two arguments: ``func = kind(r, f)``
        """
        self.orb.update_f(r, f, kind)
        self.R = self.orb.R

    def radial(self, r, is_radius=True):
        r""" Calculate the radial part of the wavefunction :math:`f(\mathbf R)`

        The position `r` is a vector from the origin of this orbital.

        Parameters
        -----------
        r : array_like
           radius from the orbital origin, for `is_radius=False` `r` must be vectors
        is_radius : bool, optional
           whether `r` is a vector or the radius

        Returns
        -------
        f : the orbital value at point `r`
        """
        return self.orb.radial(r, is_radius=is_radius)

    def phi(self, r):
        r""" Calculate :math:`\phi(\mathbf r)` at a given point (or more points)

        The position `r` is a vector from the origin of this orbital.

        Parameters
        -----------
        r : array_like
           the vector from the orbital origin

        Returns
        -------
        phi : the orbital value at point `r`
        """
        return self.orb.phi(r, self.m)
