from __future__ import print_function, division

# To check for integers
from numbers import Integral
from math import pi

import numpy as np
from scipy.misc import factorial
from scipy.special import lpmv, sph_harm
from scipy.interpolate import interp1d

from ._help import ensure_array, _str
import sisl._array as _a


__all__ = ['Orbital', 'SphericalOrbital', 'AtomicOrbital']


def xyz_spher_psi(r, maxR):
    r""" Transfer a vector to spherical coordinates

    Parameters
    ----------
    r : array_like
       the cartesian vectors
    maxR : float
       cutoff of the spherical coordinate calculations

    Returns
    -------
    n : int
       number of total points
    idx : numpy.ndarray
       indices of points with ``r <= maxR``
    r : numpy.ndarray
       radius in spherical coordinates
    theta : numpy.ndarray
       angle in the :math:`x-y` plane from :math:`x` (azimuthal)
    cos_phi : numpy.ndarray
       cosine to the angle from :math:`z` axis (polar)
    """
    r = ensure_array(r, np.float64)
    r.shape = (-1, 3)
    n = len(r)
    rr = np.sqrt((r ** 2).sum(1))
    idx = (rr <= maxR).nonzero()[0]
    # Only calculate where we are interested
    r = r[idx, :]
    rr = rr[idx]
    theta = np.arctan2(r[:, 1], r[:, 0])
    cos_phi = _a.zerosd(len(idx))
    idx2 = rr > 0
    cos_phi[idx2] = r[idx2, 2] / rr[idx2]
    return n, idx, rr, theta, cos_phi


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
       angle in :math:`x-y` plane (azimuthal)
    phi : array_like
       angle from :math:`z` axis (polar)
    """
    # Probably same as:
    #return (-1) ** m * ( (2*l+1)/(4*pi) * factorial(l-m) / factorial(l+m) ) ** 0.5 \
    #    * lpmv(m, l, np.cos(theta)) * np.exp(1j * m * phi)
    return sph_harm(m, l, theta, phi) * (-1) ** m


def rspherical_harm(m, l, theta, cos_phi):
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
    P = lpmv(m, l, cos_phi)
    if m == 0:
        return ((2*l + 1) / (4 * pi)) ** .5 * P
    elif m < 0:
        return -(2 * (2*l + 1) / (4 * pi) * factorial(l-m) / factorial(l+m)) ** .5 * P * np.sin(m*theta) * (-1) ** m
    return (2 * (2*l + 1) / (4 * pi) * factorial(l-m) / factorial(l+m)) ** .5 * P * np.cos(m*theta)


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
            return self.__class__.__name__ + '{{R: {0}, tag: {1}}}'.format(self.R, tag)
        return self.__class__.__name__ + '{{R: {0}}}'.format(self.R)

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
        if not isinstance(other, Orbital):
            return False
        same = np.isclose(self.R, other.R)
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
        return self.__class__(self.R, self.tag)

    def scale(self, scale):
        """ Scale the orbital by extending R by `scale` """
        R = self.R * scale
        if R < 0:
            R = -1.
        return self.__class__(R, self.tag)

    def __eq__(self, other):
        return self.equal(other)

    def radial(self, r, *args, **kwargs):
        raise NotImplementedError

    def psi(self, r, *args, **kwargs):
        raise NotImplementedError

    def toGrid(self, c=1., precision=0.05, R=None, dtype=np.float64):
        """ Create a Grid with *only* this orbital wavefunction on it

        Parameters
        ----------
        c : float or complex, optional
           coefficient for the orbital
        precision : float, optional
           used separation in the `Grid` between voxels (in Ang)
        R : float, optional
            box size of the grid (default to the orbital range)
        dtype : numpy.dtype, optional
            the used separation in the `Grid` between voxels
        """
        if R is None:
            R = self.R
        if R < 0:
            raise ValueError(self.__class__.__name__ + " was unable to create "
                             "the orbital grid for plotting, the orbital range is negative.")
        # Since all these things depend on other elements
        # we will simply import them here.
        from .supercell import SuperCell
        from .geometry import Geometry
        from .grid import Grid
        from .atom import Atom
        sc = SuperCell(R*2, origo=[-R] * 3)
        g = Geometry([0] * 3, Atom(1, self), sc=sc)
        n = int(np.rint(2 * R / precision))
        G = Grid([n] * 3, dtype=dtype, geom=g)
        G.psi(c)
        return G

    def __getstate__(self):
        """ Return the state of this object """
        return {'R': self.R, 'tag': self.tag}

    def __setstate__(self, d):
        """ Re-create the state of this object """
        self.__init__(d['R'], d['tag'])


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
    tag : str or None
        a tag for this orbital
    """
    # Additional slots (inherited classes retain the same slots)
    __slots__ = ['l', 'f']

    def __init__(self, l, rf_or_func, tag=None):
        """ Initialize a spherical orbital via a radial grid

        Parameters
        ----------
        l : int
           azimuthal quantum number
        rf_or_func : tuple of (r, f) or func
           the radial components as a tuple/list, or the function which can interpolate to any R
           See `set_radial` for details.
        tag : str, optional
           tag of the orbital

        Examples
        --------
        >>> from scipy.interpolate import interp1d
        >>> orb = SphericalOrbital(1, (np.arange(10), np.arange(10)))
        >>> orb.equal(SphericalOrbital(1, interp1d(np.arange(10), np.arange(10),
        ...       fill_value=(0., 0.), kind='cubic')))
        """
        self.l = l

        # Set the internal function
        if callable(rf_or_func):
            self.set_radial(rf_or_func)
        else:
            # it must be two arguments
            self.set_radial(rf_or_func[0], rf_or_func[1])

        # Initialize R and tag through the parent
        # Note that the maximum range of the orbital will be the
        # maximum value in r.
        super(SphericalOrbital, self).__init__(self.R, tag)

    def copy(self):
        """ Create an exact copy of this object """
        return self.__class__(self.l, self.f, self.tag)

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
        same = super(SphericalOrbital, self).equal(other, psi, radial)
        if not same:
            return False
        if isinstance(other, SphericalOrbital):
            same &= self.l == other.l
        return same

    def set_radial(self, *args):
        """ Update the internal radial function used as a :math:`f(|\mathbf r|)`

        This can be called in several ways:

              set_radial(r, f)
                    which uses ``scipy.interpolate.interp1d(r, f, bounds_error=False, kind='cubic')``
                    to define the interpolation function.
                    Here the maximum radius of the orbital is the maximum `r` value,
                    regardless of ``f(r)`` is zero for smaller `r`.

              set_radial(func)
                    which sets the interpolation function directly.
                    The maximum orbital range is determined automatically to a precision
                    of 0.0001 AA.
        """
        if len(args) == 0:
            # Return immediately
            return
        elif len(args) == 1 and callable(args[0]):
            self.f = args[0]
            # Determine the maximum R
            # We should never expect a radial components above
            # 50 Ang (is this not fine? ;))
            # Precision of 0.05 A
            r = np.linspace(0.05, 50, 1000)
            f = self.f(r) ** 2
            # Find maximum R and focus around this point
            idx = (f > 0).nonzero()[0]
            if len(idx) > 0:
                idx = idx.max()
                # Preset
                self.R = r[idx]
                # This should give us a precision of 0.0001 A
                r = np.linspace(r[idx]-0.025+0.0001, r[idx]+0.025, 500)
                f = self.f(r) ** 2
                # Find minimum R and focus around this point
                idx = (f > 0).nonzero()[0]
                if len(idx) > 0:
                    idx = idx.max()
                    self.R = r[idx]
            else:
                # The orbital radius
                # Is undefined, no values are above 0 in a range
                # of 50 A
                self.R = -1

        elif len(args) > 1:

            # A radial and function component has been passed
            r = ensure_array(args[0], np.float64)
            f = ensure_array(args[1], np.float64)
            # Sort r and f
            idx = np.argsort(r)
            r = r[idx]
            f = f[idx]
            # Also update R to the maximum R value
            self.R = r[-1]

            self.f = interp1d(r, f, kind='cubic', fill_value=(f[0], 0.), bounds_error=False, assume_sorted=True)
        else:
            raise ValueError('Arguments for set_radial are in-correct, please see the documentation of SphericalOrbital.set_radial')

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
           radius from the orbital origin, for ``is_radius=False`` `r` must be vectors
        is_radius : bool, optional
           whether `r` is a vector or the radius

        Returns
        -------
        f : the orbital value at point `r`
        """
        r = ensure_array(r, np.float64)
        if is_radius:
            s = r.shape
        else:
            r = np.sqrt((r ** 2).sum(-1))
            s = r.shape
        r.shape = (-1,)
        n = len(r)
        # Only calculate where it makes sense, all other points are removed and set to zero
        idx = (r <= self.R).nonzero()[0]
        # Reduce memory immediately
        r = r[idx]
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
        r : array_like
           the vector from the orbital origin
        m : int, optional
           magnetic quantum number, must be in range ``-self.l <= m <= self.l``

        Returns
        -------
        psi : the orbital value at point `r`
        """
        r = ensure_array(r, np.float64)
        s = r.shape[:-1]
        # Convert to spherical coordinates
        n, idx, r, theta, phi = xyz_spher_psi(r, self.R)
        p = _a.zerosd(n)
        if len(idx) > 0:
            p[idx] = self.f(r) * rspherical_harm(m, self.l, theta, phi)
            # Reduce memory immediately
            del idx, r, theta, phi
        p.shape = s
        return p

    def toAtomicOrbital(self, m=None, n=None, Z=1, P=False):
        """ Create a list of `AtomicOrbital` objects 

        This defaults to create a list of `AtomicOrbital` objects for every `m` (for m in -l:l).
        One may optionally specify the sub-set of `m` to retrieve.

        Parameters
        ----------
        m : int or list or None
           if ``None`` it defaults to ``-l:l``, else only for the requested `m`
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

    def __getstate__(self):
        """ Return the state of this object """
        # A function is not necessarily pickable, so we store interpolated
        # data which *should* ensure the correct pickable state (to close agreement)
        r = np.linspace(0, self.R, 1000)
        f = self.f(r)
        return {'l': self.l, 'r': r, 'f': f, 'tag': self.tag}

    def __setstate__(self, d):
        """ Re-create the state of this object """
        self.__init__(d['l'], (d['r'], d['f']), d['tag'])


class AtomicOrbital(Orbital):
    r""" A projected atomic orbital made of real  """

    # All of these follow standard notation:
    #   n = principal quantum number
    #   l = azimuthal quantum number
    #   m = magnetic quantum number
    #   Z = zeta shell
    #   P = polarization shell or not
    # orb is the SphericalOrbital class that retains the radial
    # grid and enables to calculate psi(r)
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
        self.orb = None

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
                iZ = s.find('Z')
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
                # Figure out if it is a sphericalorbital
                if len(args) > 0:
                    if isinstance(args[0], SphericalOrbital):
                        self.orb = args.pop(0)
                    else:
                        self.orb = SphericalOrbital(l, args.pop(0))
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
                    else:
                        self.orb = SphericalOrbital(l, args.pop(0))

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
            raise ValueError(self.__class__.__name__ + ' does not implement shell h and above!')

        s = kwargs.get('spherical', None)
        if s is None:
            # Expect the orbital to already be set
            pass
        elif isinstance(s, Orbital):
            self.orb = s
        else:
            self.orb = SphericalOrbital(l, s)

        if self.orb is None:
            raise ValueError(self.__class__.__name__ + " is not initialized with an "
                             "orbital which contains the radial function.")

        self.R = self.orb.R

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
            same &= self.Z == other.Z
            same &= self.P == other.P
        elif isinstance(other, Orbital):
            same = self.orb.equal(other)
        else:
            return False
        return same

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
                return name + r'\zeta^{}\mathrm{{P}}'.format(self.Z)
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
            return self.__class__.__name__ + '{{{0}, tag: {1}, {2}}}'.format(self.name(), tag, repr(self.orb))
        return self.__class__.__name__ + '{{{0}, {1}}}'.format(self.name(), repr(self.orb))

    def set_radial(self, *args):
        """ Update the internal radial function used as a :math:`f(|\mathbf r|)`

        See `SphericalOrbital.set_radial` where these arguments are passed to.
        """
        self.orb.set_radial(*args)
        self.R = self.orb.R

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
        f : the orbital value at point `r`
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
        psi : the orbital value at point `r`
        """
        return self.orb.psi(r, self.m)

    def __getstate__(self):
        """ Return the state of this object """
        # A function is not necessarily pickable, so we store interpolated
        # data which *should* ensure the correct pickable state (to close agreement)
        r = np.linspace(0, self.R, 1000)
        f = self.orb.f(r)
        return {'name': self.name(), 'r': r, 'f': f, 'tag': self.tag}

    def __setstate__(self, d):
        """ Re-create the state of this object """
        self.__init__(d['name'], (d['r'], d['f']), d['tag'])
