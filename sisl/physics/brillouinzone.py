"""Brillouin zone classes
=========================

.. module:: sisl.physics.brillouinzone
   :noindex:

The Brillouin zone objects are all special classes enabling easy manipulation
of an underlying physical quantity.

Quite often a physical quantity will be required to be averaged, or calculated individually
over a number of k-points. In this regard can the Brillouin zone objects help.

A basic principle of the BrillouinZone objects is that *any* method called on a BrillouinZone
object will defer to the attached parent to the class. Lets take an example.

>>> H = Hamiltonian(...)
>>> bz = BrillouinZone(H)
>>> bz.eigh()

This will actually calculate the eigenvalues for all k-points associated with the BrillouinZone.
This may be extremely convenient when calculating band-structures:

>>> H = Hamiltonian(...)
>>> bs = BandStructure(H, [[0, 0, 0], [0.5, 0, 0]], 100)
>>> bs_eig = bs.eigh().T

and then you have all eigenvalues for all the k-points.

Sometimes one may want to post-process the data for each k-point.
As an example lets post-process the DOS on a per k-point basis.
 
>>> H = Hamiltonian(...)
>>> mp = MonkhorstPack(H, [10, 10, 10])
>>> E = np.linspace(-2, 2, 100)
>>> def wrap_DOS(eigenstate):
...    # Calculate the DOS for the eigenstates
...    DOS = eigenstate.DOS(E)
...    # Calculate the velocity for the eigenstates
...    v = eigenstate.velocity()
...    V = (v ** 2).sum(1)
...    return DOS.reshape(-1, 1) * v ** 2 / V.reshape(-1, 1)
>>> DOS = mp.asaverage().eigenstate(wrap=wrap_DOS, eta=True)

This will, calculate the Monkhorst pack k-averaged DOS split into 3 Cartesian
directions based on the eigenstates velocity direction. This method of manipulating
the result can be extremely powerful to calculate many quantities while running an
efficient BrillouinZone average. The `eta` flag will print, to stdout, a progress-bar.
The usage of the ``wrap`` method are also passed optional arguments, ``parent`` which is
``H`` in the above example. ``k`` and ``weight`` are the current k-point and weight of the
corresponding k-point. An example could be to manipulate the DOS depending on the k-point and
weight:

>>> H = Hamiltonian(...)
>>> mp = MonkhorstPack(H, [10, 10, 10])
>>> E = np.linspace(-2, 2, 100)
>>> def wrap_DOS(eigenstate, k, weight):
...    # Calculate the DOS for the eigenstates and weight by k_x and weight
...    return eigenstate.DOS(E) * k[0] * weight
>>> DOS = mp.assum().eigenstate(wrap=wrap_DOS, eta=True)

When using wrap to calculate more than one quantity per eigenstate it may be advantageous
to use `~sisl.oplist` to handle cases of `BrillouinZone.asaverage` and `BrillouinZone.assum`.

>>> H = Hamiltonian(...)
>>> mp = MonkhorstPack(H, [10, 10, 10])
>>> E = np.linspace(-2, 2, 100)
>>> def wrap_multiple(eigenstate):
...    # Calculate DOS/PDOS for eigenstates
...    DOS = eigenstate.DOS(E)
...    PDOS = eigenstate.PDOS(E)
...    # Calculate velocity for the eigenstates
...    v = eigenstate.velocity()
...    return oplist([DOS, PDOS, v])
>>> DOS, PDOS, v = mp.asaverage().eigenstate(wrap=wrap_multiple, eta=True)

Which does mathematical operations (averaging/summing) using `~sisl.oplist`.

.. autosummary::
   :toctree:

   BrillouinZone
   MonkhorstPack
   BandStructure

"""
from __future__ import print_function, division

import types
from numbers import Integral, Real

from numpy import pi
import numpy as np
from numpy import sum, dot, cross

from sisl.oplist import oplist
from sisl.unit import units
from sisl.quaternion import Quaternion
from sisl.utils.mathematics import cart2spher, fnorm
from sisl.utils.misc import allow_kwargs
import sisl._array as _a
from sisl.messages import info, SislError, tqdm_eta
from sisl.supercell import SuperCell
from sisl.grid import Grid


__all__ = ['BrillouinZone', 'MonkhorstPack', 'BandStructure']


class BrillouinZone(object):
    """ A class to construct Brillouin zone related quantities

    It takes any object (which has access to cell-vectors) as an argument
    and can then return the k-points in non-reduced units from reduced units.

    The object associated with the BrillouinZone object *has* to implement
    at least two different properties:

    1. `cell` which is the lattice vector
    2. `rcell` which is the reciprocal lattice vectors.

    The object may also be an array of floats in which case an internal
    `SuperCell` object will be created from the cell vectors (see `SuperCell` for
    details).

    Parameters
    ----------
    parent : object or array_like
       An object with associated ``parent.cell`` and ``parent.rcell`` or
       an array of floats which may be turned into a `SuperCell`
    k : array_like, optional
       k-points that this Brillouin zone represents
    weight : array_like, optional
       weights for the k-points. Must have the same length as `k`.
    """

    def __init__(self, parent, k=None, weight=None):
        self.set_parent(parent)

        # Gamma point
        if k is None:
            self._k = _a.zerosd([1, 3])
            self._w = _a.onesd(1)
        else:
            self._k = _a.arrayd(k).reshape(-1, 3)
            if weight is None:
                n = self._k.shape[0]
                self._w = _a.fulld(n, 1. / n)
            else:
                self._w = _a.arrayd(weight).ravel()
        if len(self.k) != len(self.weight):
            raise ValueError(self.__class__.__name__ + '.__init__ requires input k-points and weights to be of equal length.')

        # Instantiate the array call
        self.asarray()

    def set_parent(self, parent):
        """ Update the parent associated to this object

        Parameters
        ----------
        parent : object or array_like
           an object containing cell vectors
        """
        try:
            # It probably has the supercell attached
            parent.cell
            parent.rcell
            self.parent = parent
        except:
            self.parent = SuperCell(parent)

    def __getstate__(self):
        """ Return dictionary with the current state """
        return {
            'parent_class': self.parent.__class__,
            'parent': self.parent.__getstate__(),
            'k': self._k.copy(),
            'weight': self._w.copy()
        }

    def __setstate__(self, state):
        """ Reset state of the object """
        self._k = state['k']
        self._w = state['weight']
        parent = state['parent_class'].__new__(state['parent_class'])
        parent.__setstate__(state['parent'])
        self.set_parent(parent)

    def __str__(self):
        """ String representation of the BrillouinZone """
        if isinstance(self.parent, SuperCell):
            return self.__class__.__name__ + '{{nk: {},\n {}\n}}'.format(len(self), str(self.parent).replace('\n', '\n '))
        return self.__class__.__name__ + '{{nk: {},\n {}\n}}'.format(len(self), str(self.parent.sc).replace('\n', '\n '))

    @classmethod
    def parametrize(self, sc, func, N, *args, **kwargs):
        """ Generate a new `BrillouinZone` object with k-points parameterized via the function `func` in `N` separations

        Generator of a parameterized Brillouin zone object that contains a parameterized k-point
        list.

        Basically this generates a new BrillouinZone object as:

        >>> def func(sc, frac):
        ...    return [frac, 0, 0]
        >>> bz = BrillouinZone.parametrize(1, func, 10)
        >>> len(bz) == 10
        True
        >>> np.allclose(bz.k[-1, :], [9./10, 0, 0])
        True

        Parameters
        ----------
        sc : SuperCell, or SuperCellChild
           the supercell used to construct the k-points
        func : callable
           method that parameterizes the k-points, *must* at least accept two arguments, ``sc``
           (super-cell object containing the unit-cell and reciprocal cell) and ``frac``
           (current parametrization fraction, between 0 and ``(N-1)/N``. It must return
           a k-point in 3 dimensions.
        N : int
           number of k-points generated using the parameterization
        args : list of arguments
           arguments passed directly to `func`
        kwargs : dictionary of arguments
           keyword arguments passed directly to `func`
        """
        k = np.empty([N, 3], np.float64)
        for i in range(N):
            k[i, :] = func(sc, i / N, *args, **kwargs)
        return BrillouinZone(sc, k)

    @classmethod
    def param_circle(self, sc, N_or_dk, kR, normal, origo, loop=False):
        r""" Create a parameterized k-point list where the k-points are generated on a circle around an origo

        The generated circle is a perfect circle in the reciprocal space (Cartesian coordinates).
        To generate a perfect circle in units of the reciprocal lattice vectors one can
        generate the circle for a diagonal supercell with side-length :math:`2\pi`, see
        example below.

        Parameters
        ----------
        sc : SuperCell, or SuperCellChild
           the supercell used to construct the k-points
        N_or_dk : int
           number of k-points generated using the parameterization (if an integer),
           otherwise it specifies the discretization length on the circle (in 1/Ang),
           If the latter case will use less than 4 points a warning will be raised and
           the number of points increased to 4.
        kR : float
           radius of the k-point. In 1/Ang
        normal : array_like of float
           normal vector to determine the circle plane
        origo : array_like of float
           origo of the circle used to generate the circular parameterization
        loop : bool, optional
           whether the first and last point are equal

        Examples
        --------

        >>> sc = SuperCell([1, 1, 10, 90, 90, 60])
        >>> bz = BrillouinZone.param_circle(sc, 10, 0.05, [0, 0, 1], [1./3, 2./3, 0])

        To generate a circular set of k-points in reduced coordinates (reciprocal

        >>> sc = SuperCell([1, 1, 10, 90, 90, 60])
        >>> bz = BrillouinZone.param_circle(sc, 10, 0.05, [0, 0, 1], [1./3, 2./3, 0])
        >>> bz_rec = BrillouinZone.param_circle(2*np.pi, 10, 0.05, [0, 0, 1], [1./3, 2./3, 0])
        >>> bz.k[:, :] = bz_rec.k[:, :]

        Returns
        -------
        BrillouinZone : with the parameterized k-points.
        """
        if isinstance(N_or_dk, Integral):
            N = N_or_dk
        else:
            # Calculate the required number of points
            N = int(kR ** 2 * np.pi / N_or_dk + 0.5)
            if N < 4:
                N = 4
                info('BrillouinZone.param_circle increased the number of circle points to 4.')

        # Conversion object
        bz = BrillouinZone(sc)

        normal = _a.arrayd(normal)
        origo = _a.arrayd(origo)
        k_n = bz.tocartesian(normal)
        k_o = bz.tocartesian(origo)

        # Generate a preset list of k-points on the unit-circle
        if loop:
            radians = _a.aranged(N) / (N-1) * 2 * np.pi
        else:
            radians = _a.aranged(N) / N * 2 * np.pi
        k = _a.emptyd([N, 3])
        k[:, 0] = np.cos(radians)
        k[:, 1] = np.sin(radians)
        k[:, 2] = 0.

        # Now generate the rotation
        _, theta, phi = cart2spher(k_n)
        if theta != 0:
            pv = _a.arrayd([k_n[0], k_n[1], 0])
            pv /= fnorm(pv)
            q = Quaternion(phi, pv, rad=True) * Quaternion(theta, [0, 0, 1], rad=True)
        else:
            q = Quaternion(0., [0, 0, k_n[2] / abs(k_n[2])], rad=True)

        # Calculate k-points
        k = q.rotate(k)
        k *= kR / fnorm(k).reshape(-1, 1)
        k = bz.toreduced(k + k_o)

        # The sum of weights is equal to the BZ area
        W = np.pi * kR ** 2
        w = np.repeat([W / N], N)

        return BrillouinZone(sc, k, w)

    def copy(self):
        """ Create a copy of this object """
        bz = self.__class__(self.parent, self._k, self.weight)
        bz._k = self._k.copy()
        bz._w = self._w.copy()
        return bz

    @property
    def k(self):
        """ A list of all k-points (if available) """
        return self._k

    @property
    def weight(self):
        """ Weight of the k-points in the `BrillouinZone` object """
        return self._w

    @property
    def cell(self):
        return self.parent.cell

    @property
    def rcell(self):
        return self.parent.rcell

    def tocartesian(self, k):
        """ Transfer a k-point in reduced coordinates to the Cartesian coordinates

        Parameters
        ----------
        k : list of float
           k-point in reduced coordinates

        Returns
        -------
        k : in units of 1/Ang
        """
        return dot(k, self.rcell)

    def toreduced(self, k):
        """ Transfer a k-point in Cartesian coordinates to the reduced coordinates

        Parameters
        ----------
        k : list of float
           k-point in Cartesian coordinates

        Returns
        -------
        k : in units of reciprocal lattice vectors ]-0.5 ; 0.5] (if k is in the primitive cell)
        """
        return dot(k, self.cell.T / (2 * pi))

    @staticmethod
    def in_primitive(k):
        """ Move the k-point into the primitive point(s) ]-0.5 ; 0.5]

        Parameters
        ----------
        k : array_like
           k-point(s) to move into the primitive cell

        Returns
        -------
        k : all k-points moved into the primitive cell
        """
        k = _a.arrayd(k) % 1.

        # Ensure that we are in the interval ]-0.5; 0.5]
        idx = (k.ravel() > 0.5).nonzero()[0]
        while len(idx) > 0:
            k[np.unravel_index(idx, k.shape)] -= 1.
            idx = (k.ravel() > 0.5).nonzero()[0]

        idx = (k.ravel() <= -0.5).nonzero()[0]
        while len(idx) > 0:
            k[np.unravel_index(idx, k.shape)] += 1.
            idx = (k.ravel() <= -0.5).nonzero()[0]

        return k

    _bz_attr = None

    def __getattr__(self, attr):
        try:
            getattr(self.parent, attr)
            self._bz_attr = attr
            return self
        except AttributeError:
            raise AttributeError("'{}' does not exist in class '{}'".format(
                attr, self.parent.__class__.__name__))

    def _bz_get_func(self):
        """ Internal method to retrieve the actual function to be called """
        if callable(self._bz_attr):
            return self._bz_attr
        return getattr(self.parent, self._bz_attr)

    def call(self, func, *args, **kwargs):
        """ Call the function `func` and run as though the function has been called

        This is a wrapper to call user-defined functions not attached to the parent
        object.

        The below example shows that the equivalence of the call.

        Examples
        --------
        >>> H = Hamiltonian(...)
        >>> bz = BrillouinZone(H)
        >>> bz.eigh() == bz.call(H.eigh)

        Parameters
        ----------
        func : callable
           method used
        *args :
           arguments passed to func in the call sequence
        **kwargs :
           keyword arguments passed to func in the call sequence
        """
        self._bz_attr = func
        return self(*args, **kwargs)

    # Implement wrapper calls
    def asarray(self):
        """ Return `self` with `numpy.ndarray` returned quantities

        This forces the `__call__` routine to return a single array.

        Notes
        -----
        All invocations of sub-methods are added these keyword-only arguments:

        eta : bool, optional
           if true a progress-bar is created, default false.
        wrap : callable, optional
           a function that accepts the output of the given routine and post-process
           it. Defaults to ``lambda x: x``.

        Examples
        --------
        >>> obj = BrillouinZone(...)
        >>> obj.asarray().eigh(eta=True)

        To compute multiple things in one go one should use wrappers to contain
        the calculation

        >>> E = np.linspace(-2, 2, 100)
        >>> dist = get_distribution('gaussian', smearing=0.1)
        >>> def wrap(es, parent, k, weight):
        ...    DOS = es.DOS(E, distribution=dist)
        ...    PDOS = es.PDOS(E, distribution=dist)
        ...    occ = es.occupation()
        ...    spin_moment = (es.spin_moment(E, distribution=dist) * occ.reshape(-1, 1)).sum(0)
        ...    return oplist([DOS, PDOS, spin_moment])
        >>> bz = BrillouinZone(hamiltonian)
        >>> DOS, PDOS, spin_moment = bz.asaverage().eigenstate(wrap=wrap)

        See Also
        --------
        asyield : all output returned through an iterator
        asaverage : take the average (with k-weights) of the Brillouin zone
        assum : return the sum of values in the Brillouin zone
        aslist : all output returned as a Python list
        """

        def _call(self, *args, **kwargs):
            func = self._bz_get_func()
            has_wrap = 'wrap' in kwargs
            if has_wrap:
                wrap = allow_kwargs('parent', 'k', 'weight')(kwargs.pop('wrap'))
            eta = tqdm_eta(len(self), self.__class__.__name__ + '.asarray',
                           'k', kwargs.pop('eta', False))
            parent = self.parent
            k = self.k
            w = self.weight
            if has_wrap:
                v = wrap(func(*args, k=k[0], **kwargs), parent=parent, k=k[0], weight=w[0])
            else:
                v = func(*args, k=k[0], **kwargs)
            if v.ndim == 0:
                a = np.empty([len(self)], dtype=v.dtype)
            else:
                a = np.empty((len(self), ) + v.shape, dtype=v.dtype)
            a[0] = v
            del v
            eta.update()
            if has_wrap:
                for i in range(1, len(k)):
                    a[i] = wrap(func(*args, k=k[i], **kwargs), parent=parent, k=k[i], weight=w[i])
                    eta.update()
            else:
                for i in range(1, len(k)):
                    a[i] = func(*args, k=k[i], **kwargs)
                    eta.update()
            eta.close()
            return a
        # Set instance __bz_call
        setattr(self, '_bz_call', types.MethodType(_call, self))
        return self

    def asnone(self):
        """ Return `self` with None, this may be done for instance when wrapping the function calls.

        This forces the `__call__` routine to return ``None``. This usage is mainly intended when
        creating custom `wrap` function calls.

        Notes
        -----
        All invocations of sub-methods are added these keyword-only arguments:

        eta : bool, optional
           if true a progress-bar is created, default false.
        wrap : callable, optional
           a function that accepts the output of the given routine and post-process
           it. Defaults to ``lambda x: x``.

        Examples
        --------
        >>> obj = BrillouinZone(...)
        >>> obj.asnone().eigh(eta=True)

        See Also
        --------
        asyield : all output returned through an iterator
        asaverage : take the average (with k-weights) of the Brillouin zone
        assum : return the sum of values in the Brillouin zone
        aslist : all output returned as a Python list
        """

        def _call(self, *args, **kwargs):
            func = self._bz_get_func()
            wrap = allow_kwargs('parent', 'k', 'weight')(kwargs.pop('wrap', lambda x: x))
            eta = tqdm_eta(len(self), self.__class__.__name__ + '.asnone',
                           'k', kwargs.pop('eta', False))
            parent = self.parent
            k = self.k
            w = self.weight
            for i in range(len(k)):
                wrap(func(*args, k=k[i], **kwargs), parent=parent, k=k[i], weight=w[i])
                eta.update()
            eta.close()
        # Set instance __call__
        setattr(self, '_bz_call', types.MethodType(_call, self))
        return self

    def aslist(self):
        """ Return `self` with `list` returned quantities

        This forces the `__call__` routine to return a list with returned values.

        Notes
        -----
        All invocations of sub-methods are added these keyword-only arguments:

        eta : bool, optional
           if true a progress-bar is created, default false.
        wrap : callable, optional
           a function that accepts the output of the given routine and post-process
           it. Defaults to ``lambda x: x``.

        Examples
        --------
        >>> obj = BrillouinZone(...)
        >>> def first_ten(es):
        ...    return es.sub(range(10))
        >>> obj.aslist().eigenstate(eta=True, wrap=first_ten)

        See Also
        --------
        asarray : all output as a single array
        asyield : all output returned through an iterator
        assum : return the sum of values in the Brillouin zone
        asaverage : take the average (with k-weights) of the Brillouin zone
        """

        def _call(self, *args, **kwargs):
            func = self._bz_get_func()
            has_wrap = 'wrap' in kwargs
            if has_wrap:
                wrap = allow_kwargs('parent', 'k', 'weight')(kwargs.pop('wrap'))
            eta = tqdm_eta(len(self), self.__class__.__name__ + '.aslist',
                           'k', kwargs.pop('eta', False))
            a = [None] * len(self)
            parent = self.parent
            k = self.k
            w = self.weight
            if has_wrap:
                for i in range(len(k)):
                    a[i] = wrap(func(*args, k=k[i], **kwargs), parent=parent, k=k[i], weight=w[i])
                    eta.update()
            else:
                for i in range(len(k)):
                    a[i] = func(*args, k=k[i], **kwargs)
                    eta.update()
            eta.close()
            return a
        # Set instance __call__
        setattr(self, '_bz_call', types.MethodType(_call, self))
        return self

    def asyield(self):
        """ Return `self` with yielded quantities

        This forces the `__call__` routine to return a an iterator which may
        yield the quantities calculated.

        Notes
        -----
        All invocations of sub-methods are added these keyword-only arguments:

        eta : bool, optional
           if true a progress-bar is created, default false.
        wrap : callable, optional
           a function that accepts the output of the given routine and post-process
           it. Defaults to ``lambda x: x``.

        Examples
        --------
        >>> obj = BrillouinZone(Hamiltonian)
        >>> obj.asyield().eigh(eta=True)

        See Also
        --------
        asarray : all output as a single array
        asaverage : take the average (with k-weights) of the Brillouin zone
        assum : return the sum of values in the Brillouin zone
        aslist : all output returned as a Python list
        """

        def _call(self, *args, **kwargs):
            func = self._bz_get_func()
            has_wrap = 'wrap' in kwargs
            if has_wrap:
                wrap = allow_kwargs('parent', 'k', 'weight')(kwargs.pop('wrap'))
            eta = tqdm_eta(len(self), self.__class__.__name__ + '.asyield',
                           'k', kwargs.pop('eta', False))
            parent = self.parent
            k = self.k
            w = self.weight
            if has_wrap:
                for i in range(len(k)):
                    yield wrap(func(*args, k=k[i], **kwargs), parent=parent, k=k[i], weight=w[i])
                    eta.update()
            else:
                for i in range(len(k)):
                    yield func(*args, k=k[i], **kwargs)
                    eta.update()
            eta.close()
        # Set instance __call__
        setattr(self, '_bz_call', types.MethodType(_call, self))
        return self

    def asaverage(self):
        """ Return `self` with k-averaged quantities

        This forces the `__call__` routine to return a single k-averaged value.

        Notes
        -----
        All invocations of sub-methods are added these keyword-only arguments:

        eta : bool, optional
           if true a progress-bar is created, default false.
        wrap : callable, optional
           a function that accepts the output of the given routine and post-process
           it. Defaults to ``lambda x: x``.

        Examples
        --------
        >>> obj = BrillouinZone(Hamiltonian)
        >>> obj.asaverage().DOS(np.linspace(-2, 2, 100))

        >>> obj = BrillouinZone(Hamiltonian)
        >>> obj.asaverage()
        >>> obj.DOS(np.linspace(-2, 2, 100))
        >>> obj.PDOS(np.linspace(-2, 2, 100), eta=True)

        >>> obj = BrillouinZone(Hamiltonian)
        >>> obj.asaverage()
        >>> E = np.linspace(-2, 2, 100)
        >>> def wrap(es):
        ...    return es.DOS(E), es.PDOS(E)
        >>> DOS, PDOS = obj.eigenstate(wrap=wrap)

        See Also
        --------
        asarray : all output as a single array
        asyield : all output returned through an iterator
        assum : return the sum of values in the Brillouin zone
        aslist : all output returned as a Python list
        """

        def _call(self, *args, **kwargs):
            func = self._bz_get_func()
            has_wrap = 'wrap' in kwargs
            if has_wrap:
                wrap = allow_kwargs('parent', 'k', 'weight')(kwargs.pop('wrap'))
            eta = tqdm_eta(len(self), self.__class__.__name__ + '.asaverage',
                           'k', kwargs.pop('eta', False))
            parent = self.parent
            k = self.k
            w = self.weight
            if has_wrap:
                v = wrap(func(*args, k=k[0], **kwargs), parent=parent, k=k[0], weight=w[0]) * w[0]
                eta.update()
                for i in range(1, len(k)):
                    v += wrap(func(*args, k=k[i], **kwargs), parent=parent, k=k[i], weight=w[i]) * w[i]
                    eta.update()
            else:
                v = func(*args, k=k[0], **kwargs) * w[0]
                eta.update()
                for i in range(1, len(k)):
                    v += func(*args, k=k[i], **kwargs) * w[i]
                    eta.update()
            eta.close()
            return v
        # Set instance __call__
        setattr(self, '_bz_call', types.MethodType(_call, self))
        return self

    def assum(self):
        """ Return `self` with summed quantities

        This forces the `__call__` routine to return all k-point values summed.

        Notes
        -----
        All invocations of sub-methods are added these keyword-only arguments:

        eta : bool, optional
           if true a progress-bar is created, default false.
        wrap : callable, optional
           a function that accepts the output of the given routine and post-process
           it. Defaults to ``lambda x: x``.

        Examples
        --------
        >>> obj = BrillouinZone(Hamiltonian)
        >>> obj.assum().DOS(np.linspace(-2, 2, 100))

        >>> obj = BrillouinZone(Hamiltonian)
        >>> obj.assum()
        >>> obj.DOS(np.linspace(-2, 2, 100))
        >>> obj.PDOS(np.linspace(-2, 2, 100), eta=True)

        >>> E = np.linspace(-2, 2, 100)
        >>> dist = get_distribution('gaussian', smearing=0.1)
        >>> def wrap(es, parent, k, weight):
        ...    DOS = es.DOS(E, distribution=dist) * weight
        ...    PDOS = es.PDOS(E, distribution=dist) * weight
        ...    occ = es.occupation()
        ...    spin_moment = (es.spin_moment(E, distribution=dist) * occ.reshape(-1, 1)).sum(0) * weight
        ...    return oplist([DOS, PDOS, spin_moment])
        >>> bz = BrillouinZone(hamiltonian)
        >>> DOS, PDOS, spin_moment = bz.assum().eigenstate(wrap=wrap)

        See Also
        --------
        asarray : all output as a single array
        asyield : all output returned through an iterator
        asaverage : take the average (with k-weights) of the Brillouin zone
        aslist : all output returned as a Python list
        """

        def _call(self, *args, **kwargs):
            func = self._bz_get_func()
            has_wrap = 'wrap' in kwargs
            if has_wrap:
                wrap = allow_kwargs('parent', 'k', 'weight')(kwargs.pop('wrap'))
            eta = tqdm_eta(len(self), self.__class__.__name__ + '.assum',
                           'k', kwargs.pop('eta', False))
            parent = self.parent
            k = self.k
            w = self.weight
            if has_wrap:
                v = wrap(func(*args, k=k[0], **kwargs), parent=parent, k=k[0], weight=w[0])
                if isinstance(v, tuple):
                    v = oplist(v)
                eta.update()
                for i in range(1, len(k)):
                    v += wrap(func(*args, k=k[i], **kwargs), parent=parent, k=k[i], weight=w[i])
                    eta.update()
            else:
                v = func(*args, k=k[0], **kwargs)
                if isinstance(v, tuple):
                    v = oplist(v)
                eta.update()
                for i in range(1, len(k)):
                    v += func(*args, k=k[i], **kwargs)
                    eta.update()
            eta.close()
            return v
        # Set instance __call__
        setattr(self, '_bz_call', types.MethodType(_call, self))
        return self

    def __call__(self, *args, **kwargs):
        """ Calls the given attribute of the internal object and returns the quantity

        Parameters
        ----------
        *args : optional
            arguments passed to the attribute call, note that an argument `k=k` will be
            added by this routine as a way to loop the k-points.
        **kwargs : optional
            keyword arguments passed to the attribute call, note that the first argument
            will *always* be `k`

        Returns
        -------
        getattr(self, attr)(k, *args, **kwargs) : whatever this returns
        """
        try:
            call = getattr(self, '_bz_call')
        except Exception:
            raise NotImplementedError("Could not call the object it self")
        return call(*args, **kwargs)

    def iter(self, ret_weight=False):
        """ An iterator for the k-points and (possibly) the weights

        Parameters
        ----------
        ret_weight : bool, optional
          if true, also yield the weight for the respective k-point

        Yields
        ------
        kpt : k-point
        weight : weight of k-point, only if `ret_weight` is true.
       """
        if ret_weight:
            for i in range(len(self)):
                yield self.k[i], self.weight[i]
        else:
            for k in self.k:
                yield k

    __iter__ = iter

    def __len__(self):
        return len(self._k)

    def write(self, sile, *args, **kwargs):
        """ Writes k-points to a `~sisl.io.tableSile`.

        This allows one to pass a `tableSile` or a file-name.
        """
        from sisl.io import tableSile
        kw = np.concatenate((self.k, self.weight.reshape(-1, 1)), axis=1)
        if isinstance(sile, tableSile):
            sile.write_data(kw.T, *args, **kwargs)
        else:
            with tableSile(sile, 'w') as fh:
                fh.write_data(kw.T, *args, **kwargs)


class MonkhorstPack(BrillouinZone):
    r""" Create a Monkhorst-Pack grid for the Brillouin zone

    Parameters
    ----------
    parent : object or array_like
       An object with associated `parent.cell` and `parent.rcell` or
       an array of floats which may be turned into a `SuperCell`
    nktp : array_like of ints
       a list of number of k-points along each cell direction
    displacement : float or array_like of float, optional
       the displacement of the evenly spaced grid, a single floating
       number is the displacement for the 3 directions, else they
       are the individual displacements
    size : float or array_like of float, optional
       the size of the Brillouin zone sampled. This reduces the boundaries
       of the Brillouin zone around the displacement to the fraction specified.
       I.e. `size` must be of values :math:`]0 ; 1]`. Defaults to the entire BZ.
       Note that this will also reduce the weights such that the weights
       are normalized to the entire BZ.
    centered : bool, optional
       whether the k-points are :math:`\Gamma`-centered (for zero displacement)
    trs : bool, optional
       whether time-reversal symmetry exists in the Brillouin zone.

    Examples
    --------
    >>> sc = SuperCell(3.)
    >>> MonkhorstPack(sc, 10) # 10 x 10 x 10 (with TRS)
    >>> MonkhorstPack(sc, [10, 5, 5]) # 10 x 5 x 5 (with TRS)
    >>> MonkhorstPack(sc, [10, 5, 5], trs=False) # 10 x 5 x 5 (without TRS)
    """

    def __init__(self, parent, nkpt, displacement=None, size=None, centered=True, trs=True):
        super(MonkhorstPack, self).__init__(parent)

        if isinstance(nkpt, Integral):
            nkpt = np.diag([nkpt] * 3)
        elif isinstance(nkpt[0], Integral):
            nkpt = np.diag(nkpt)

        # Now we have a matrix of k-points
        if np.any(nkpt - np.diag(np.diag(nkpt)) != 0):
            raise NotImplementedError(self.__class__.__name__ + " with off-diagonal components is not implemented yet")

        if displacement is None:
            displacement = np.zeros(3, np.float64)
        elif isinstance(displacement, Real):
            displacement = _a.fulld(3, displacement)

        if size is None:
            size = _a.onesd(3)
        elif isinstance(size, Real):
            size = _a.fulld(3, size)
        else:
            size = _a.arrayd(size)

        # Retrieve the diagonal number of values
        Dn = np.diag(nkpt).astype(np.int32)
        if np.any(Dn) == 0:
            raise ValueError(self.__class__.__name__ + ' *must* be initialized with '
                             'diagonal elements different from 0.')

        i_trs = -1
        if trs:
            # Figure out which direction to TRS
            nmax = 0
            for i in [0, 1, 2]:
                if displacement[i] in [0., 0.5] and Dn[i] > nmax:
                    nmax = Dn[i]
                    i_trs = i
            if nmax == 1:
                i_trs = -1
            if i_trs == -1:
                # If we still haven't decided (say for weird displacements)
                # simply take the one with the maximum number of k-points.
                i_trs = np.argmax(Dn)

        # Calculate k-points and weights along all directions
        kw = [self.grid(Dn[i], displacement[i], size[i], centered, i == i_trs) for i in (0, 1, 2)]

        self._k = _a.emptyd((kw[0][0].size, kw[1][0].size, kw[2][0].size, 3))
        self._w = _a.onesd(self._k.shape[:-1])
        for i in (0, 1, 2):
            k = kw[i][0].reshape(-1, 1, 1)
            w = kw[i][1].reshape(-1, 1, 1)
            self._k[..., i] = np.rollaxis(k, 0, i + 1)
            self._w[...] *= np.rollaxis(w, 0, i + 1)

        del kw
        self._k.shape = (-1, 3)
        self._k = np.where(self._k > .5, self._k - 1, self._k)
        self._w.shape = (-1,)

        # Store information regarding size and diagonal elements
        # This information is basically only necessary when
        # we want to replace special k-points
        self._diag = Dn # vector
        self._displ = displacement # vector
        self._size = size # vector
        self._centered = centered
        self._trs = i_trs

    def __str__(self):
        """ String representation of MonkhorstPack """
        if isinstance(self.parent, SuperCell):
            p = self.parent
        else:
            p = self.parent.sc
        return ('{cls}{{nk: {nk:d}, size: [{size[0]:.3f} {size[1]:.3f} {size[0]:.3f}], trs: {trs},'
                '\n diagonal: [{diag[0]:d} {diag[1]:d} {diag[2]:d}], displacement: [{disp[0]:.3f} {disp[1]:.3f} {disp[2]:.3f}],'
                '\n {sc}\n}}').format(cls=self.__class__.__name__, nk=len(self),
                                      size=self._size, trs={0: 'A', 1: 'B', 2: 'C'}.get(self._trs, 'no'),
                                      diag=self._diag, disp=self._displ, sc=str(p).replace('\n', '\n '))

    def __getstate__(self):
        """ Return dictionary with the current state """
        state = super(MonkhorstPack, self).__getstate__()
        state['diag'] = self._diag
        state['displ'] = self._displ
        state['size'] = self._size
        state['centered'] = self._centered
        state['trs'] = self._trs
        return state

    def __setstate__(self, state):
        """ Reset state of the object """
        super(MonkhorstPack, self).__setstate__(state)
        self._diag = state['diag']
        self._displ = state['displ']
        self._size = state['size']
        self._centered = state['centered']
        self._trs = state['trs']

    def copy(self):
        """ Create a copy of this object """
        bz = self.__class__(self.parent, self._diag, self._displ, self._size, self._centered, self._trs >= 0)
        bz._k = self._k.copy()
        bz._w = self._w.copy()
        return bz

    def asgrid(self):
        """ Return `self` with Grid quantities

        This forces the `__call__` routine to return all k-point values in a regular grid.

        The calculation of values on a grid requires some careful thought before
        running the calculation as the returned grid may be somewhat difficult
        to comprehend.

        Notes
        -----
        All invocations of sub-methods are added these keyword-only arguments:

        eta : bool, optional
           if true a progress-bar is created, default false.
        wrap : callable, optional
           a function that accepts the output of the given routine and post-process
           it. Defaults to ``lambda x: x``.
        data_axis : int, optional
           the Grid axis to put in the data values in. Has to be specified if the
           subsequent routine calls return more than 1 data-point per k-point.
        grid_unit : {'b', 'Ang', 'Bohr'}, optional
           for 'b' the returned grid will be a cube, otherwise the grid will be the reciprocal lattice
           vectors (for any other value) and in the given reciprocal unit ('Ang' => 1/Ang)

        Examples
        --------
        >>> obj = MonkhorstPack(Hamiltonian, [10, 1, 10])
        >>> grid = obj.asgrid().eigh(data_axis=1)

        See Also
        --------
        asarray : all output as a single array
        asyield : all output returned through an iterator
        asaverage : take the average (with k-weights) of the Brillouin zone
        aslist : all output returned as a Python list
        """

        def _call(self, *args, **kwargs):
            data_axis = kwargs.pop('data_axis', None)
            grid_unit = kwargs.pop('grid_unit', 'b')

            func = self._bz_get_func()
            wrap = allow_kwargs('parent', 'k', 'weight')(kwargs.pop('wrap', lambda x: x))
            eta = tqdm_eta(len(self), self.__class__.__name__ + '.asgrid',
                           'k', kwargs.pop('eta', False))
            parent = self.parent
            k = self.k
            w = self.weight

            # Extract information from the MP grid, these values
            # define the Grid size, etc.
            diag = self._diag.copy()
            if not np.all(self._displ == 0):
                raise SislError(self.__class__.__name__ + '.{} requires the displacement to be 0 for all k-points.'.format(self._bz_attr))
            displ = self._displ.copy()
            size = self._size.copy()
            steps = size / diag
            if self._centered:
                offset = np.where(diag % 2 == 0, steps, steps / 2)
            else:
                offset = np.where(diag % 2 == 0, steps / 2, steps)

            # Instead of doing
            #    _in_primitive(k) + 0.5 - offset
            # we can do it here
            #    _in_primitive(k) + offset'
            offset -= 0.5

            # Check the TRS direction
            trs_axis = self._trs
            _in_primitive = self.in_primitive
            _rint = np.rint
            _int32 = np.int32
            def k2idx(k):
                # In case TRS is applied two indices may be returned
                return _rint((_in_primitive(k) - offset) / steps).astype(_int32)
                # To find the opposite k-point, do this
                #  idx[i] = [diag[i] - idx[i] - 1, idx[i]
                # with i in [0, 1, 2]

            # Create cell from the reciprocal cell.
            if grid_unit == 'b':
                cell = np.diag(self._size)
            else:
                cell = parent.sc.rcell * self._size.reshape(1, -1) / units('Ang', grid_unit)

            # Find the grid origo
            origo = -(cell * 0.5).sum(0)

            # Calculate first k-point (to get size and dtype)
            v = wrap(func(*args, k=k[0], **kwargs), parent=parent, k=k[0], weight=w[0])

            if data_axis is None:
                if v.size != 1:
                    raise SislError(self.__class__.__name__ + '.{} requires one value per-kpoint because of the 3D grid values'.format(self._bz_attr))

            else:

                # Check the weights
                weights = self.grid(diag[data_axis], displ[data_axis], size[data_axis],
                                    centered=self._centered, trs=trs_axis == data_axis)[1]

                # Correct the Grid size
                diag[data_axis] = len(v)
                # Create the orthogonal cell direction to ensure it is orthogonal
                # Since array axis is cyclic for negative numbers, we simply do this
                cell[data_axis, :] = cross(cell[data_axis-1, :], cell[data_axis-2, :])
                # Check whether we should rotate it
                if cart2spher(cell[data_axis, :])[2] > pi / 4:
                    cell[data_axis, :] *= -1

            # Correct cell for the grid
            if trs_axis >= 0:
                origo[trs_axis] = 0.
                # Correct offset since we only have the positive halve
                if self._diag[trs_axis] % 2 == 0 and not self._centered:
                    offset[trs_axis] = steps[trs_axis] / 2
                else:
                    offset[trs_axis] = 0.

                # Find number of points
                if trs_axis != data_axis:
                    diag[trs_axis] = len(self.grid(diag[trs_axis], displ[trs_axis], size[trs_axis],
                                                   centered=self._centered, trs=True)[1])

            # Create the grid in the reciprocal cell
            sc = SuperCell(cell, origo=origo)
            grid = Grid(diag, sc=sc, dtype=v.dtype)
            if data_axis is None:
                grid[k2idx(k[0])] = v
            else:
                idx = k2idx(k[0]).tolist()
                weight = weights[idx[data_axis]]
                idx[data_axis] = slice(None)
                grid[idx] = v * weight

            del v

            # Now perform calculation
            eta.update()
            if data_axis is None:
                for i in range(1, len(k)):
                    grid[k2idx(k[i])] = wrap(func(*args, k=k[i], **kwargs),
                                             parent=parent, k=k[i], weight=w[i])
                    eta.update()
            else:
                for i in range(1, len(k)):
                    idx = k2idx(k[i]).tolist()
                    weight = weights[idx[data_axis]]
                    idx[data_axis] = slice(None)
                    grid[idx] = wrap(func(*args, k=k[i], **kwargs),
                                     parent=parent, k=k[i], weight=w[i]) * weight
                    eta.update()
            eta.close()
            return grid

        # Set instance __call__
        setattr(self, '_bz_call', types.MethodType(_call, self))
        return self

    @classmethod
    def grid(cls, n, displ=0., size=1., centered=True, trs=False):
        r""" Create a grid of `n` points with an offset of `displ` and sampling `size` around `displ`

        The :math:`k`-points are :math:`\Gamma` centered.

        Parameters
        ----------
        n : int
           number of points in the grid. If `trs` is ``True`` this may be smaller than `n`
        displ : float, optional
           the displacement of the grid
        size : float, optional
           the total size of the Brillouin zone to sample
        centered : bool, optional
           if the points are centered
        trs : bool, optional
           whether time-reversal-symmetry is applied

        Returns
        -------
        k : numpy.ndarray
           the list of k-points in the Brillouin zone to be sampled
        w : numpy.ndarray
           weights for the k-points
        """
        # First ensure that displ is in the Brillouin
        displ = displ % 1.
        if displ > 0.5:
            displ -= 1.
        if displ < -0.5:
            displ += 1.

        # Centered _only_ has effect IFF
        #  displ == 0. and size == 1
        # Otherwise we resort to other schemes
        if displ != 0. or size != 1.:
            centered = False

        # We create the full grid, then afterwards we figure out TRS
        n_half = n // 2
        if n % 2 == 1:
            k = _a.aranged(-n_half, n_half + 1) * size / n + displ
        else:
            k = _a.aranged(-n_half, n_half) * size / n + displ
            if not centered:
                # Shift everything by halve the size each occupies
                k += size / (2 * n)

        # Move k to the primitive cell and generate weights
        k = cls.in_primitive(k)
        w = _a.fulld(n, size / n)

        # Check for TRS points
        if trs and np.any(k < 0.):
            # Make all positive to remove the double conting terms
            k_pos = np.abs(k)

            # Sort k-points and weights
            idx = np.argsort(k_pos)

            # Re-arange according to k value
            k_pos = k_pos[idx]
            w = w[idx]

            # Find indices of all equivalent k-points (tolerance of 1e-10 in reciprocal units)
            #  1e-10 ~ 1e10 k-points (no body will do this!)
            idx_same = (np.diff(k_pos) < 1e-10).nonzero()[0]

            # The above algorithm should never create more than two duplicates.
            # Hence we can simply remove all idx_same and double the weight for all
            # idx_same + 1.
            w[idx_same + 1] *= 2
            # Delete the duplicated k-points (they are already sorted)
            k = np.delete(k_pos, idx_same, axis=0)
            w = np.delete(w, idx_same)
        else:
            # Sort them, because it makes more visual sense
            idx = np.argsort(k)
            k = k[idx]
            w = w[idx]

        # Return values
        return k, w

    def replace(self, k, mp):
        r""" Replace a k-point with a new set of k-points from a Monkhorst-Pack grid

        This method tries to replace an area corresponding to `mp.size` around the k-point `k`
        such that the k-points are replaced.
        This enables one to zoom in on specific points in the Brillouin zone for detailed analysis.

        Parameters
        ----------
        k : array_like
           k-point in this object to replace
        mp : MonkhorstPack
           object containing the replacement k-points.

        Examples
        --------

        This example creates a zoomed-in view of the :math:`\Gamma`-point by replacing it with
        a 3x3x3 Monkhorst-Pack grid.

        >>> sc = SuperCell(1.)
        >>> mp = MonkhorstPack(sc, [3, 3, 3])
        >>> mp.replace([0, 0, 0], MonkhorstPack(sc, [3, 3, 3], size=1./3))

        This example creates a zoomed-in view of the :math:`\Gamma`-point by replacing it with
        a 4x4x4 Monkhorst-Pack grid.

        >>> sc = SuperCell(1.)
        >>> mp = MonkhorstPack(sc, [3, 3, 3])
        >>> mp.replace([0, 0, 0], MonkhorstPack(sc, [4, 4, 4], size=1./3))

        This example creates a zoomed-in view of the :math:`\Gamma`-point by replacing it with
        a 4x4x1 Monkhorst-Pack grid.

        >>> sc = SuperCell(1.)
        >>> mp = MonkhorstPack(sc, [3, 3, 3])
        >>> mp.replace([0, 0, 0], MonkhorstPack(sc, [4, 4, 1], size=1./3))

        Raises
        ------
        SislError : if the size of the replacement `MonkhorstPack` grid is not compatible with the
                    k-point spacing in this object.
        """
        # First we find all k-points within k +- mp.size
        # Those are the points we wish to remove.
        # Secondly we need to ensure that the k-points we remove are occupying *exactly*
        # the Brillouin zone we wish to replace.
        if not isinstance(mp, MonkhorstPack):
            raise ValueError('Object `mp` is not a MonkhorstPack object')

        # We can easily figure out the BZ that each k-point is averaging
        k_vol = self._size / self._diag
        # Compare against the size of this one
        # Since we can remove more than one k-point, we require that the
        # size of the replacement MP is an integer multiple of the
        # k-point volumes.
        k_int = mp._size / k_vol
        if not np.allclose(np.rint(k_int), k_int):
            raise SislError(self.__class__.__name__ + '.reduce could not replace k-point, BZ '
                            'volume replaced is not equivalent to the inherent k-point volume.')
        k_int = np.rint(k_int)

        # 1. find all k-points
        k = self.in_primitive(k).reshape(1, 3)
        dk = (mp._size / 2).reshape(1, 3)
        # Find all points within [k - dk; k + dk]
        # Since the volume of each k-point is non-zero we know that no k-points will be located
        # on the boundary.
        # This does remove boundary points because we shift everything into the positive
        # plane.
        diff_k = self.in_primitive(self.k % 1. - k % 1.)
        idx = np.logical_and.reduce(np.abs(diff_k) <= dk, axis=1).nonzero()[0]
        if len(idx) == 0:
            raise SislError(self.__class__.__name__ + '.reduce could not find any points to replace.')

        # Now we have the k-points we need to remove
        # Figure out if the total weight is consistent
        total_weight = self.weight[idx].sum()
        replace_weight = mp.weight.sum()
        if abs(total_weight - replace_weight) < 1e-8:
            weight_factor = 1.
        elif abs(total_weight - replace_weight * 2) < 1e-8:
            weight_factor = 2.
            if self._trs < 0:
                info(self.__class__.__name__ + '.reduce assumes that the replaced k-point has double weights.')
        else:
            print('k-point to replace:')
            print(' ', k.ravel())
            print('delta-k:')
            print(' ', dk.ravel())
            print('Found k-indices that will be replaced:')
            print(' ', idx)
            print('k-points replaced:')
            print(self.k[idx, :])
            raise SislError(self.__class__.__name__ + '.reduce could not assert the weights are consistent during replacement.')

        self._k = np.delete(self._k, idx, axis=0)
        self._w = np.delete(self._w, idx)

        # Append the new k-points and weights
        self._k = np.concatenate((self._k, mp._k), axis=0)
        self._w = np.concatenate((self._w, mp._w * weight_factor))


class BandStructure(BrillouinZone):
    """ Create a path in the Brillouin zone for plotting band-structures etc.

    Parameters
    ----------
    parent : object or array_like
       An object with associated `parentcell` and `parent.rcell` or
       an array of floats which may be turned into a `SuperCell`
    point : array_like of float
       a list of points that are the *corners* of the path
    division : int or array_like of int
       number of divisions in each segment.
       If a single integer is passed it is the total number
       of points on the path (equally separated).
       If it is an array_like input it must have length one
       less than `point`.
    name : array_like of str
       the associated names of the points on the Brillouin Zone path

    Examples
    --------
    >>> sc = SuperCell(10)
    >>> bs = BandStructure(sc, [[0] * 3, [0.5] * 3], 200)
    >>> bs = BandStructure(sc, [[0] * 3, [0.5] * 3, [1.] * 3], 200)
    >>> bs = BandStructure(sc, [[0] * 3, [0.5] * 3, [1.] * 3], 200, ['Gamma', 'M', 'Gamma'])
    """

    def __init__(self, parent, point, division, name=None):
        super(BandStructure, self).__init__(parent)

        # Copy over points
        self.point = _a.arrayd(point)

        # If the array has fewer points we try and determine
        if self.point.shape[1] < 3:
            if self.point.shape[1] != np.sum(self.parent.nsc > 1):
                raise ValueError('Could not determine the non-periodic direction')

            # fix the points where there are no periodicity
            for i in [0, 1, 2]:
                if self.parent.nsc[i] == 1:
                    self.point = np.insert(self.point, i, 0., axis=1)

        # Ensure the shape is correct
        self.point.shape = (-1, 3)

        # Now figure out what to do with the divisions
        if isinstance(division, Integral):

            # Calculate points (we need correct units for distance)
            kpts = [self.tocartesian(pnt) for pnt in self.point]
            if len(kpts) == 2:
                dists = [sum(np.diff(kpts, axis=0) ** 2) ** .5]
            else:
                dists = sum(np.diff(kpts, axis=0)**2, axis=1) ** .5
            dist = sum(dists)

            div = np.floor(dists / dist * division).astype(dtype=np.int32)
            n = sum(div)
            if n < division:
                div[-1] +=1
                n = sum(div)
            while n < division:
                # Get the separation of k-points
                delta = dist / n

                idx = np.argmin(dists - delta * div)
                div[idx] += 1

                n = sum(div)

            division = div[:]

        self.division = _a.arrayi(division)
        self.division.shape = (-1,)

        if name is None:
            self.name = 'ABCDEFGHIJKLMNOPQRSTUVXYZ'[:len(self.point)]
        else:
            self.name = name

        self._k = _a.arrayd([k for k in self])
        self._w = _a.fulld(len(self.k), 1 / len(self.k))

    def __getstate__(self):
        """ Return dictionary with the current state """
        state = super(BandStructure, self).__getstate__()
        state['point'] = self.point.copy()
        state['division'] = self.division.copy()
        state['name'] = list(self.name)
        return state

    def __setstate__(self, state):
        """ Reset state of the object """
        super(BandStructure, self).__setstate__(state)
        self.point = state['point']
        self.division = state['division']
        self.name = state['name']

    def __iter__(self):
        """ Iterate through the path """

        # Calculate points
        dk = np.diff(self.point, axis=0)

        for i in range(len(dk)):

            # Calculate this delta
            if i == len(dk) - 1:
                # to get end-point
                delta = dk[i, :] / (self.division[i] - 1)
            else:
                delta = dk[i, :] / self.division[i]

            for j in range(self.division[i]):
                yield self.point[i] + j * delta

    def lineartick(self):
        """ The tick-marks corresponding to the linear-k values

        Returns
        -------
        linear_k : The positions in reciprocal space determined by the distance between points

        See Also
        --------
        lineark : Routine used to calculate the tick-marks.
        """
        return self.lineark(True)[1:3]

    def lineark(self, ticks=False):
        """ A 1D array which corresponds to the delta-k values of the path

        This is meant for plotting

        Examples
        --------

        >>> p = BandStructure(...)
        >>> eigs = Hamiltonian.eigh(p)
        >>> for i in range(len(Hamiltonian)):
        ...     plt.plot(p.lineark(), eigs[:, i])

        >>> p = BandStructure(...)
        >>> eigs = Hamiltonian.eigh(p)
        >>> lk, kt, kl = p.lineark(True)
        >>> plt.xticks(kt, kl)
        >>> for i in range(len(Hamiltonian)):
        ...     plt.plot(lk, eigs[:, i])

        Parameters
        ----------
        ticks : bool, optional
           if `True` the ticks for the points are also returned

           lk, xticks, label_ticks, lk = BandStructure.lineark(True)

        Returns
        -------
        linear_k : The positions in reciprocal space determined by the distance between points
        k_tick : Linear k-positions of the points, only returned if `ticks` is ``True``
        k_label : Labels at `k_tick`, only returned if `ticks` is ``True``
        """
        # Calculate points
        k = [self.tocartesian(pnt) for pnt in self.point]
        # Get difference between points
        dk = np.diff(k, axis=0)
        # Calculate the cumultative distance between points
        k_len = np.insert(_a.cumsumd((dk ** 2).sum(1) ** .5), 0, 0.)
        xtick = [None] * len(k)
        # Prepare output array
        dK = _a.emptyd(len(self))

        # short-hand
        ls = np.linspace

        xtick = np.insert(_a.cumsumi(self.division), 0, 0)
        for i in range(len(dk)):
            n = self.division[i]
            end = i == len(dk) - 1

            dK[xtick[i]:xtick[i+1]] = ls(k_len[i], k_len[i+1], n, dtype=np.float64, endpoint=end)
        xtick[-1] -= 1

        # Get label tick, in case self.name is a single string 'ABCD'
        label_tick = [a for a in self.name]
        if ticks:
            return dK, dK[xtick], label_tick
        return dK

    def __len__(self):
        return sum(self.division)
