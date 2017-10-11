from __future__ import print_function, division

import types
from numbers import Integral, Real

from numpy import pi
import numpy as np
from numpy import sum, dot

from sisl.supercell import SuperCell


__all__ = ['BrillouinZone', 'MonkhorstPackBZ', 'PathBZ']


class BrillouinZone(object):
    """ A class to construct Brillouin zone related quantities

    It takes any object as an argument and can then return
    the k-points in non-reduced units from reduced units.

    The object associated with the BrillouinZone object *has* to implement
    at least two different properties:

    1. `cell` which is the lattice vector
    2. `rcell` which is the reciprocal lattice vectors.

    The object may also be an array of floats in which case an internal
    `SuperCell` object will be created from the cell vectors (see `SuperCell` for
    details).
    """

    def __init__(self, obj):
        """ Initialize a `BrillouinZone` object from a given `SuperCell`

        Parameters
        ----------
        obj : object or array_like
           An object with associated `obj.cell` and `obj.rcell` or
           an array of floats which may be turned into a `SuperCell`
        """
        try:
            obj.cell
            obj.rcell
            self.obj = obj
        except:
            self.obj = SuperCell(obj)

        # Gamma point
        self._k = np.zeros([1, 3], np.float64)
        self._w = np.ones(1, np.float64)

        # Instantiate the array call
        self.array()

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
        return self.obj.cell

    @property
    def rcell(self):
        return self.obj.rcell

    def tocartesian(self, k):
        """ Transfer a k-point in reduced coordinates to the Cartesian coordinates

        Parameters
        ----------
        k : list of float
           k-point in reduced coordinates
        """
        return dot(self.rcell, k)

    def toreduced(self, k):
        """ Transfer a k-point in Cartesian coordinates to the reduced coordinates

        Parameters
        ----------
        k : list of float
           k-point in Cartesian coordinates
        """
        return dot(k, self.cell) * 0.5 / pi

    __attr = None

    def __getattr__(self, attr):
        try:
            getattr(self.obj, attr)
            self.__attr = attr
            return self
        except AttributeError:
            raise AttributeError("'{}' does not exist in class '{}'".format(
                attr, self.obj.__class__.__name__))

    # Implement wrapper calls
    def array(self, dtype=np.float64):
        """ Return `self` with `numpy.ndarray` returned quantities

        This forces the `__call__` routine to return a single array.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            the data-type to cast the values to

        Examples
        --------
        >>> obj = BrillouinZone(...) # doctest: +SKIP
        >>> obj.array().eigh() # doctest: +SKIP

        See Also
        --------
        yields : all output returned through an iterator
        average : take the average (with k-weights) of the Brillouin zone
        """

        def _call(self, *args, **kwargs):
            func = getattr(self.obj, self.__attr)
            for i, k in enumerate(self):
                if i == 0:
                    v = func(k, *args, **kwargs)
                    if len(self) == 1:
                        return v
                    shp = [len(self)]
                    shp.extend(v.shape)
                    a = np.empty(shp, dtype=dtype)
                    a[i, :] = v[:]
                    del v
                else:
                    a[i, :] = func(k, *args, **kwargs)
            return a
        # Set instance __call__
        self.__call__ = types.MethodType(_call, self)
        return self

    def yields(self, dtype=np.float64):
        """ Return `self` with yielded quantities

        This forces the `__call__` routine to return a an iterator which may
        yield the quantities calculated.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            the data-type to cast the values to

        Examples
        --------
        >>> obj = BrillouinZone(Hamiltonian) # doctest: +SKIP
        >>> obj.yields().eigh() # doctest: +SKIP

        See Also
        --------
        array : all output as a single array
        average : take the average (with k-weights) of the Brillouin zone
        """

        def _call(self, *args, **kwargs):
            func = getattr(self.obj, self.__attr)
            for k in self:
                yield func(k, *args, **kwargs).astype(dtype, copy=False)
        # Set instance __call__
        self.__call__ = types.MethodType(_call, self)
        return self

    def average(self, dtype=np.float64):
        """ Return `self` with yielded quantities

        This forces the `__call__` routine to return a an iterator which may
        yield the quantities calculated.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            the data-type to cast the values to

        Examples
        --------
        >>> obj = BrillouinZone(Hamiltonian) # doctest: +SKIP
        >>> obj.average().eigh() # doctest: +SKIP

        >>> obj = BrillouinZone(Hamiltonian) # doctest: +SKIP
        >>> obj.average() # doctest: +SKIP
        >>> obj.eigh() # doctest: +SKIP
        >>> obj.eighs() # doctest: +SKIP

        See Also
        --------
        array : all output as a single array
        yields : all output returned through an iterator
        """

        def _call(self, *args, **kwargs):
            func = getattr(self.obj, self.__attr)
            w = self.weight[:]
            for i, k in enumerate(self):
                if i == 0:
                    v = func(k, *args, **kwargs) * w[i]
                else:
                    v += func(k, *args, **kwargs) * w[i]
            return v
        # Set instance __call__
        self.__call__ = types.MethodType(_call, self)
        return self

    mean = average

    def __call__(self, *args, **kwargs):
        """ Calls the given attribute of the internal object and returns the quantity

        Parameters
        ----------
        *args :
            arguments passed to the attribute call, note that the first argument
            will *always* be `k`
        **kwargs :
            keyword arguments passed to the attribute call, note that the first argument
            will *always* be `k`

        Returns
        -------
        getattr(self, attr)(k, *args, **kwargs) : whatever this returns
        """
        try:
            call = getattr(self, '__call__')
        except Exception as e:
            raise NotImplementedError("Could not call the object it self")
        return call(*args, **kwargs)

    def __iter__(self):
        """ Returns all k-points associated with this Brillouin zone object

        The default `BrillouinZone` class only has the Gamma point.
        """
        for k in self.k:
            yield k

    def __len__(self):
        return len(self._k)


class MonkhorstPackBZ(BrillouinZone):
    """ Create a Monkhorst-Pack grid for the Brillouin zone """

    def __init__(self, obj, nkpt, symmetry=True, displacement=None, size=None):
        r""" Instantiate the `MonkhorstPackBZ` by a number of points in each direction

        Parameters
        ----------
        obj : object or array_like
           An object with associated `obj.cell` and `obj.rcell` or
           an array of floats which may be turned into a `SuperCell`
        nktp : array_like of ints
           a list of number of k-points along each cell direction
        symmetry : bool, optional
           whether symmetry exists in the Brillouin zone.
        displacement : float or array_like of float, optional
           the displacement of the evenly spaced grid, a single floating
           number is the displacement for the 3 directions, else they
           are the individual displacements
        size : float or array_like of float, optional
           the size of the Brillouin zone sampled. This reduces the boundaries
           of the Brillouin zone to the fraction specified. I.e. `size` must
           be of values :math:`]0 ; 1]`. Default to the entire BZ.
           Note that this will also reduce the weights such that the weights
           are normalized to the entire BZ.
        """
        super(MonkhorstPackBZ, self).__init__(obj)

        if isinstance(nkpt, Integral):
            nkpt = np.diag([nkpt] * 3)
        elif isinstance(nkpt[0], Integral):
            nkpt = np.diag(nkpt)

        # Now we have a matrix of k-points
        if np.any(nkpt - np.diag(np.diag(nkpt)) != 0):
            raise NotImplementedError("MonkhorstPackBZ with off-diagonal components is not implemented yet")

        if displacement is None:
            displacement = np.zeros(3, np.float64)
        elif isinstance(displacement, Real):
            displacement = np.zeros(3, np.float64) + displacement

        if size is None:
            size = np.ones(3, np.float64)
        elif isinstance(size, Real):
            size = np.zeros(3, np.float64) + size

        # Retrieve the diagonal number of values
        Dn = np.diag(nkpt)

        # Correct for 1's where it does not
        # make sense to reduce the BZ
        size = np.where(Dn == 1, 1., size)

        def link(n, d, s):
            return (np.arange(n) * 2 - n + 1) * s / (2 * n) + d

        # Now we are ready to create the list of k-points
        self._k = np.empty([np.prod(Dn), 3], np.float64)
        self._k.shape = (Dn[0], Dn[1], Dn[2], 3)
        self._k[..., 0] = link(Dn[0], displacement[0], size[0]).reshape(-1, 1, 1)
        self._k[..., 1] = link(Dn[1], displacement[1], size[1]).reshape(1, -1, 1)
        self._k[..., 2] = link(Dn[2], displacement[2], size[2]).reshape(1, 1, -1)

        # Return to original shape
        self._k.shape = (-1, 3)
        self._k = np.where(self._k > .5, self._k - 1, self._k)
        N = len(self._k)
        # We have to correct for the size of the Brillouin zone
        self._w = np.ones([N], np.float64) * np.prod(size) / N

    def __iter__(self):
        """ Iterate through the Monkhorst pack-grid

        Yields
        ------
        k, w : k-point and associated weight
        """
        for i in range(len(self)):
            yield self._k[i], self._w[i]


class PathBZ(BrillouinZone):
    """ Create a path in the Brillouin zone for plotting band-structures etc. """

    def __init__(self, obj, point, division, name=None):
        """ Instantiate the `PathBZ` by a set of special `points` separated in `divisions`

        Parameters
        ----------
        obj : object or array_like
           An object with associated `obj.cell` and `obj.rcell` or
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
        """
        super(PathBZ, self).__init__(obj)

        # Copy over points
        self.point = np.array(point, dtype=np.float64)

        # If the array has fewer points we try and determine
        if self.point.shape[1] < 3:
            if self.point.shape[1] != np.sum(self.obj.nsc > 1):
                raise ValueError('Could not determine the non-periodic direction')

            # fix the points where there are no periodicity
            for i in [0, 1, 2]:
                if self.obj.nsc[i] == 1:
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

            div = np.array(np.floor(dists / dist * division), np.int32)
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

        self.division = np.array(division, np.int32)
        self.division.shape = (-1,)

        if name is None:
            self.name = 'ABCDEFGHIJKLMNOPQRSTUVXYZ'[:len(self.point)]
        else:
            self.name = name

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
        """ The tick-marks corresponding to the linear-k values """
        return self.lineark(True)[0:2]

    def lineark(self, ticks=False):
        """ A 1D array which corresponds to the delta-k values of the path

        This is meant for plotting

        Examples
        --------

        >>> p = PathBZ(...) # doctest: +SKIP
        >>> eigs = Hamiltonian.eigh(p) # doctest: +SKIP
        >>> for i in range(len(Hamiltonian)): # doctest: +SKIP
        >>>     pyplot.plot(p.lineark(), eigs[:, i]) # doctest: +SKIP

        Parameters
        ----------
        ticks : bool
           if `True` the ticks for the points are also returned

           xticks, label_ticks, lk = PathBZ.lineark(True)

        """
        # Calculate points
        k = [self.tocartesian(pnt) for pnt in self.point]
        dk = np.diff(k, axis=0)
        xtick = [None] * len(k)
        # Prepare output array
        dK = np.empty(len(self), np.float64)

        ii, add = 0, 0.
        for i in range(len(dk)):
            xtick[i] = ii
            n = self.division[i]

            # Calculate the delta along this segment
            delta = sum(dk[i, :] ** 2) ** .5

            if i == len(dk) - 1:
                # to get end-point correctly
                delta /= n - 1
            else:
                delta /= n
            dK[ii:ii+n] = np.linspace(add, add + delta * n, n, dtype=np.float64)
            ii += n

            # Calculate the next separation
            # The addition is the latest delta point plus the
            # missing delta to reach the starting point for the
            # next point in the BZ
            add = dK[ii-1] + delta

        # Final tick-mark
        xtick[len(dk)] = ii - 1

        # Get label tick
        label_tick = [a for a in self.name]
        if ticks:
            return dK[xtick], label_tick, dK
        return dK

    def __len__(self):
        return sum(self.division)
