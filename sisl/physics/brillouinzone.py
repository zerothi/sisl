""" Different tools for contsructing k-points and paths in the Brillouin zone """

from __future__ import print_function, division

import warnings
from numbers import Integral

from numpy import pi
import numpy as np
from numpy import sum, dot

from sisl.supercell import SuperCell, SuperCellChild


__all__ = ['BrillouinZone', 'PathBZ']


class BrillouinZone(SuperCellChild):
    """ A class to construct Brillouin zone related quantities

    It takes a super-cell as an argument and can then return
    the k-points in non-reduced units from reduced units.
    """

    def __init__(self, sc):
        """ Initialize a `BrillouinZone` object from a given `SuperCell`

        Parameters
        ----------
        sc : SuperCell or array_like
           the attached supercell
        """
        try:
            if isinstance(sc.sc, SuperCell):
                self.set_supercell(sc.sc)
        except:
            self.set_supercell(sc)

    def kb(self, k):
        """ Return the k-point in reduced coordinates """
        return sum(dot(k, self.cell) * 0.5 / pi, axis=0)

    def k(self, kb):
        """ Return the k-point in 1/Ang coordinates """
        return dot(self.rcell, kb)

    def __call__(self, obj, *args, **kwargs):
        """ Return the eigenspectrum for the object passed 

        Parameters
        ----------
        obj : object to return the eigenvalues of
            this object is required to have a function called  
        reduced : bool
           whether the input k-point is in reduced coordinates
           If `True` it returns the k-points according to the 
           supercell, else it returns the reduced k-point.

        Returns
        -------
        eig : an iterator for all the eigenvalues
        """
        # This could be of substantial size, so we yield the values
        for k in self:
            yield obj.eigh(k, *args, **kwargs)

    def __iter__(self):
        """ Returns all k-points associated with this Brillouin zone object

        The default `BrillouinZone` class only has the Gamma point
        """
        yield np.zeros([3], np.float64)

    def __len__(self):
        return 1


class PathBZ(BrillouinZone):
    """ Create a path in the Brillouin zone for plotting band-structures etc. """

    def __init__(self, sc, point, division, name=None):
        """ Instantiate the `PathBZ` by a set of special `points` separated in `divisions`

        Parameters
        ----------
        sc : SuperCell or array_like
           the unit-cell of the Brillouin zone
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
        self.set_supercell(sc)

        # Copy over points
        self.point = np.array(point, dtype=np.float64)
        nk = len(self.point)

        # If the array has fewer points we try and determine
        if self.point.shape[1] < 3:
            if self.point.shape[1] != np.sum(self.nsc > 1):
                raise ValueError('Could not determine the non-periodic direction')

            # fix the points where there are no periodicity
            for i in [0, 1, 2]:
                if self.nsc[i] == 1:
                    self.point = np.insert(self.point, i, 0., axis=1)

        # Ensure the shape is correct
        self.point.shape = (-1, 3)

        # Now figure out what to do with the divisions
        if isinstance(division, Integral):

            # Calculate points (we need correct units for distance)
            kpts = [self.k(pnt) for pnt in self.point]
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
        n = len(self.point)
        xtick = np.zeros(n, np.float64)

        ii = 0
        for i in range(n-1):
            xtick[i] = ii
            ii += self.division[i]

        # Final tick-mark
        xtick[n-1] = ii - 1

        # Get label tick
        label_tick = [a for a in self.name]

        return xtick, label_tick

    def lineark(self, ticks=False):
        """ A 1D array which corresponds to the delta-k values of the path

        This is meant for plotting

        Examples
        --------

        >>> p = PathBZ(...)
        >>> eigs = Hamiltonian.eigh(p)
        >>> for i in range(len(Hamiltonian)):
        >>>     pyplot.plot(p.lineark(), eigs[:, i])

        Parameters
        ----------
        ticks : bool
           if `True` the ticks for the points are also returned

           xticks, label_ticks, lk = PathBZ.lineark(True)

        """

        nk = len(self)
        # Calculate points
        k = [self.k(pnt) for pnt in self.point]
        dk = np.diff(k, axis=0)
        xtick = np.zeros(len(k), np.float64)
        # Prepare output array
        dK = np.empty(nk, np.float64)

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
            return xtick, label_tick, dK
        return dK

    def __len__(self):
        return sum(self.division)
