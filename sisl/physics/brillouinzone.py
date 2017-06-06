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
        self.set_supercell(sc)

    def kb(self, k):
        """ Return the k-point in reduced coordinates """
        return sum(dot(k, self.cell) * 0.5 / pi, axis=0)

    def k(self, kb):
        """ Return the k-point in 1/Ang coordinates """
        return dot(self.rcell, kb)

    def __call__(self, k, reduced=True):
        """ Return the k-point in 1/Ang coordinates

        Parameters
        ----------
        k : array_like of float
           the input k-point (in units determined by `reduced`)
        reduced : bool
           whether the input k-point is in reduced coordinates
           If `True` it returns the k-points according to the 
           supercell, else it returns the reduced k-point.
        """
        if reduced:
            return self.k(k)
        return self.kb(k)

    def __iter__(self):
        """ Returns all k-points associated with this Brillouin zone object

        The default `BrillouinZone` class only has the Gamma point
        """
        yield np.zeros([3], np.float64)

    def __len__(self):
        return 1


class PathBZ(BrillouinZone):
    """ Create a path in the Brillouin zone for plotting band-structures etc. """

    def __init__(self, sc, points, divisions):
        """ Instantiate the `PathBZ` by a set of special `points` separated in `divisions`

        Parameters
        ----------
        sc : SuperCell or array_like
           the unit-cell of the Brillouin zone
        points : array_like of float
           a list of points that are the *corners* of the path
        divisions : int or array_like of int
           number of divisions in each segment. 
           If a single integer is passed it is the total number 
           of points on the path (equally separated).
           If it is an array_like input it must have length one
           less than `points`.
        """
        self.set_supercell(sc)

        # Copy over points
        self.points = np.array(points, dtype=np.float64)
        nk = len(self.points)

        # If the array has fewer points we try and determine
        if self.points.shape[1] < 3:
            if self.points.shape[1] != np.sum(self.nsc > 1):
                raise ValueError('Could not determine the non-periodic direction')

            # fix the points where there are no periodicity
            for i in [0, 1, 2]:
                if self.nsc[i] == 1:
                    self.points = np.insert(self.points, i, 0., axis=1)

        # Ensure the shape is correct
        self.points.shape = (-1, 3)

        # Now figure out what to do with the divisions
        if isinstance(divisions, Integral):

            # Calculate points (we need correct units for distance)
            kpts = [self(point) for point in self.points]
            if len(kpts) == 2:
                dists = [sum(np.diff(kpts, axis=0) ** 2) ** .5]
            else:
                dists = sum(np.diff(kpts, axis=0)**2, axis=1) ** .5
            dist = sum(dists)

            div = np.array(np.floor(dists / dist * divisions), np.int32)
            n = sum(div)
            if n < divisions:
                div[-1] +=1
                n = sum(div)
            while n < divisions:
                # Get the separation of k-points
                delta = dist / n

                idx = np.amin(dists - delta * div)
                div[idx] += 1

                n = sum(div)

            divisions = div[:]

        self.divisions = np.array(divisions, np.int32)
        self.divisions.shape = (-1,)

    def __iter__(self):
        """ Iterate through the path """

        # Calculate points
        dk = np.diff(self.points, axis=0)

        for i in range(len(dk)):

            # Calculate this delta
            if i == len(dk) - 1:
                # to get end-point
                delta = dk[i, :] / (self.divisions[i] - 1)
            else:
                delta = dk[i, :] / self.divisions[i]

            for j in range(self.divisions[i]):
                yield self.points[i] + j * delta

    def lineark(self):
        """ A 1D array which corresponds to the delta-k values of the path

        This is meant for plotting

        Examples
        --------

        >>> p = PathBZ(...)
        >>> eigs = Hamiltonian.eigh(p)
        >>> for i in range(len(Hamiltonian)):
        >>>     pyplot.plot(p.lineark(), eigs[:, i])

        """

        nk = len(self)
        # Calculate points
        k = [self(point) for point in self.points]
        dk = np.diff(k, axis=0)
        # Prepare output array
        dK = np.empty(nk, np.float64)

        ii, add = 0, 0.
        for i in range(len(dk)):
            n = self.divisions[i]

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

        return dK

    def __len__(self):
        return sum(self.divisions)
