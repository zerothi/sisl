from __future__ import print_function

import numpy as np

from ..sile import add_sile, sile_fh_open, sile_raise_write
from .sile import *

from sisl.unit.siesta import unit_convert


__all__ = ['kpSileSiesta', 'rkpSileSiesta']

Bohr2Ang = unit_convert('Bohr', 'Ang')


class kpSileSiesta(SileSiesta):
    """ k-points file in 1/Bohr units """

    @sile_fh_open()
    def read_data(self, sc=None):
        """ Returns K-points from the file (note that these are in reciprocal units)

        Parameters
        ----------
        sc : SuperCellChild, optional
           if supplied the returned k-points will be in reduced coordinates

        Returns
        -------
        k : k-points, in units 1/Bohr
        w : weights for k-points
        """
        nk = int(self.readline())

        k = np.empty([nk, 3], np.float64)
        w = np.empty([nk], np.float64)
        for ik in range(nk):
            l = self.readline().split()
            k[ik, :] = float(l[1]), float(l[2]), float(l[3])
            w[ik] = float(l[4])

        # Correct units to 1/Ang
        k /= Bohr2Ang

        if sc is None:
            return k, w
        return np.dot(k, sc.cell.T / (2 * np.pi)), w

    @sile_fh_open()
    def write_data(self, k, weight, fmt='.9e'):
        """ Writes K-points to file

        Parameters
        ----------
        k : array_like
           k-points in units 1/Bohr
        weight : array_like
           same length as k, weights of k-points
        fmt : str, optional
           format for the k-values
        """
        sile_raise_write(self)

        nk = len(k)
        self._write('{}\n'.format(nk))
        _fmt = ('{:d}' + (' {:' + fmt + '}') * 4) + '\n'

        for i, (kk, w) in enumerate(zip(np.atleast_2d(k), weight)):
            self._write(_fmt.format(i + 1, kk[0], kk[1], kk[2], w))

    @sile_fh_open()
    def read_brillouinzone(self, sc):
        """ Returns K-points from the file (note that these are in reciprocal units)

        Parameters
        ----------
        sc : SuperCellChild
           required supercell for the BrillouinZone object

        Returns
        -------
        bz : BrillouinZone
        """
        k, w = self.read_data(sc)
        from sisl.physics.brillouinzone import BrillouinZone

        bz = BrillouinZone(sc)
        bz._k = k
        bz._w = w
        return bz

    @sile_fh_open()
    def write_brillouinzone(self, bz, fmt='.9e'):
        """ Writes BrillouinZone-points to file

        Parameters
        ----------
        bz : BrillouinZone
           object contain all weights and k-points
        fmt : str, optional
           format for the k-values
        """
        # And convert to 1/Bohr
        k = bz.tocartesian(bz.k) * Bohr2Ang
        self.write_data(k, bz.weight, fmt)


class rkpSileSiesta(kpSileSiesta):
    """ Special k-point file with units in reciprocal lattice vectors

    Its main usage is as input for the kgrid.File fdf-option, in which case this
    file provides the k-points in the correct format.
    """

    @sile_fh_open()
    def read_data(self):
        """ Returns K-points from the file (note that these are in reciprocal units)

        Returns
        -------
        k : k-points, in units of the reciprocal lattice vectors
        w : weights for k-points
        """
        nk = int(self.readline())

        k = np.empty([nk, 3], np.float64)
        w = np.empty([nk], np.float64)
        for ik in range(nk):
            l = self.readline().split()
            k[ik, :] = float(l[1]), float(l[2]), float(l[3])
            w[ik] = float(l[4])

        return k, w

    @sile_fh_open()
    def read_brillouinzone(self, sc):
        """ Returns K-points from the file

        Parameters
        ----------
        sc : SuperCellChild
           required supercell for the BrillouinZone object

        Returns
        -------
        bz : BrillouinZone
        """
        k, w = self.read_data()
        from sisl.physics.brillouinzone import BrillouinZone

        bz = BrillouinZone(sc)
        bz._k = k
        bz._w = w
        return bz

    @sile_fh_open()
    def write_brillouinzone(self, bz, fmt='.9e'):
        """ Writes BrillouinZone-points to file

        Parameters
        ----------
        bz : BrillouinZone
           object contain all weights and k-points
        fmt : str, optional
           format for the k-values
        """
        self.write_data(bz.k, bz.weight, fmt)


add_sile('KP', kpSileSiesta, case=False, gzip=True)
add_sile('RKP', rkpSileSiesta, case=False, gzip=True)
