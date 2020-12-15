import numpy as np

# Import sile objects
from .sile import SileVASP
from ..sile import add_sile, sile_fh_open

from sisl._internal import set_module


__all__ = ['eigenvalSileVASP']


@set_module("sisl.io.vasp")
class eigenvalSileVASP(SileVASP):
    """ Kohn-Sham eigenvalues """

    @sile_fh_open()
    def read_data(self, k=False):
        r""" Read eigenvalues, as calculated and written by VASP

        Parameters
        ----------
        k : bool, optional
           also return k points and weights

        Returns
        -------
        numpy.ndarray : all eigenvalues, shape ``(ns, nk, nb)``
                        where ``ns`` number of spin-components, ``nk`` number of k-points and
                        ``nb`` number of bands
        numpy.ndarray : k-points (if `k` is true), shape ``(nk, 3)``
        numpy.ndarray : weights for k-points (if `k` is true), shape ``(nk)``

        """
        # read first line
        line = self.readline()  # NIONS, NIONS, NBLOCK * KBLOCK, NSPIN
        ns = int(line.split()[-1])
        line = self.readline()  # AOMEGA, LATT_CUR%ANORM(1:3) *1e-10, POTIM * 1e-15
        line = self.readline()  # TEMP
        line = self.readline()  # ' CAR '
        line = self.readline()  # name
        line = list(map(int, self.readline().split()))  # electrons, k-points, bands
        nk = line[1]
        nb = line[2]
        eigs = np.empty([ns, nk, nb], np.float64)
        kk = np.empty([nk, 3], np.float64)
        w = np.empty([nk], np.float64)
        for ik in range(nk):
            self.readline()  # empty line
            line = self.readline().split() # k-point, weight
            kk[ik, :] = list(map(float, line[:3]))
            w[ik] = float(line[3])
            for ib in range(nb):
                # band, eig_UP, eig_DOWN, pop_UP, pop_DOWN
                # We currently neglect the populations
                E = map(float, self.readline().split()[1:ns+1])
                eigs[:, ik, ib] = list(E)
        if k:
            return eigs, kk, w
        return eigs


add_sile('EIGENVAL', eigenvalSileVASP, gzip=True)
