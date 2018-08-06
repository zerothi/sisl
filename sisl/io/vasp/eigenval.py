from __future__ import print_function

import numpy as np

# Import sile objects
from .sile import SileVASP
from ..sile import *


__all__ = ['eigenvalSileVASP']


class eigenvalSileVASP(SileVASP):
    """ Kohn-Sham eigenvalues """

    @sile_fh_open()
    def read_data(self):
        r""" Read eigenvalues, as calculated and written by VASP

        Returns
        -------
        numpy.ndarray : all eigenvalues, shape ``(ns, nk, nb)``
                        where ``ns`` number of spin-components, ``nk`` number of k-points and
                        ``nb`` number of bands
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
        eigs = np.empty([ns, nk, nb], np.float32)
        for ik in range(nk):
            self.readline()  # empty line
            self.readline()  # ... k-point, weight
            for ib in range(nb):
                # band, eig_UP, eig_DOWN, pop_UP, pop_DOWN
                # We currently neglect the populations
                E = map(float, self.readline().split()[1:ns+1])
                eigs[:, ik, ib] = list(E)
        return eigs


add_sile('EIGENVAL', eigenvalSileVASP, gzip=True)
