from __future__ import print_function

import numpy as np

# Import sile objects
from .sile import SileVASP
from ..sile import *


__all__ = ['doscarSileVASP']


class doscarSileVASP(SileVASP):
    """ Density of states output """

    @sile_fh_open()
    def read_data(self):
        r""" Read DOS, as calculated and written by VASP

        Returns
        -------
        E : numpy.ndarray
            energy points (in eV)
        DOS : numpy.ndarray
            DOS points (in 1/eV)
        """
        # read first line
        line = self.readline()  # NIONS, NIONS, JOBPAR_, WDES%INCDIJ
        line = self.readline()  # AOMEGA, LATT_CUR%ANORM(1:3) *1e-10, POTIM * 1e-15
        line = self.readline()  # TEMP
        line = self.readline()  # ' CAR '
        line = self.readline()  # name
        line = self.readline().split()
        Emax = float(line[0])
        Emin = float(line[1])
        NE = int(line[2])
        Ef = float(line[3])

        E = np.empty([NE], np.float32)
        # Determine output
        line = list(map(float, self.readline().split()))
        ns = (len(line) - 1) // 2
        DOS = np.empty([ns, NE], np.float32)
        E[0] = line[0]
        DOS[:, 0] = line[1:ns+1]
        for ie in range(1, NE):
            line = list(map(float, self.readline().split()))
            E[ie] = line[0]
            DOS[:, ie] = line[1:ns+1]
        return E - Ef, DOS


add_sile('DOSCAR', doscarSileVASP, gzip=True)
