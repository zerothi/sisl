import numpy as np

from .sile import SileVASP
from ..sile import add_sile, sile_fh_open

from sisl._array import arrayf
from sisl._internal import set_module


__all__ = ['doscarSileVASP']


@set_module("sisl.io.vasp")
class doscarSileVASP(SileVASP):
    """ Density of states output """

    @sile_fh_open(True)
    def read_fermi_level(self):
        r""" Query the Fermi-level contained in the file

        Returns
        -------
        Ef : fermi-level of the system
        """
        self.readline()  # NIONS, NIONS, JOBPAR_, WDES%INCDIJ
        self.readline()  # AOMEGA, LATT_CUR%ANORM(1:3) *1e-10, POTIM * 1e-15
        self.readline()  # TEMP
        self.readline()  # ' CAR '
        self.readline()  # name
        line = self.readline().split()
        return float(line[3])

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
        self.readline()  # NIONS, NIONS, JOBPAR_, WDES%INCDIJ
        self.readline()  # AOMEGA, LATT_CUR%ANORM(1:3) *1e-10, POTIM * 1e-15
        self.readline()  # TEMP
        self.readline()  # ' CAR '
        self.readline()  # name
        line = self.readline().split()
        Emax = float(line[0])
        Emin = float(line[1])
        NE = int(line[2])
        Ef = float(line[3])

        E = np.empty([NE], np.float32)
        # Determine output
        line = arrayf(self.readline().split())
        ns = (len(line) - 1) // 2
        DOS = np.empty([ns, NE], np.float32)
        E[0] = line[0]
        DOS[:, 0] = line[1:ns+1]
        for ie in range(1, NE):
            line = arrayf(self.readline().split())
            E[ie] = line[0]
            DOS[:, ie] = line[1:ns+1]
        return E - Ef, DOS


add_sile('DOSCAR', doscarSileVASP, gzip=True)
