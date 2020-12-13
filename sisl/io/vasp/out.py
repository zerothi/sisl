import numpy as np

# Import sile objects
from .sile import SileVASP
from ..sile import add_sile, sile_fh_open

from sisl._internal import set_module


__all__ = ['outSileVASP']


@set_module("sisl.io.vasp")
class outSileVASP(SileVASP):
    """ OUTCAR VASP file """

    _file_read = False
    _job_completed = False
    _cpu_time = None
    _accuracy_reached = False
    _energy = []

    def _read(self):
        itt = iter(self)
        for line in itt:
            # job completion
            if 'General timing and accounting' in line:
                self._job_completed = True
                line = next(itt)
                line = next(itt)
                line = next(itt)
                self._cpu_time = float(line.split()[5])
            # relaxation finished
            if 'reached required accuracy' in line:
                self._accuracy_reached = True
            # energy output
            if 'free  energy   TOTEN  =' in line:
                free_energy = float(line.split()[4])
                line = next(itt) # read blank line
                line = next(itt) # energy(sigma->0) line
                esigma0 = float(line.split()[6])
                self._energy.append([free_energy, esigma0])
        self._file_read = True

    @property
    def job_completed(self):
        """ True if the line "General timing and accounting" was found. """
        return self._job_completed

    @property
    def cpu_time(self):
        """ Returns the consumed cpu time (in seconds). """
        return self._cpu_time

    @property
    def accuracy_reached(self):
        """ True if the line "reached required accuracy" was found. """
        return self._accuracy_reached

    @sile_fh_open()
    def read_energy(self, all=False):
        """ Reads the free energy and energy(sigma->0) in units of eV

        Parameters
        ----------
        all: bool, optional                                                                                                                                                                                                  return a list of energies from each step

        Returns
        -------
        list or numpy.ndarray
        """
        if not self._file_read:
            self._read()
        if all:
            return np.array(self._energy)
        else:
            return self._energy[-1]

add_sile('OUTCAR', outSileVASP, gzip=True)
