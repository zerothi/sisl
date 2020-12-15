import numpy as np

# Import sile objects
from .sile import SileVASP
from ..sile import add_sile, sile_fh_open

from sisl._internal import set_module


__all__ = ['outSileVASP']


@set_module("sisl.io.vasp")
class outSileVASP(SileVASP):
    """ Output file from VASP """
    _job_completed = False

    @property
    def job_completed(self):
        """ True if the line "General timing and accounting" was found. """
        return self._job_completed

    @sile_fh_open()
    def cpu_time(self, flag="General timing and accounting"):
        """ Returns the consumed cpu time (in seconds) from a given section """
        if flag == "General timing and accounting":
            nskip, iplace = 3, 5
        else:
            raise ValueError(f"{self.__class__.__name__}.cpu_time unknown flag '{flag}'")

        found = self.step_to(flag, reread=False)[0]
        if found:
            self._job_completed = True
            for _ in range(nskip):
                line = self.readline()
            return float(line.split()[iplace])
        raise KeyError(f"{self.__class__.__name__}.cpu_time could not find flag '{flag}' in file")

    @sile_fh_open()
    def accuracy_reached(self):
        """ True if the line "reached required accuracy" was found. """
        return self.step_to("reached required accuracy")[0]

    @sile_fh_open()
    def read_energy(self, all=False):
        """ Reads the energy specification from the  and energy(sigma->0) in units of eV

        Parameters
        ----------
        all: bool, optional
            return a list of energies from each step

        Returns
        -------
        dict : all energies from the "Free energy of the ion-electron system" segment of VASP output
        """
        name_conv = {
            "alpha": "Z",
            "Ewald": "Ewald",
            "-Hartree": "Ehartree",
            "-exchange": "ExcHF",
            "-V(xc)+E(xc)": "Exc",
            "PAW": "Epaw",
            "entropy": "Ets",
            "eigenvalues": "Ebs",
            "atomic": "Eion",
            "Solvation": "Esolv",
        }

        def readE(itt):
            nonlocal name_conv
            # read the energy tables
            f = self.step_to("Free energy of the ion-electron system", reread=False)[0]
            if not f:
                return None
            next(itt) # -----
            line = next(itt)
            E = {}
            while "----" not in line:
                key, *v = line.split()
                if key == "PAW":
                    E[f"{name_conv[key]}1"] = float(v[-2])
                    E[f"{name_conv[key]}2"] = float(v[-1])
                else:
                    E[name_conv[key]] = float(v[-1])
                line = next(itt)
            line = next(itt)
            E["Efree"] = float(line.split()[-2])
            line = next(itt)
            line = next(itt)
            v = line.split()
            E["Etot"] = float(v[4])
            E["Esigma0"] = float(v[-1])
            return E

        itt = iter(self)
        E = []
        e = readE(itt)
        while e is not None:
            E.append(e)
            e = readE(itt)
        try:
            # this just puts the job_completed flag. But otherwise not used
            self.cpu_time()
        except:
            pass

        if all:
            return E
        if len(E) > 0:
            return E[-1]
        return None

add_sile('OUTCAR', outSileVASP, gzip=True)
