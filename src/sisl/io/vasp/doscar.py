# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np

from sisl._array import arrayf
from sisl._internal import set_module
from sisl.typing import UnitsVar
from sisl.unit import serialize_units_arg, unit_convert

from ..sile import add_sile, sile_fh_open
from .sile import SileVASP

__all__ = ["doscarSileVASP"]


@set_module("sisl.io.vasp")
class doscarSileVASP(SileVASP):
    """Density of states output"""

    @sile_fh_open(True)
    def read_fermi_level(self, units: UnitsVar = "eV") -> float:
        r"""Query the Fermi-level contained in the file

        Returns
        -------
        float
            fermi-level of the system
        """
        units = serialize_units_arg(units)
        eV2unit = unit_convert("eV", units["energy"])

        self.readline()  # NIONS, NIONS, JOBPAR_, WDES%INCDIJ
        self.readline()  # AOMEGA, LATT_CUR%ANORM(1:3) *1e-10, POTIM * 1e-15
        self.readline()  # TEMP
        self.readline()  # ' CAR '
        self.readline()  # name
        line = self.readline().split()

        return float(line[3]) * eV2unit

    @sile_fh_open()
    def read_data(self, units: UnitsVar = "eV"):
        r"""Read DOS, as calculated and written by VASP

        The energy points are shifted to the Fermi level.

        Parameters
        ----------
        units :
            selects units in the returned data

        Returns
        -------
        E : numpy.ndarray
            energy points (units as selected in argument)
        DOS : numpy.ndarray
            DOS points (units of 1/energy_units)
        """
        units = serialize_units_arg(units)
        eV2unit = unit_convert("eV", units["energy"])

        # read first line
        self.readline()  # NIONS, NIONS, JOBPAR_, WDES%INCDIJ
        self.readline()  # AOMEGA, LATT_CUR%ANORM(1:3) *1e-10, POTIM * 1e-15
        self.readline()  # TEMP
        self.readline()  # ' CAR '
        self.readline()  # name
        line = self.readline().split()
        # Emax = float(line[0])
        # Emin = float(line[1])
        NE = int(line[2])
        Ef = float(line[3])

        E = np.empty([NE], np.float32)
        # Determine output
        line = arrayf(self.readline().split())
        ns = (len(line) - 1) // 2
        DOS = np.empty([ns, NE], np.float32)
        E[0] = line[0]
        DOS[:, 0] = line[1 : ns + 1]
        for ie in range(1, NE):
            line = arrayf(self.readline().split())
            E[ie] = line[0]
            DOS[:, ie] = line[1 : ns + 1]

        return (E - Ef) * eV2unit, DOS / eV2unit


add_sile("DOSCAR", doscarSileVASP, gzip=True)
