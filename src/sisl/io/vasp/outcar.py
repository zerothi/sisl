# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np

from sisl._internal import set_module
from sisl.messages import deprecate_argument, deprecation
from sisl.typing import UnitsVar
from sisl.unit import serialize_units_arg, unit_convert
from sisl.utils import PropertyDict

from .._multiple import SileBinder
from ..sile import add_sile, sile_fh_open
from .sile import SileVASP

__all__ = ["outcarSileVASP"]


@set_module("sisl.io.vasp")
class outcarSileVASP(SileVASP):
    """OUTCAR file from VASP"""

    _info_attributes_ = [
        dict(
            name="completed",
            searcher=r".*General timing and accounting",
            parser=lambda attr, instance, match: lambda: True,
            default=lambda: False,
            not_found="warn",
        ),
        dict(
            name="accuracy_reached",
            searcher=r".*reached required accuracy",
            parser=lambda attr, instance, match: lambda: True,
            default=lambda: False,
            not_found="warn",
        ),
    ]

    @deprecation(
        "outcarSileVASP.completed is deprecated in favor of outcarSileVASP.info.completed",
        "0.15",
        "0.17",
    )
    def completed(self):
        """True if the line "General timing and accounting" was found."""
        return self.info.completed()

    @deprecation(
        "outcarSileVASP.accuracy_reached is deprecated in favor of outcarSileVASP.info.accuracy_reached",
        "0.15",
        "0.17",
    )
    def accuracy_reached(self):
        """True if the line "reached required accuracy" was found."""
        return self.info.accuracy_reached()

    @sile_fh_open()
    def cpu_time(self, flag="General timing and accounting"):
        """Returns the consumed cpu time (in seconds) from a given section"""
        if flag == "General timing and accounting":
            nskip, iplace = 3, 5
        else:
            raise ValueError(
                f"{self.__class__.__name__}.cpu_time unknown flag '{flag}'"
            )

        found = self.step_to(flag, allow_reread=False)[0]
        if found:
            for _ in range(nskip):
                line = self.readline()
            return float(line.split()[iplace])
        raise KeyError(
            f"{self.__class__.__name__}.cpu_time could not find flag '{flag}' in file"
        )

    @SileBinder()
    @sile_fh_open()
    @deprecation(
        "WARNING: direct calls to outcarSileVASP.read_energy() no longer returns the last entry! Now the next block on file is returned.",
        "0.14",
        "0.17",
    )
    def read_energy(self, units: UnitsVar = "eV") -> PropertyDict:
        """Reads an energy specification block from OUTCAR

        The function steps to the next occurrence of the "Free energy of the ion-electron system" segment

        Parameters
        ----------
        units :
            selects units in the returned data

        Notes
        -----
        The name convention in the dictionary is as follows:
            OUTCAR string           Key

            alpha Z        PSCENC = Z
            Ewald energy   TEWEN  = Ewald
            -Hartree energ DENC   = hartree
            -exchange      EXHF   = xcHF
            -V(xc)+E(xc)   XCENC  = xc
            PAW double counting   = paw1 paw2
            entropy T*S    EENTRO = entropy
            eigenvalues    EBANDS = band
            atomic energy  EATOM  = ion
            Solvation  Ediel_sol  = solvation

            free energy    TOTEN  = free
            energy without entropy= total
            energy(sigma->0)      = sigma0

        Returns
        -------
        out : PropertyDict
            all energies from a single "Free energy of the ion-electron system" segment
        """

        units = serialize_units_arg(units)
        eV2unit = unit_convert("eV", units["energy"])

        name_conv = {
            "alpha": "Z",
            "Ewald": "Ewald",
            "-Hartree": "hartree",
            "-exchange": "xcHF",
            "-V(xc)+E(xc)": "xc",
            "PAW": "paw",
            "entropy": "entropy",
            "eigenvalues": "band",
            "atomic": "ion",
            "Solvation": "solvation",
        }

        # read the energy tables
        f = self.step_to("Free energy of the ion-electron system", allow_reread=False)[
            0
        ]
        if not f:
            return None
        self.readline()  # -----
        line = self.readline()
        E = PropertyDict()
        while "----" not in line:
            key, *v = line.split()
            if key == "PAW":
                E[f"{name_conv[key]}1"] = float(v[-2]) * eV2unit
                E[f"{name_conv[key]}2"] = float(v[-1]) * eV2unit
            else:
                E[name_conv[key]] = float(v[-1]) * eV2unit
            line = self.readline()
        line = self.readline()
        E.free = float(line.split()[-2]) * eV2unit
        self.readline()
        line = self.readline()
        v = line.split()
        E.total = float(v[4]) * eV2unit
        E.sigma0 = float(v[-1]) * eV2unit
        return E

    @SileBinder()
    @sile_fh_open()
    def read_trajectory(self):
        """Reads cell+position+force data from OUTCAR for an ionic trajectory step

        The function steps to the block defined by the "VOLUME and BASIS-vectors are now :"
        line to first read the cell vectors, then it steps to the "TOTAL-FORCE (eV/Angst)" segment
        to read the atom positions and forces.

        Returns
        -------
        PropertyDict
            Trajectory step defined by cell vectors (`.cell`), atom positions (`.xyz`), and forces (`.force`)
        """

        f = self.step_to("VOLUME and BASIS-vectors are now :", allow_reread=False)[0]
        if not f:
            return None
        for i in range(4):
            self.readline()  # skip 4 lines
        C = []
        for i in range(3):
            line = self.readline()
            v = line.split()
            C.append(v[:3])  # direct lattice vectors
        # read a position-force table
        f = self.step_to("TOTAL-FORCE (eV/Angst)", allow_reread=False)[0]
        if not f:
            return None
        self.readline()  # -----
        P, F = [], []
        line = self.readline()
        while "----" not in line:
            v = line.split()
            # positions and forces
            P.append(v[:3])
            F.append(v[3:6])
            line = self.readline()
        step = PropertyDict()
        step.cell = np.array(C, dtype=np.float64)
        step.xyz = np.array(P, dtype=np.float64)
        step.force = np.array(F, dtype=np.float64)
        return step


outSileVASP = deprecation(
    "outSileVASP has been deprecated in favor of outcarSileVASP.", "0.15", "0.17"
)(outcarSileVASP)

add_sile("OUTCAR", outcarSileVASP, gzip=True)
