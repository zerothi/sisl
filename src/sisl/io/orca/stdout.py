# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np

from sisl._internal import set_module
from sisl.messages import deprecation
from sisl.unit import serialize_units_arg, unit_convert
from sisl.utils import PropertyDict

from .._multiple import SileBinder
from ..sile import add_sile, sile_fh_open
from .sile import SileORCA

__all__ = ["outputSileORCA", "stdoutSileORCA"]


_A = SileORCA.InfoAttr


@set_module("sisl.io.orca")
class stdoutSileORCA(SileORCA):
    """Output file from ORCA"""

    _info_attributes_ = [
        _A(
            "na",
            r".*Number of atoms",
            lambda attr, match: int(match.string.split()[-1]),
            not_found="error",
        ),
        _A(
            "no",
            r".*Number of basis functions",
            lambda attr, match: int(match.string.split()[-1]),
            not_found="error",
        ),
        _A(
            "vdw_correction",
            r".*DFT DISPERSION CORRECTION",
            lambda attr, match: True,
            default=False,
            not_found="ignore",
        ),
        _A(
            "completed",
            r".*ORCA TERMINATED NORMALLY",
            lambda attr, match: True,
            default=False,
            not_found="warn",
        ),
    ]

    def completed(self):
        """True if the full file has been read and "ORCA TERMINATED NORMALLY" was found."""
        return self.info.completed

    @property
    @deprecation(
        "stdoutSileORCA.na is deprecated in favor of stdoutSileORCA.info.na", "0.16"
    )
    def na(self):
        """Number of atoms"""
        return self.info.na

    @property
    @deprecation(
        "stdoutSileORCA.no is deprecated in favor of stdoutSileORCA.info.no", "0.16"
    )
    def no(self):
        """Number of orbitals (basis functions)"""
        return self.info.no

    @SileBinder(postprocess=np.array)
    @sile_fh_open()
    def read_electrons(self):
        """Read number of electrons (alpha, beta)

        Returns
        -------
        ndarray or list of ndarrays : alpha and beta electrons
        """
        f = self.step_to("N(Alpha)", allow_reread=False)
        if f[0]:
            alpha = float(f[1].split()[-2])
            beta = float(self.readline().split()[-2])
            return alpha, beta

        return None

    @SileBinder()
    @sile_fh_open()
    def read_charge(
        self,
        name="mulliken",
        projection="orbital",
        orbitals=None,
        reduced=True,
        spin=False,
    ):
        """Reads from charge (or spin) population analysis

        Parameters
        ----------
        name : {'mulliken', 'loewdin'}
            name of the charge scheme to be read
        projection : {'orbital', 'atom'}
            whether to get orbital- or atom-resolved quantities
        orbitals : str, optional
            allows to extract the atom-resolved orbitals matching this keyword
        reduced : bool, optional
            whether to search for full or reduced orbital projections
        spin : bool, optional
            whether to return the spin block instead of charge

        Returns
        -------
        PropertyDicts or ndarray or list thereof: atom/orbital-resolved charge (or spin) data
        """
        if name.lower() in ("mulliken", "m"):
            name = "mulliken"
        elif name.lower() in ("loewdin", "lowdin", "löwdin", "l"):
            name = "loewdin"
        else:
            raise NotImplementedError(f"name={name} is not implemented")

        if projection.lower() in ("atom", "atoms", "a"):
            projection = "atom"
        elif projection.lower() in ("orbital", "orbitals", "orb", "o"):
            projection = "orbital"
        else:
            raise ValueError(f"Projection must be atom or orbital")

        if projection == "atom":
            if name == "mulliken":
                step_to = "MULLIKEN ATOMIC CHARGES"
            elif name == "loewdin":
                step_to = "LOEWDIN ATOMIC CHARGES"

            def read_block(step_to):
                f, line = self.step_to(step_to, allow_reread=False)
                if not f:
                    return None
                if "SPIN" in line:
                    spin_block = True
                else:
                    spin_block = False

                self.readline()  # skip ---
                A = np.empty(self.info.na, np.float64)
                for ia in range(self.info.na):
                    line = self.readline()
                    v = line.split()
                    if spin_block and not spin:
                        A[ia] = v[-2]
                    elif not spin_block and spin:
                        return None
                    else:
                        A[ia] = v[-1]
                return A

        elif projection == "orbital" and reduced:
            if name == "mulliken":
                step_to = "MULLIKEN REDUCED ORBITAL CHARGES"
            elif name == "loewdin":
                step_to = "LOEWDIN REDUCED ORBITAL CHARGES"

            def read_reduced_orbital_block():
                D = PropertyDict()
                v = self.readline().split()
                while len(v) > 0:
                    if len(v) == 8:
                        ia = int(v[0])
                        D[(ia, v[2])] = float(v[4])
                        D[(ia, v[5])] = float(v[7])
                    elif len(v) == 6:
                        D[(ia, v[0])] = float(v[2])
                        D[(ia, v[3])] = float(v[5])
                    else:
                        D[(ia, v[0])] = float(v[2])
                    v = self.readline().split()
                return D

            def read_block(step_to):
                f, line = self.step_to(step_to, allow_reread=False)
                if not f:
                    return None

                if "SPIN" in line:
                    spin_block = True
                else:
                    spin_block = False

                if spin_block and spin:
                    self.step_to("SPIN", allow_reread=False)
                elif spin_block:
                    self.step_to("CHARGE", allow_reread=False)
                elif not spin:
                    self.readline()  # skip ---
                else:
                    return None

                D = read_reduced_orbital_block()
                if orbitals is None:
                    return D
                else:
                    Da = np.zeros(self.info.na, np.float64)
                    for (ia, orb), d in D.items():
                        if orb == orbitals:
                            Da[ia] = d
                    return Da

        elif projection == "orbital" and not reduced:
            if name == "mulliken":
                step_to = "MULLIKEN ORBITAL CHARGES"
            elif name == "loewdin":
                step_to = "LOEWDIN ORBITAL CHARGES"

            def read_block(step_to):
                f, line = self.step_to(step_to, allow_reread=False)
                if "SPIN" in line:
                    spin_block = True
                else:
                    spin_block = False

                if not f:
                    return None

                self.readline()  # skip ---
                if "MULLIKEN" in step_to:
                    self.readline()  # skip line "The uncorrected..."

                Do = np.empty(self.info.no, np.float64)  # orbital-resolved
                Da = np.zeros(self.info.na, np.float64)  # atom-resolved
                for io in range(self.info.no):
                    v = self.readline().split()  # io, ia+element, orb, chg, (spin)

                    # split atom number and element from v[1]
                    ia, element = "", ""
                    for s in v[1]:
                        if s.isdigit():
                            ia += s
                        else:
                            element += s
                    ia = int(ia)
                    if spin_block and spin:
                        Do[io] = v[4]
                    elif not spin_block and spin:
                        return None
                    else:
                        Do[io] = v[3]
                    if v[2] == orbitals:
                        Da[ia] += Do[io]
                if orbitals is None:
                    return Do
                else:
                    return Da

        return read_block(step_to)

    @SileBinder()
    @sile_fh_open()
    def read_energy(self, units="eV"):
        """Reads the energy blocks

        Parameters
        ----------
        units : {str, dict, list, tuple}
            selects units in the returned data

        Note
        ----
        Energies written by ORCA have units of Ha.

        Returns
        -------
        PropertyDict or list of PropertyDict : all energy data from the "TOTAL SCF ENERGY" and "DFT DISPERSION CORRECTION" blocks
        """
        f = self.step_to("TOTAL SCF ENERGY", allow_reread=False)[0]
        if not f:
            return None

        self.readline()  # skip ---
        self.readline()  # skip blank line

        units = serialize_units_arg(units)
        Ha2unit = unit_convert("Ha", units["energy"])

        E = PropertyDict()

        line = self.readline()
        while "----" not in line:
            v = line.split()
            if "Total Energy" in line:
                E["total"] = float(v[-4]) * Ha2unit
            elif "E(X)" in line:
                E["exchange"] = float(v[-2]) * Ha2unit
            elif "E(C)" in line:
                E["correlation"] = float(v[-2]) * Ha2unit
            elif "E(XC)" in line:
                E["xc"] = float(v[-2]) * Ha2unit
            elif "DFET-embed. en." in line:
                E["embedding"] = float(v[-2]) * Ha2unit
            line = self.readline()

        if self.info.vdw_correction:
            self.step_to("DFT DISPERSION CORRECTION")
            v = self.step_to("Dispersion correction", allow_reread=False)[1].split()
            E["vdw"] = float(v[-1]) * Ha2unit

        return E

    @SileBinder()
    @sile_fh_open()
    def read_orbital_energies(self):
        """Reads the "ORBITAL ENERGIES" blocks

        Returns
        -------
        ndarray or list of ndarray : orbital energies (in eV) from the "ORBITAL ENERGIES" blocks
        """
        f = self.step_to("ORBITAL ENERGIES", allow_reread=False)[0]
        if not f:
            return None

        self.readline()  # skip ---
        if "SPIN UP ORBITALS" in self.readline():
            spin = True
            E = np.empty([self.info.no, 2], np.float64)
        else:
            spin = False
            E = np.empty([self.info.no, 1], np.float64)

        self.readline()  # Skip "NO OCC" header line

        v = self.readline().split()
        while len(v) > 0:
            i = int(v[0])
            E[i, 0] = v[-1]
            v = self.readline().split()

        if not spin:
            return E.ravel()

        self.readline()  # skip "SPIN DOWN ORBITALS"
        self.readline()  # Skip "NO OCC" header line
        v = self.readline().split()
        while len(v) > 0 and "---" not in v[0]:
            i = int(v[0])
            E[i, 1] = v[-1]
            v = self.readline().split()
        return E


outputSileORCA = deprecation(
    "outputSileORCA has been deprecated in favor of stdoutSileOrca.", "0.15"
)(stdoutSileORCA)

add_sile("output", stdoutSileORCA, gzip=True, case=False)
add_sile("orca.out", stdoutSileORCA, gzip=True, case=False)
add_sile("out", stdoutSileORCA, gzip=True, case=False)
