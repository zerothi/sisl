# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np

from sisl._internal import set_module
from sisl.geometry import Geometry
from sisl.messages import deprecation
from sisl.unit import units
from sisl.utils import PropertyDict

from .._multiple import SileBinder
from ..sile import add_sile, sile_fh_open
from .sile import SileORCA

__all__ = ["txtSileORCA"]

_A = SileORCA.InfoAttr


@set_module("sisl.io.orca")
class txtSileORCA(SileORCA):
    """Output from the ORCA property.txt file"""

    _info_attributes_ = [
        _A(
            "na",
            r".*Number of atoms:",
            lambda attr, match: int(match.string.split()[-1]),
        ),
        _A(
            "no",
            r".*Number of basis functions:",
            lambda attr, match: int(match.string.split()[-1]),
        ),
        _A(
            "vdw_correction",
            r".*\$ VdW_Correction",
            lambda attr, match: True,
            default=False,
        ),
    ]

    @property
    @deprecation(
        "txtSileORCA.na is deprecated in favor of txtSileORCA.info.na", "0.16.0"
    )
    def na(self):
        """Number of atoms"""
        return self.info.na

    @property
    @deprecation(
        "txtSileORCA.no is deprecated in favor of txtSileORCA.info.no", "0.16.0"
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
        f = self.step_to("Number of Alpha Electrons", allow_reread=False)
        if f[0]:
            alpha = float(f[1].split()[-1])
            beta = float(self.readline().split()[-1])
        else:
            return None
        return alpha, beta

    @SileBinder()
    @sile_fh_open()
    def read_energy(self):
        """Reads the energy blocks

        Returns
        -------
        PropertyDict or list of PropertyDict : all data (in eV) from the "DFT_Energy" and "VdW_Correction" blocks
        """
        # read the DFT_Energy block
        f = self.step_to("$ DFT_Energy", allow_reread=False)[0]
        if not f:
            return None
        self.readline()  # description
        self.readline()  # geom. index
        self.readline()  # prop. index

        Ha2eV = units("Ha", "eV")
        E = PropertyDict()

        line = self.readline()
        while "----" not in line:
            v = line.split()
            value = float(v[-1]) * Ha2eV
            if v[0] == "Exchange":
                E["exchange"] = value
            elif v[0] == "Correlation":
                if v[2] == "NL":
                    E["correlation_nl"] = value
                else:
                    E["correlation"] = value
            elif v[0] == "Exchange-Correlation":
                E["xc"] = value
            elif v[0] == "Embedding":
                E["embedding"] = value
            elif v[1] == "DFT":
                E["total"] = value
            line = self.readline()

        line = self.readline()
        if self.info.vdw_correction:
            v = self.step_to("Van der Waals Correction:")[1].split()
            E["vdw"] = float(v[-1]) * Ha2eV

        return E

    @SileBinder()
    @sile_fh_open()
    def read_geometry(self):
        """Reads the geometry from ORCA property.txt file

        Returns
        -------
        geometries: Geometry or list of Geometry
        """
        # Read the Geometry block
        f = self.step_to("!GEOMETRY!", allow_reread=False)[0]
        if not f:
            return None

        line = self.readline()
        na = int(line.split()[-1])
        self.readline()  # skip Geometry index
        self.readline()  # skip Coordinates line

        atoms = []
        xyz = np.empty([na, 3], np.float64)
        for ia in range(na):
            l = self.readline().split()
            atoms.append(l[1])
            xyz[ia] = l[2:5]

        return Geometry(xyz, atoms)


add_sile("txt", txtSileORCA, gzip=True)
