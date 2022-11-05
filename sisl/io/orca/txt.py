# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
from .sile import SileORCA
from ..sile import add_sile, sile_fh_open

from sisl.utils import PropertyDict
from sisl._internal import set_module
from sisl import Geometry

__all__ = ['txtSileORCA']


@set_module("sisl.io.orca")
class txtSileORCA(SileORCA):
    """ Output property txt file from ORCA """

    def _setup(self, *args, **kwargs):
        """ Ensure the class has essential tags """
        super()._setup(*args, **kwargs)
        self._na = None
        self._no = None

    def readline(self, *args, **kwargs):
        line = super().readline(*args, **kwargs)
        if self._na is None and "Number of atoms:"[1:] in line:
            v = line.split()
            self._na = int(v[-1])
        elif self._no is None and "Number of basis functions:"[1:] in line:
            v = line.split()
            self._no = int(v[-1])
        return line

    @property
    @sile_fh_open()
    def na(self):
        """ Number of atoms """
        if self._na is None:
            v = self.step_to("Number of atoms"[1:])[1].split()
            self._na = int(v[-1])
        return self._na

    @property
    @sile_fh_open()
    def no(self):
        """ Number of orbitals (basis functions) """
        if self._no is None:
            v = self.step_to("Number of basis functions"[1:])[1].split()
            self._no = int(v[-1])
        return self._no

    @sile_fh_open()
    def read_energy(self, all=False):
        """ Reads the energy specification from ORCA property txt file and returns 
        energy dictionary in units of eV (and related info from the block)

        Parameters
        ----------
        all: bool, optional
            return a list of dictionaries from each step

        Returns
        -------
        PropertyDict : all data from the "DFT_Energy" segment of ORCA property output
        """

        def readE(itt):
            # read the DFT_Energy block
            f = self.step_to("$ DFT_Energy", reread=False)[0]
            if not f:
                return None
            next(itt) # description
            next(itt) # geom. index
            next(itt) # prop. index
            line = next(itt)
            E = PropertyDict()
            while "----" not in line:
                v = line.split()
                value = float(v[-1])
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
                line = next(itt)
            line = next(itt)
            if "$ VdW_Correction" in line:
                v = self.step_to("Van der Waals Correction:", reread=False)[1].split()
                E["vdw"] = float(v[-1])
                E["total_vdw"] = E["total"] + float(v[-1])

            return E

        itt = iter(self)
        E = []
        e = readE(itt)
        while e is not None:
            E.append(e)
            e = readE(itt)

        if all:
            return E
        if len(E) > 0:
            return E[-1]
        return None

    @sile_fh_open()
    def read_geometry(self, all=False):
        """ Reads the geometry from ORCA property txt file

        Parameters
        ----------
        all: bool, optional
            return a list of all geometries instead of the last one

        Returns
        -------
        geometries: list or Geometry or None
            if all is False only one geometry will be returned (or None). Otherwise
            a list of geometries corresponding to each step.
        """
        def readG(itt):
            # Read the Geometry block
            f = self.step_to("!GEOMETRY!", reread=False)[0]
            if not f:
                return None
            line = next(itt)
            na = int(line.split()[-1])
            next(itt) # skip Geometry index
            next(itt) # skip Coordinates line
            atoms = []
            xyz = np.empty([na, 3], np.float64)
            for ia in range(na):
                line = next(itt)
                l = line.split()
                atoms.append(l[1])
                xyz[ia] = l[2:5]
            return Geometry(xyz, atoms)

        itt = iter(self)
        G = []
        g = readG(itt)
        while g is not None:
            G.append(g)
            g = readG(itt)

        if all:
            return G
        if len(G) > 0:
            return G[-1]
        return None

add_sile('txt', txtSileORCA, gzip=True)
