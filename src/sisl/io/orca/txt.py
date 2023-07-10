# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np

from sisl._internal import set_module
from sisl.geometry import Geometry
from sisl.unit import units
from sisl.utils import PropertyDict

from .._multiple import SileBinder
from ..sile import add_sile, sile_fh_open
from .sile import SileORCA

__all__ = ['txtSileORCA']


@set_module("sisl.io.orca")
class txtSileORCA(SileORCA):
    """ Output from the ORCA property.txt file """

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
            f = self.step_to("Number of atoms"[1:])
            if f[0]:
                self._na = int(f[1].split()[-1])
            else:
                return None
        return self._na

    @property
    @sile_fh_open()
    def no(self):
        """ Number of orbitals (basis functions) """
        if self._no is None:
            f = self.step_to("Number of basis functions"[1:])
            if f[0]:
                self._no = int(f[1].split()[-1])
            else:
                return None
        return self._no

    @SileBinder(postprocess=np.array)
    @sile_fh_open()
    def read_electrons(self):
        """ Read number of electrons (alpha, beta)

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
        """ Reads the energy blocks

        Returns
        -------
        PropertyDict or list of PropertyDict : all data (in eV) from the "DFT_Energy" and "VdW_Correction" blocks
        """
        # read the DFT_Energy block
        f = self.step_to("$ DFT_Energy", allow_reread=False)[0]
        if not f:
            return None
        self.readline() # description
        self.readline() # geom. index
        self.readline() # prop. index

        Ha2eV = units('Ha', 'eV')
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
        if "$ VdW_Correction" in line:
            v = self.step_to("Van der Waals Correction:")[1].split()
            E["vdw"] = float(v[-1]) * Ha2eV

        return E

    @SileBinder()
    @sile_fh_open()
    def read_geometry(self):
        """ Reads the geometry from ORCA property.txt file

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
        self.readline() # skip Geometry index
        self.readline() # skip Coordinates line

        atoms = []
        xyz = np.empty([na, 3], np.float64)
        for ia in range(na):
            l = self.readline().split()
            atoms.append(l[1])
            xyz[ia] = l[2:5]

        return Geometry(xyz, atoms)


add_sile('txt', txtSileORCA, gzip=True)
