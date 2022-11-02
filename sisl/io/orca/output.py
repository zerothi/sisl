# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
from .sile import SileORCA
from ..sile import add_sile, sile_fh_open

from sisl.utils import PropertyDict
from sisl._internal import set_module
from sisl import Geometry

__all__ = ['outputSileORCA']


@set_module("sisl.io.orca")
class outputSileORCA(SileORCA):
    """ Output file from ORCA """

    @sile_fh_open()
    def read_mulliken_atomic(self):
        """ Reads the atom-resolved Mulliken charge and spin population analysis

        Returns
        -------
        ndarray : atom-resolved charge and spin populations
        """

        f = self.step_to("MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS", reread=False)[0]
        if not f:
            return None

        itt = iter(self)

        next(itt) # ---
        M = []
        line = next(itt)
        while "Sum of atomic" not in line:
            v = line.split()
            ia = int(v[0])
            M.append([float(v[-2]), float(v[-1])])
            line = next(itt)

        return np.array(M)


    @sile_fh_open()
    def read_loewdin_atomic(self):
        """ Reads the atom-resolved Loewdin charge and spin population analysis

        Returns
        -------
        ndarray : atom-resolved charge and spin populations
        """

        f = self.step_to("LOEWDIN ATOMIC CHARGES AND SPIN POPULATIONS", reread=False)[0]
        if not f:
            return None

        itt = iter(self)

        next(itt) # ---
        L = []
        v = next(itt).split()
        while len(v) > 0:
            ia = int(v[0])
            L.append([float(v[-2]), float(v[-1])])
            v = next(itt).split()

        return np.array(L)


    def _read_orbital_block(self):
        itt = iter(self)
        D = PropertyDict()
        v = next(itt).split()
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
            v = next(itt).split()
        return D


    @sile_fh_open()
    def read_mulliken_orbitals(self):
        """ Reads the orbital-resolved Mulliken charge and spin population analysis

        Returns
        -------
        PropertyDicts : charge and spin dictionaries
        """

        f = self.step_to("MULLIKEN REDUCED ORBITAL CHARGES AND SPIN POPULATIONS", reread=False)[0]
        if not f:
            return None

        f = self.step_to("CHARGE", reread=False)[0]
        charge = self._read_orbital_block()

        f = self.step_to("SPIN", reread=False)[0]
        spin = self._read_orbital_block()

        return charge, spin


    @sile_fh_open()
    def read_loewdin_orbitals(self):
        """ Reads the orbital-resolved Loewdin charge and spin population analysis

        Returns
        -------
        PropertyDicts : charge and spin dictionaries
        """

        f = self.step_to("LOEWDIN REDUCED ORBITAL CHARGES AND SPIN POPULATIONS", reread=False)[0]
        if not f:
            return None

        f = self.step_to("CHARGE", reread=False)[0]
        charge = self._read_orbital_block()

        f = self.step_to("SPIN", reread=False)[0]
        spin = self._read_orbital_block()

        return charge, spin


add_sile('output', outputSileORCA, gzip=True)
