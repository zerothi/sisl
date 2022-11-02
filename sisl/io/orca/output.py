# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
from functools import lru_cache
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
    def completed(self):
        """ True if the full file has been read and "ORCA TERMINATED NORMALLY" was found. """
        return self.step_to("ORCA TERMINATED NORMALLY")[0]


    @property
    @lru_cache(1)
    def _natoms(self):
        f, line = self.step_to("Number of atoms")
        v = line.split()
        return int(v[-1])


    def _read_atomic_block(self, natoms):
        itt = iter(self)
        next(itt) # skip ---
        A = np.empty((natoms, 2), np.float64)
        for ia in range(natoms):
            line = next(itt)
            v = line.split()
            A[ia] = float(v[-2]), float(v[-1])
        return A


    @sile_fh_open()
    def read_mulliken_atomic(self):
        """ Reads the atom-resolved Mulliken charge and spin population analysis

        Returns
        -------
        ndarray : atom-resolved charge and spin populations
        """

        natoms = self._natoms

        f = self.step_to("MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS")[0]
        if not f:
            return None

        return self._read_atomic_block(natoms)


    @sile_fh_open()
    def read_loewdin_atomic(self):
        """ Reads the atom-resolved Loewdin charge and spin population analysis

        Returns
        -------
        ndarray : atom-resolved charge and spin populations
        """

        natoms = self._natoms

        f = self.step_to("LOEWDIN ATOMIC CHARGES AND SPIN POPULATIONS")[0]
        if not f:
            return None

        return self._read_atomic_block(natoms)


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
    def read_mulliken_orbitals(self, orbital=None):
        """ Reads the orbital-resolved Mulliken charge and spin population analysis

        Parameters
        ----------
        orbital : str, optional
            if an orbital string is specified a ndarray is returned with the corresponding
            values per atom, otherwise the whole orbital block is returned as dictionary

        Returns
        -------
        ndarray or PropertyDicts : charge and spin data
        """

        f = self.step_to("MULLIKEN REDUCED ORBITAL CHARGES AND SPIN POPULATIONS", reread=False)[0]
        if not f:
            return None

        self.step_to("CHARGE", reread=False)
        charge = self._read_orbital_block()

        self.step_to("SPIN", reread=False)
        spin = self._read_orbital_block()

        if orbital is not None:
            natoms = self._natoms
            cs = np.zeros((natoms, 2), np.float64)
            for key in charge:
                ia, orb = key
                if orb == orbital:
                    cs[ia, 0] = charge[key]
                    cs[ia, 1] = spin[key]
            return cs

        return charge, spin


    @sile_fh_open()
    def read_loewdin_orbitals(self, orbital=None):
        """ Reads the orbital-resolved Loewdin charge and spin population analysis

        Parameters
        ----------
        orbital : str, optional
            if an orbital string is specified a ndarray is returned with the corresponding
            values per atom, otherwise the whole orbital block is returned as dictionary

        Returns
        -------
        ndarray or PropertyDicts : charge and spin data
        """

        f = self.step_to("LOEWDIN REDUCED ORBITAL CHARGES AND SPIN POPULATIONS", reread=False)[0]
        if not f:
            return None

        self.step_to("CHARGE", reread=False)
        charge = self._read_orbital_block()

        self.step_to("SPIN", reread=False)
        spin = self._read_orbital_block()

        if orbital is not None:
            natoms = self._natoms
            cs = np.zeros((natoms, 2), np.float64)
            for key in charge:
                ia, orb = key
                if orb == orbital:
                    cs[ia, 0] = charge[key]
                    cs[ia, 1] = spin[key]
            return cs

        return charge, spin


add_sile('output', outputSileORCA, gzip=True)
