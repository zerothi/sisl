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

    def _setup(self, *args, **kwargs):
        """ Ensure the class has essential tags """
        super()._setup(*args, **kwargs)
        self._completed = None
        self._na = None
        self._no = None

    def readline(self, *args, **kwargs):
        line = super().readline(*args, **kwargs)
        if self._completed is None and "ORCA TERMINATED NORMALLY" in line:
            self._completed = True
        elif self._na is None and "Number of atoms" in line:
            v = line.split()
            self._na = int(v[-1])
        elif self._no is None and "Number of basis functions" in line:
            v = line.split()
            self._no = int(v[-1])
        return line

    readline.__doc__ = SileORCA.readline.__doc__

    @sile_fh_open()
    def completed(self):
        """ True if the full file has been read and "ORCA TERMINATED NORMALLY" was found. """
        if self._completed is None:
            completed = self.step_to("ORCA TERMINATED NORMALLY")[0]
        else:
            completed = self._completed
        if completed:
            self._completed = True
        return completed

    @property
    @sile_fh_open()
    def na(self):
        """ Number of atoms """
        if self._na is None:
            v = self.step_to("Number of atoms")[1].split()
            self._na = int(v[-1])
        return self._na

    @property
    @sile_fh_open()
    def no(self):
        """ Number of orbitals (basis functions) """
        if self._no is None:
            v = self.step_to("Number of basis functions")[1].split()
            self._no = int(v[-1])
        return self._no

    def _read_atomic_block(self):
        itt = iter(self)
        next(itt) # skip ---
        A = np.empty((self.na, 2), np.float64)
        for ia in range(self.na):
            line = next(itt)
            v = line.split()
            A[ia] = float(v[-2]), float(v[-1])
        return A

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
    def read_charge(self, name='mulliken', projection='orbital', orbital=None):
        """ Reads from charge and spin population analysis

        Parameters
        ----------
        name : {'mulliken', 'loewdin'}
            name of the charge scheme to be read
        projection : {'orbital', 'atom'}
            whether to get orbital- or atom-resolved quantities
        orbital : str, optional
            allows to extract the atom-resolved orbital values matching this keyword

        Returns
        -------
        PropertyDicts or ndarray : atom/orbital-resolved charge and spin data
        """

        if projection.lower() == 'atom':
            if name.lower() == 'mulliken':
                f = self.step_to("MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS")[0]
            elif name.lower() in ['loewdin', 'lowdin', 'löwdin']:
                f = self.step_to("LOEWDIN ATOMIC CHARGES AND SPIN POPULATIONS")[0]
            if not f:
                return None
            return self._read_atomic_block()

        elif projection.lower() == 'orbital':
            if name.lower() == 'mulliken':
                f = self.step_to("MULLIKEN REDUCED ORBITAL CHARGES AND SPIN POPULATIONS", reread=False)[0]
            elif name.lower() in ['loewdin', 'lowdin', 'löwdin']:
                f = self.step_to("LOEWDIN REDUCED ORBITAL CHARGES AND SPIN POPULATIONS", reread=False)[0]
            if not f:
                return None

            self.step_to("CHARGE", reread=False)
            charge = self._read_orbital_block()

            self.step_to("SPIN", reread=False)
            spin = self._read_orbital_block()

            if orbital is None:
                return charge, spin

            else:
                cs = np.zeros((self.na, 2), np.float64)
                for key in charge:
                    ia, orb = key
                    if orb == orbital:
                        cs[ia, 0] = charge[key]
                        cs[ia, 1] = spin[key]
                return cs


add_sile('output', outputSileORCA, gzip=True)
