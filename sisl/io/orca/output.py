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
            f = self.step_to("Number of atoms")
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
            f = self.step_to("Number of atoms")
            if f[0]:
                self._no = int(f[1].split()[-1])
            else:
                return None
        return self._no

    @sile_fh_open()
    def read_charge(self, name='mulliken', projection='orbital', orbital=None,
                    reduced=True, all=False):
        """ Reads from charge and spin population analysis

        Parameters
        ----------
        name : {'mulliken', 'loewdin'}
            name of the charge scheme to be read
        projection : {'orbital', 'atom'}
            whether to get orbital- or atom-resolved quantities
        orbital : str, optional
            allows to extract the atom-resolved orbital values matching this keyword
        reduced : bool, optional
            whether to search for full or reduced orbital projections
        all: bool, optional
            return a list of all population analysis blocks instead of the last one

        Returns
        -------
        PropertyDicts or ndarray : atom/orbital-resolved charge and spin data
        """

        if name.lower() in ['mulliken', 'm']:
            name = 'mulliken'
        elif name.lower() in ['loewdin', 'lowdin', 'löwdin', 'l']:
            name = 'loewdin'
        else:
            raise NotImplementedError(f"name={name} is not implemented")

        if projection.lower() in ['atom', 'atoms', 'a']:
            projection = 'atom'
        elif projection.lower() in ['orbital', 'orbitals', 'orb', 'o']:
            projection = 'orbital'
        else:
            raise ValueError(f"Projection must be atom or orbital")

        if projection == 'atom':
            if name == 'mulliken':
                step_to = "MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS"
            elif name == 'loewdin':
                step_to = "LOEWDIN ATOMIC CHARGES AND SPIN POPULATIONS"

            def read_block(itt, step_to, reread=True):
                f = self.step_to(step_to, reread=reread)[0]
                if not f:
                    return None
                next(itt) # skip ---
                A = np.empty((self.na, 2), np.float64)
                for ia in range(self.na):
                    line = next(itt)
                    v = line.split()
                    A[ia] = float(v[-2]), float(v[-1])
                return A

        elif projection == 'orbital' and reduced:
            if name == 'mulliken':
                step_to = "MULLIKEN REDUCED ORBITAL CHARGES AND SPIN POPULATIONS"
            elif name == 'loewdin':
                step_to = "LOEWDIN REDUCED ORBITAL CHARGES AND SPIN POPULATIONS"

            def read_reduced_orbital_block(itt):
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

            def read_block(itt, step_to, reread=True):
                f = self.step_to(step_to, reread=reread)[0]
                if not f:
                    return None
                self.step_to("CHARGE", reread=False)
                charge = read_reduced_orbital_block(itt)
                self.step_to("SPIN", reread=False)
                spin = read_reduced_orbital_block(itt)
                if orbital is None:
                    return charge, spin
                else:
                    # atom-resolved (charge, spin) from dictionary
                    csa = np.zeros((self.na, 2), np.float64)
                    for key in charge:
                        ia, orb = key
                        if orb == orbital:
                            csa[ia, 0] = charge[key]
                            csa[ia, 1] = spin[key]
                    return csa

        elif projection == 'orbital' and not reduced:
            if name == 'mulliken':
                step_to = "MULLIKEN ORBITAL CHARGES AND SPIN POPULAITONS"
            elif name == 'loewdin':
                step_to = "LOEWDIN ORBITAL CHARGES AND SPIN POPULATIONS"

            def read_block(itt, step_to, reread=True):
                f = self.step_to(step_to, reread=reread)[0]
                if not f:
                    return None
                next(itt) # skip ---
                if "MULLIKEN" in step_to:
                    next(itt) # skip another line
                cso = np.empty((self.no, 2), np.float64) # orbital-resolved (charge, spin)
                csa = np.zeros((self.na, 2), np.float64) # atom-resolved (charge, spin)
                for io in range(self.no):
                    v = next(itt).split() # io, ia+element, orb, chg, spin
                    # split atom number and element from v[1]
                    ia, element = '', ''
                    for s in v[1]:
                        if s.isdigit():
                            ia += s
                        else:
                            element += s
                    ia = int(ia)
                    cso[io] = v[3:5]
                    if orbital == v[2]:
                        csa[ia] += cso[io]
                if orbital is None:
                    return cso
                else:
                    return csa

        itt = iter(self)
        blocks = []
        block = read_block(itt, step_to)
        while block is not None:
            blocks.append(block)
            block = read_block(itt, step_to, reread=False)

        if all:
            return blocks
        if len(blocks) > 0:
            return blocks[-1]
        return None


add_sile('output', outputSileORCA, gzip=True)
