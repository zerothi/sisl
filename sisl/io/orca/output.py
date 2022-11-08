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
    def read_electrons(self, all=False):
        """ Read number of electrons (alpha, beta)

        Parameters
        ----------
        all : bool, optional
            return electron numbers from all steps (instead of last)
        """

        def readE(itt, reopen=False):
            f = self.step_to("N(Alpha)", reopen=reopen, allow_reread=False)
            if f[0]:
                alpha = float(f[1].split()[-2])
                beta = float(next(itt).split()[-2])
            else:
                return None
            return alpha, beta

        itt = iter(self)
        E = []
        e = readE(itt, reopen=True)
        while e is not None:
            E.append(e)
            e = readE(itt)

        if all:
            return np.array(E)
        if len(E) > 0:
            return np.array(E[-1])
        return None

    @sile_fh_open()
    def read_charge(self, name='mulliken', projection='orbital', orbital=None,
                    reduced=True, spin=False, all=False):
        """ Reads from charge (or spin) population analysis

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
        spin : bool, optional
            whether to return the spin block instead of charge
        all: bool, optional
            return a list of all population analysis blocks instead of the last one

        Returns
        -------
        PropertyDicts or ndarray : atom/orbital-resolved charge (or spin) data
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
                step_to = "MULLIKEN ATOMIC CHARGES"
            elif name == 'loewdin':
                step_to = "LOEWDIN ATOMIC CHARGES"

            def read_block(itt, step_to, reopen=False):
                f, line = self.step_to(step_to, reopen=reopen, allow_reread=False)
                if not f:
                    return None
                next(itt) # skip ---
                if "SPIN" in line:
                    spin_block = True
                else:
                    spin_block = False
                A = np.empty(self.na, np.float64)
                for ia in range(self.na):
                    line = next(itt)
                    v = line.split()
                    if spin_block and not spin:
                        A[ia] = float(v[-2])
                    elif not spin_block and spin:
                        return None
                    else:
                        A[ia] = float(v[-1])
                return A

        elif projection == 'orbital' and reduced:
            if name == 'mulliken':
                step_to = "MULLIKEN REDUCED ORBITAL CHARGES"
            elif name == 'loewdin':
                step_to = "LOEWDIN REDUCED ORBITAL CHARGES"

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

            def read_block(itt, step_to, reopen=False):
                f, line = self.step_to(step_to, reopen=reopen, allow_reread=False)
                if not f:
                    return None
                if "SPIN" in line:
                    spin_block = True
                else:
                    spin_block = False
                if spin_block and spin:
                    self.step_to("SPIN")
                elif spin_block:
                    self.step_to("CHARGE")
                elif not spin:
                    next(itt) # skip ---
                else:
                    return None
                D = read_reduced_orbital_block(itt)
                if orbital is None:
                    return D
                else:
                    Da = np.zeros(self.na, np.float64)
                    for key in D:
                        ia, orb = key
                        if orb == orbital:
                            Da[ia] = D[key]
                    return Da

        elif projection == 'orbital' and not reduced:
            if name == 'mulliken':
                step_to = "MULLIKEN ORBITAL CHARGES"
            elif name == 'loewdin':
                step_to = "LOEWDIN ORBITAL CHARGES"

            def read_block(itt, step_to, reopen=False):
                f, line = self.step_to(step_to, reopen=reopen, allow_reread=False)
                if "SPIN" in line:
                    spin_block = True
                else:
                    spin_block = False
                if not f:
                    return None
                next(itt) # skip ---
                if "MULLIKEN" in step_to:
                    next(itt) # skip line "The uncorrected..."
                Do = np.empty(self.no, np.float64) # orbital-resolved
                Da = np.zeros(self.na, np.float64) # atom-resolved
                for io in range(self.no):
                    v = next(itt).split() # io, ia+element, orb, chg, (spin)
                    # split atom number and element from v[1]
                    ia, element = '', ''
                    for s in v[1]:
                        if s.isdigit():
                            ia += s
                        else:
                            element += s
                    ia = int(ia)
                    if spin_block and spin:
                        Do[io] = float(v[4])
                    elif not spin_block and spin:
                        return None
                    else:
                        Do[io] = float(v[3])
                    if orbital == v[2]:
                        Da[ia] += Do[io]
                if orbital is None:
                    return Do
                else:
                    return Da

        itt = iter(self)
        blocks = []
        block = read_block(itt, step_to, reopen=True)
        while block is not None:
            blocks.append(block)
            block = read_block(itt, step_to)

        if all:
            return blocks
        if len(blocks) > 0:
            return blocks[-1]
        return None

    @sile_fh_open()
    def read_energy(self, all=False, convert=True):
        """ Reads the energy specification from ORCA output file

        Parameters
        ----------
        all: bool, optional
            return a list of dictionaries from each step

        Returns
        -------
        PropertyDict : all data from the "TOTAL SCF ENERGY" segment
        """

        Hartree2eV = 27.2113834

        def readE(itt, vdw, reopen=False):
            if convert:
                sc = Hartree2eV
            else:
                sc = 1
            f = self.step_to("TOTAL SCF ENERGY", reopen=reopen, allow_reread=False)[0]
            if not f:
                return None
            next(itt) # skip ---
            next(itt) # skip blank line
            line = next(itt)
            print(line)
            E = PropertyDict()
            while "----" not in line:
                v = line.split()
                if "Total Energy" in line:
                    E["total"] = float(v[-4]) * sc
                elif "E(X)" in line:
                    E["exchange"] = float(v[-2]) * sc
                elif "E(C)" in line:
                    E["correlation"] = float(v[-2]) * sc
                elif "E(XC)" in line:
                    E["xc"] = float(v[-2]) * sc
                elif "DFET-embed. en." in line:
                    E["embedding"] = float(v[-2]) * sc
                line = next(itt)
            if vdw:
                self.step_to("DFT DISPERSION CORRECTION")[1]
                v = self.step_to("Dispersion correction")[1].split()
                E["vdw"] = float(v[-1]) * sc
            return E

        # check if vdw block is present
        vdw = self.step_to("DFT DISPERSION CORRECTION")[0]

        itt = iter(self)
        E = []
        e = readE(itt, vdw, reopen=True)
        while e is not None:
            E.append(e)
            e = readE(itt, vdw)

        if all:
            return E
        if len(E) > 0:
            return E[-1]
        return None


add_sile('output', outputSileORCA, gzip=True)
