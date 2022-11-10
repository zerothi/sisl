# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
from .sile import SileORCA
from ..sile import add_sile, sile_fh_open

from sisl.utils import PropertyDict
from sisl._internal import set_module
from sisl.unit import units

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
            f = self.step_to("Number of basis functions")
            if f[0]:
                self._no = int(f[1].split()[-1])
            else:
                return None
        return self._no

    @sile_fh_open(True)
    def read_electrons(self, all=False):
        """ Read number of electrons (alpha, beta)

        Parameters
        ----------
        all : bool, optional
            return electron numbers from all steps (instead of last)

        Returns
        -------
        ndarray or list of ndarrays : alpha and beta electrons
        """

        def readE(itt):
            f = self.step_to("N(Alpha)", allow_reread=False)
            if f[0]:
                alpha = float(f[1].split()[-2])
                beta = float(next(itt).split()[-2])
            else:
                return None
            return alpha, beta

        itt = iter(self)
        E = []
        e = readE(itt)
        while e is not None:
            E.append(e)
            e = readE(itt)

        if all:
            return np.array(E)
        if len(E) > 0:
            return np.array(E[-1])
        return None

    @sile_fh_open(True)
    def read_charge(self, name='mulliken', projection='orbital', orbitals=None,
                    reduced=True, spin=False, all=False):
        """ Reads from charge (or spin) population analysis

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
        all: bool, optional
            return a list of all population analysis blocks instead of the last one

        Returns
        -------
        PropertyDicts or ndarray or lists : atom/orbital-resolved charge (or spin) data
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

            def read_block(itt, step_to):
                f, line = self.step_to(step_to, allow_reread=False)
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

            def read_block(itt, step_to):
                f, line = self.step_to(step_to, allow_reread=False)
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
                if orbitals is None:
                    return D
                else:
                    Da = np.zeros(self.na, np.float64)
                    for (ia, orb), d in D.items():
                        if orb == orbitals:
                            Da[ia] = d
                    return Da

        elif projection == 'orbital' and not reduced:
            if name == 'mulliken':
                step_to = "MULLIKEN ORBITAL CHARGES"
            elif name == 'loewdin':
                step_to = "LOEWDIN ORBITAL CHARGES"

            def read_block(itt, step_to):
                f, line = self.step_to(step_to, allow_reread=False)
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
                    if v[2] == orbitals:
                        Da[ia] += Do[io]
                if orbitals is None:
                    return Do
                else:
                    return Da

        itt = iter(self)
        blocks = []
        block = read_block(itt, step_to)
        while block is not None:
            blocks.append(block)
            block = read_block(itt, step_to)

        if all:
            return blocks
        if len(blocks) > 0:
            return blocks[-1]
        return None

    @sile_fh_open(True)
    def read_energy(self, all=False):
        """ Reads the energy blocks

        Parameters
        ----------
        all : bool, optional
            return a list of dictionaries from each step (instead of the last)

        Returns
        -------
        PropertyDict or list of PropertyDict : all energy data (in eV) from the "TOTAL SCF ENERGY" and "DFT DISPERSION CORRECTION" blocks
        """
        def readE(itt, vdw, reopen=False):
            f = self.step_to("TOTAL SCF ENERGY", reopen=reopen, allow_reread=False)[0]
            if not f:
                return None
            next(itt) # skip ---
            next(itt) # skip blank line
            line = next(itt)
            E = PropertyDict()
            while "----" not in line:
                v = line.split()
                if "Total Energy" in line:
                    E["total"] = float(v[-4]) * units('Ha', 'eV')
                elif "E(X)" in line:
                    E["exchange"] = float(v[-2]) * units('Ha', 'eV')
                elif "E(C)" in line:
                    E["correlation"] = float(v[-2]) * units('Ha', 'eV')
                elif "E(XC)" in line:
                    E["xc"] = float(v[-2]) * units('Ha', 'eV')
                elif "DFET-embed. en." in line:
                    E["embedding"] = float(v[-2]) * units('Ha', 'eV')
                line = next(itt)
            if vdw:
                self.step_to("DFT DISPERSION CORRECTION")[1]
                v = self.step_to("Dispersion correction")[1].split()
                E["vdw"] = float(v[-1]) * units('Ha', 'eV')
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

    @sile_fh_open(True)
    def read_orbital_energies(self, all=False):
        """ Reads the "ORBITAL ENERGIES" blocks

        Parameters
        ----------
        all : bool, optional
            return a list of ndarrays from each step (instead of the last)

        Returns
        -------
        ndarray or list : orbital energies (in eV) from the "ORBITAL ENERGIES" blocks
        """
        def readE(itt):
            f = self.step_to("ORBITAL ENERGIES", allow_reread=False)[0]
            if not f:
                return None
            next(itt) # skip ---
            line = next(itt)
            if "SPIN UP ORBITALS" in line:
                spin = True
            else:
                spin = False
            next(itt) # Skip "NO OCC" header line
            E = np.empty((self.no, 2), np.float64)
            v = next(itt).split()
            while len(v) > 0:
                i = int(v[0])
                E[i, 0] = float(v[-1])
                v = next(itt).split()
            if not spin:
                return E[:, 0]

            next(itt) # skip "SPIN DOWN ORBITALS"
            next(itt) # Skip "NO OCC" header line
            v = next(itt).split()
            while len(v) > 0 and '---' not in v[0]:
                i = int(v[0])
                E[i, 1] = float(v[-1])
                v = next(itt).split()
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


add_sile('output', outputSileORCA, gzip=True)
