# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from sisl import Atom, AtomicOrbital, Atoms, AtomUnknown, Geometry, PeriodicTable
from sisl._array import arrayi
from sisl._internal import set_module
from sisl.messages import deprecate_argument
from sisl.unit.siesta import unit_convert

from .._help import _fill_basis_empty
from ..sile import add_sile, sile_fh_open
from .sile import SileSiesta

__all__ = ["orbindxSileSiesta"]


Bohr2Ang = unit_convert("Bohr", "Ang")


@set_module("sisl.io.siesta")
class orbindxSileSiesta(SileSiesta):
    """Orbital information file"""

    @sile_fh_open()
    def read_lattice_nsc(self):
        """Reads the supercell number of supercell information"""
        # First line contains no no_s
        line = self.readline().split()
        no_s = int(line[1])
        self.readline()
        self.readline()
        nsc = [0] * 3

        def int_abs(i):
            return abs(int(i))

        for _ in range(no_s):
            line = self.readline().split()
            if len(line) == 16:
                isc = list(map(int_abs, line[12:15]))
                if isc[0] > nsc[0]:
                    nsc[0] = isc[0]
                if isc[1] > nsc[1]:
                    nsc[1] = isc[1]
                if isc[2] > nsc[2]:
                    nsc[2] = isc[2]

        return arrayi([n * 2 + 1 for n in nsc])

    @sile_fh_open()
    @deprecate_argument(
        "basis",
        "atoms",
        "use atoms instead of basis",
        "0.15",
        "0.17",
    )
    def read_basis(self, atoms: Optional[Union[Atoms, Geometry]] = None) -> Atoms:
        """Returns a set of atoms corresponding to the basis-sets in the ORB_INDX file

        The specie names have a short field in the ORB_INDX file, hence the name may
        not necessarily be the same as provided in the species block

        Parameters
        ----------
        atoms :
           list of atoms used for the species index
        """

        # First line contains no no_s
        line = self.readline().split()
        no = int(line[0])

        self.readline()
        self.readline()

        pt = PeriodicTable()

        if isinstance(atoms, Geometry):
            atoms = atoms.atoms

        if atoms is None:

            def crt_atom(i_s, tag, orbs):
                # The user has not specified an atomic basis
                i = pt.Z(tag)
                if isinstance(i, int):
                    # we can convert tag name to an atom
                    # Hence we don't need to add the tag
                    return Atom(i, orbs)
                return AtomUnknown(1000 + i_s, orbs, tag=tag)

        else:

            def crt_atom(i_s, tag, orbs):
                # Get the atom and add the orbitals
                kwargs = {}
                if atoms[i_s].tag != tag:
                    # we know ORB_INDX tag is correct
                    kwargs["tag"] = tag
                if len(atoms[i_s]) != len(orbs):
                    # only overwrite if # of orbitals don't match
                    kwargs["orbitals"] = orbs
                if kwargs:
                    return atoms[i_s].copy(**kwargs)
                return atoms[i_s]

        # Now we begin by reading the atoms
        atom, orbs = [], []
        species, order_species = [], []
        current_ia = 1
        tag = ""
        i_s = 0
        for _ in range(no):
            line = self.readline().split()

            ia = int(line[1])
            if ia != current_ia:
                if i_s not in species:
                    order_species.append(i_s)
                    atom.append(crt_atom(i_s, tag, orbs))
                species.append(i_s)
                current_ia = ia
                orbs = []

            # Get tag for atom
            tag = line[3]
            # and species number
            i_s = int(line[2]) - 1

            if i_s in order_species:
                # no need to collect information for the same orbital
                continue

            nlmz = list(map(int, line[5:9]))
            P = line[9] == "T"
            rc = float(line[11]) * Bohr2Ang
            # Create the orbital
            o = AtomicOrbital(n=nlmz[0], l=nlmz[1], m=nlmz[2], zeta=nlmz[3], P=P, R=rc)
            orbs.append(o)

        if i_s not in species:
            order_species.append(i_s)
            atom.append(crt_atom(i_s, tag, orbs))
        species.append(i_s)
        atom = Atoms([atom[i] for i in np.argsort(order_species)])

        return _fill_basis_empty(np.array(species), atom)


add_sile("ORB_INDX", orbindxSileSiesta, gzip=True)
