# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Import sile objects
from .sile import SileScaleUp
from ..sile import *

# Import the geometry object
from sisl import Atom, Atoms

__all__ = ['orboccSileScaleUp']


class orboccSileScaleUp(SileScaleUp):
    """ orbocc file object for ScaleUp """

    @sile_fh_open()
    def read_atom(self):
        """ Reads a the atoms and returns an `Atoms` object """
        self.readline()
        _, ns = map(int, self.readline().split()[:2])
        species = self.readline().split()[:ns] # species
        orbs = self.readline().split()[:ns] # orbs per species
        # Create list of species with correct # of orbitals per specie
        species = [Atom(s, [-1] * int(o)) for s, o in zip(species, orbs)]
        return Atoms(species)

add_sile('orbocc', orboccSileScaleUp, case=False, gzip=True)
