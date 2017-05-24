"""
Sile object for reading orbocc files from ScaleUp
"""

from __future__ import division, print_function

# Import sile objects
from .sile import SileScaleUp
from ..sile import *

# Import the geometry object
from sisl import Atom, Atoms

__all__ = ['orboccSileScaleUp']


class orboccSileScaleUp(SileScaleUp):
    """ orbocc file object for ScaleUp """

    @Sile_fh_open
    def read_atom(self):
        """ Reads a the atoms and returns an `Atoms` object """
        self.readline()
        _, ns = map(int, self.readline().split()[:2])
        species = self.readline().split()[:ns] # species
        orbs = self.readline().split()[:ns] # orbs per species
        # Create list of species with correct # of orbitals per specie
        species = [Atom(s, orbs=int(o)) for s, o in zip(species, orbs)]
        return Atoms(species)

add_sile('orbocc', orboccSileScaleUp, case=False, gzip=True)
