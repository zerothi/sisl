"""
Sile object for reading/writing CONTCAR/POSCAR files
"""

from __future__ import print_function

import numpy as np

# Import sile objects
from .sile import SileVASP
from ..sile import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell

__all__ = ['CARSile', 'POSCARSile', 'CONTCARSile']


class CARSile(SileVASP):
    """ CAR file object
    This file-object handles both POSCAR and CONTCAR files
    """

    def _setup(self):
        """ Setup the `POSCARSile` after initialization """
        self._comment = []
        self._scale = 1.

    @Sile_fh_open
    def write_geom(self, geom):
        """ Writes the geometry to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

        # LABEL
        self._write('sisl output\n')

        # Scale
        self._write('  1.\n')

        # Write unit-cell
        fmt = ('   ' + '{:18.9f}' * 3) * 2 + '\n'
        for i in range(3):
            tmp[0:3] = geom.cell[i, :]
            self._write(fmt.format(*geom.cell[i, :]))

        # Figure out how many species
        d = []
        for ia, a, idx_specie in geom.iter_species():
            if idx_specie > len(d):
                d.append(0)
            d[idx_specie] += + 1
        fmt = ' ' + '{:d}' * len(d) + '\n'
        self._write(fmt.format(*d))
        self._write('Cartesian\n')

        fmt = '{:18.9f}' * 3 + '\n'
        for ia in geom:
            self._write(fmt.format(*geom.xyz[ia, :]))

    @Sile_fh_open
    def read_sc(self):
        """ Returns `SuperCell` object from the CONTCAR/POSCAR file """

        # read first line
        self.readline()  # LABEL
        # Update scale-factor
        self._scale = float(self.readline())

        # Read cell vectors
        cell = np.empty([3, 3], np.float64)
        for i in range(3):
            cell[
                i,
                :] = np.fromstring(
                self.readline(),
                dtype=float,
                count=3,
                sep=' ')
        cell *= self._scale

        return SuperCell(cell)

    @Sile_fh_open
    def read_geom(self):
        """ Returns Geometry object from the CONTCAR/POSCAR file
        """
        sc = self.read_sc()

        # First line is the species names/numbers
        species = self.readline().split()
        # Get number of each species in a list
        species_count = map(int, self.readline().split())
        if len(species) != len(species_count):
            err = '\n'.join([
                "POSTCAR format requires format:",
                "  <Specie-1> <Specie-2>",
                "  <#Specie-1> <#Specie-2>",
                "on the 6th and 7th line."])
            raise SileError(err)

        # Create list of atoms to be used subsequently
        atoms = [Atom[spec]
                 for spec, nsp in zip(species, species_count)
                 for i in range(nsp)]

        # Read whether this is selective or direct
        opt = self.readline()
        direct = True
        if opt[0] in 'Ss':
            direct = False
            opt = self.readline()

        # Check whether this is in fractional or direct
        # coordinates
        cart = False
        if opt[0] in 'CcKk':
            cart = True

        # Number of atoms
        na = len(atoms)

        xyz = np.empty([na, 3], np.float64)
        aoff = 0
        for ia in range(na):
            xyz[ia, :] = np.fromstring(
                self.readline(), dtype=float, count=3, sep=' ')
        if cart:
            # The unit of the coordinates are cartesian
            xyz *= self._scale
        else:
            xyz = np.dot(xyz, sc.cell.T)

        # The POT/CONT-CAR does not contain information on the atomic species
        return Geometry(xyz=xyz, atoms=atoms, sc=sc)


# Equivalent classes
class POSCARSile(CARSile):
    pass

class CONTCARSile(CARSile):
    pass


add_sile('CAR', CARSile, gzip=True)
add_sile('POSCAR', POSCARSile, gzip=True)
add_sile('CONTCAR', CONTCARSile, gzip=True)
