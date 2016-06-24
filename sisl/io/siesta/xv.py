"""
Sile object for reading/writing XV files
"""

from __future__ import print_function

import numpy as np

# Import sile objects
from .sile import SileSIESTA
from ..sile import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell
from sisl import Bohr


__all__ = ['XVSile']


class XVSile(SileSIESTA):
    """ XV file object """

    def _setup(self):
        """ Setup the `XVSile` after initialization """
        self._comment = []


    @Sile_fh_open
    def write_geom(self, geom):
        """ Writes the geometry to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

        # Write unit-cell
        tmp = np.zeros(6, np.float64)
        fmt = ('   ' + '{:18.9f}' * 3) * 2 + '\n'
        for i in range(3):
            tmp[0:3] = geom.cell[i, :] * Bohr
            self._write(fmt.format(*tmp))
        self._write('{:12d}\n'.format(geom.na))
        fmt = '{:3d}{:6d}'
        fmt += '{:18.9f}' * 3 + '   ' + '{:18.9f}' * 3
        fmt += '\n'
        for ia, a, ips in geom.iter_species():
            tmp[0:3] = geom.xyz[ia, :] * Bohr
            self._write(fmt.format(ips + 1, a.Z, *tmp))


    @Sile_fh_open
    def read_sc(self):
        """ Returns `SuperCell` object from the XV file """

        cell = np.empty([3, 3], np.float64)
        for i in range(3):
            cell[
                i,
                :] = np.fromstring(
                self.readline(),
                dtype=float,
                sep=' ')[
                0:3]
        cell /= Bohr

        return SuperCell(cell)


    @Sile_fh_open
    def read_geom(self):
        """ Returns Geometry object from the XV file
        """
        sc = self.read_sc()

        # Read number of atoms
        na = int(self.readline())
        atms = [None] * na
        xyz = np.empty([na, 3], np.float64)
        line = np.empty(8, np.float64)
        for ia in range(na):
            line[:] = np.fromstring(self.readline(), dtype=float, sep=' ')[0:8]
            atms[ia] = Atom[int(line[1])]
            xyz[ia, :] = line[2:5]
        xyz /= Bohr

        return Geometry(xyz=xyz, atoms=atms, sc=sc)


add_sile('XV', XVSile, gzip=True)
