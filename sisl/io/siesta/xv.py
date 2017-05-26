"""
Sile object for reading/writing XV files
"""

from __future__ import print_function

import numpy as np

# Import sile objects
from .sile import SileSiesta
from ..sile import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell
from sisl.units.siesta import unit_convert

Bohr2Ang = unit_convert('Bohr', 'Ang')

__all__ = ['XVSileSiesta']


class XVSileSiesta(SileSiesta):
    """ XV file object """

    def _setup(self):
        """ Setup the `XVSileSiesta` after initialization """
        self._comment = []

    @Sile_fh_open
    def write_geometry(self, geom, fmt='.8f'):
        """ Writes the geometry to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

        # Write unit-cell
        tmp = np.zeros(6, np.float64)

        # Create format string for the cell-parameters
        fmt_str = ('   ' + ('{:' + fmt + '} ') * 3) * 2 + '\n'
        for i in range(3):
            tmp[0:3] = geom.cell[i, :] / Bohr2Ang
            self._write(fmt_str.format(*tmp))
        self._write('{:12d}\n'.format(geom.na))

        # Create format string for the atomic coordinates
        fmt_str = '{:3d}{:4d} '
        fmt_str += ('{:' + fmt + '} ') * 3 + '   '
        fmt_str += ('{:' + fmt + '} ') * 3 + '\n'
        for ia, a, ips in geom.iter_species():
            tmp[0:3] = geom.xyz[ia, :] / Bohr2Ang
            self._write(fmt_str.format(ips + 1, a.Z, *tmp))

    @Sile_fh_open
    def read_supercell(self):
        """ Returns `SuperCell` object from the XV file """

        cell = np.empty([3, 3], np.float64)
        for i in range(3):
            cell[i, :] = list(map(float, self.readline().split()[:3]))
        cell *= Bohr2Ang

        return SuperCell(cell)

    @Sile_fh_open
    def read_geometry(self):
        """ Returns Geometry object from the XV file
        """
        sc = self.read_supercell()

        # Read number of atoms
        na = int(self.readline())
        atms = [None] * na
        xyz = np.empty([na, 3], np.float64)
        line = np.empty(8, np.float64)
        for ia in range(na):
            line[:] = list(map(float, self.readline().split()[:8]))
            atms[ia] = Atom[int(line[1])]
            xyz[ia, :] = line[2:5]
        xyz *= Bohr2Ang

        return Geometry(xyz, atms, sc=sc)

    def ArgumentParser(self, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geometry().ArgumentParser(*args, **newkw)


add_sile('XV', XVSileSiesta, gzip=True)
