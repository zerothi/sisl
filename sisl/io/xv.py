"""
Sile object for reading/writing XV files
"""

from __future__ import print_function

# Import sile objects
from sisl.io.sile import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell
from sisl import Bohr

import numpy as np

__all__ = ['XVSile']


class XVSile(Sile):
    """ XV file object """

    def _setup(self):
        """ Setup the `XVSile` after initialization """
        self._comment = []

    def write_geom(self, geom):
        """ Writes the geometry to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

        if not hasattr(self, 'fh'):
            # The file-handle has not been opened
            with self:
                return self.write_geom(geom)

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

    def read_sc(self):
        """ Returns `SuperCell` object from the XV file """
        if not hasattr(self, 'fh'):
            # The file-handle has not been opened
            with self:
                return self.read_sc()

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

    def read_geom(self):
        """ Returns Geometry object from the XV file
        """
        if not hasattr(self, 'fh'):
            # The file-handle has not been opened
            with self:
                return self.read_geom()

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


if __name__ == "__main__":
    # Create geometry
    alat = 3.57
    dist = alat * 3. ** .5 / 4
    C = Atom(Z=6, R=dist * 1.01, orbs=2)
    geom = Geometry(np.array([[0, 0, 0], [1, 1, 1]], np.float64) * alat / 4,
                    atoms=C, sc=SuperCell(np.array([[0, 1, 1],
                                                    [1, 0, 1],
                                                    [1, 1, 0]], np.float64) * alat / 2))
    # Write stuff
    print(geom)
    geom.write(XVSile('diamond.XV', 'w'))
    geomr = XVSile('diamond.XV', 'r').read_geom()
    print(geomr)
    print(geomr.cell)
    print(geomr.xyz)
