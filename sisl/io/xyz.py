"""
Sile object for reading/writing XYZ files
"""

from __future__ import print_function

# Import sile objects
from sisl.io.sile import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell

import numpy as np

__all__ = ['XYZSile']


class XYZSile(Sile):
    """ XYZ file object """

    def _setup(self):
        """ Setup the `XYZSile` after initialization """
        self._comment = []

    def write_geom(self, geom, fmt='.5f'):
        """ Writes the geometry to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

        if not hasattr(self, 'fh'):
            # The file-handle has not been opened
            with self:
                return self.write_geom(geom, fmt)

        # Write out the cell
        self._write('  ' + str(len(geom)) + '\n')
        # We write the cell coordinates as the cell coordinates
        fmt_str = '{{:{0}}} '.format(fmt) * 9 + '\n'
        self._write(fmt_str.format(*geom.cell.flatten()))

        fmt_str = '{{0:2s}}  {{1:{0}}}  {{2:{0}}}  {{3:{0}}}\n'.format(fmt)
        for ia, a, isp in geom.iter_species():
            self._write(fmt_str.format(a.symbol, *geom.xyz[ia, :]))
        # Add a single new line
        self._write('\n')

    def read_geom(self):
        """ Returns Geometry object from the XYZ file

        NOTE: Unit-cell is the Eucledian 3D space.
        """
        if not hasattr(self, 'fh'):
            # The file-handle has not been opened
            with self:
                return self.read_geom()

        cell = np.asarray(np.diagflat([1] * 3), np.float64)
        l = self.readline()
        na = int(l)
        l = self.readline()
        l = l.split()
        if len(l) == 9:
            # we possibly have the cell as a comment
            cell.shape = (9,)
            for i, il in enumerate(l):
                cell[i] = float(il)
            cell.shape = (3, 3)
        sp = [None] * na
        xyz = np.empty([na, 3], np.float64)
        for ia in range(na):
            l = self.readline().split()
            sp[ia] = l.pop(0)
            xyz[ia, :] = [float(k) for k in l[:3]]

        return Geometry(xyz, atoms=sp, sc=SuperCell(cell))


if __name__ == "__main__":
    # Create geometry
    alat = 3.57
    dist = alat * 3. ** .5 / 4
    C = Atom(Z=6, R=dist * 1.01, orbs=2)
    sc = SuperCell(np.array([[0, 1, 1],
                             [1, 0, 1],
                             [1, 1, 0]], np.float64) * alat / 2)
    geom = Geometry(np.array([[0, 0, 0], [1, 1, 1]], np.float64) * alat / 4,
                    atoms=C, sc=sc)
    # Write stuff
    print(geom)
    geom.write(XYZSile('diamond.xyz', 'w'))
    geomr = XYZSile('diamond.xyz', 'r').read_geom()
    print(geomr)
    print(geomr.cell)
    print(geomr.xyz)
