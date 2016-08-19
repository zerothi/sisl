"""
Sile object for reading/writing XYZ files
"""

from __future__ import print_function

import numpy as np

# Import sile objects
from .sile import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell


__all__ = ['XYZSile']


class XYZSile(Sile):
    """ XYZ file object """

    def _setup(self):
        """ Setup the `XYZSile` after initialization """
        self._comment = []

    @Sile_fh_open
    def write_geom(self, geom, fmt='.5f'):
        """ Writes the geometry to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

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


    @Sile_fh_open
    def read_geom(self):
        """ Returns Geometry object from the XYZ file

        NOTE: Unit-cell is the Eucledian 3D space.
        """

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

        return Geometry(xyz, atom=sp, sc=SuperCell(cell))


    def ArgumentParser(self, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geom().ArgumentParser(*args, **newkw)


add_sile('xyz', XYZSile, case=False, gzip=True)
