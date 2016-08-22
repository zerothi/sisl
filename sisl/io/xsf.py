"""
Sile object for reading/writing XSF (XCrySDen) files
"""

from __future__ import print_function

import numpy as np

# Import sile objects
from .sile import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell


__all__ = ['XSFSile']


class XSFSile(Sile):
    """ XSF file object """

    def _setup(self):
        """ Setup the `XSFSile` after initialization """
        self._comment = ['#']

    @Sile_fh_open
    def write_geom(self, geom, fmt='.8f'):
        """ Writes the geometry to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

        # The current geometry is currently only a single
        # one, and does not write the convvec
        # Is it a necessity?

        # Write out top-header stuff
        self._write('# File created by: sisl\n\n')

        self._write('CRYSTAL\n\n')

        self._write('# Primitive lattice vectors:\n\n')
        self._write('PRIMVEC\n')
        # We write the cell coordinates as the cell coordinates
        fmt_str = '{{:{0}}} '.format(fmt) * 3 + '\n'
        for i in [0, 1, 2]:
            self._write(fmt_str.format(*geom.cell[i,:]))

        # Currently not written (we should convert the geometry
        # to a conventional cell (90-90-90))
        # It seems this simply allows to store both formats in
        # the same file.
        #self._write('\n# Conventional lattice vectors:\n\n')
        #self._write('CONVVEC\n')
        #convcell = 
        #for i in [0, 1, 2]:
        #    self._write(fmt_str.format(*convcell[i,:]))

        self._write('\n# Atomic coordinates (in primitive coordinates)\n\n')
        self._write('PRIMCOORD\n')
        self._write('{} {}\n'.format(len(geom), 1))

        fmt_str = '{{0:3d}}  {{1:{0}}}  {{2:{0}}}  {{3:{0}}}\n'.format(fmt)
        for ia in geom:
            self._write(fmt_str.format(geom.atom[ia].Z, *geom.xyz[ia, :]))
        # Add a single new line
        self._write('\n')


    @Sile_fh_open
    def read_geom(self):
        """ Returns Geometry object from the XSF file
        """
        # Prepare containers...
        cell = np.zeros([3, 3], np.float64)
        cell_set = False
        atom = []
        xyz = []
        na = 0

        line = ' '
        while True:
            # skip comments
            line = self.readline()

            # We prefer the 
            if line.startswith('CONVVEC') and not cell_set:
                for i in [0, 1, 2]:
                    line = self.readline()
                    cell[i,:] = map(float, line.split())

            elif line.startswith('PRIMVEC'):
                cell_set = True
                for i in [0, 1, 2]:
                    line = self.readline()
                    cell[i,:] = map(float, line.split())

            elif line.startswith('PRIMCOORD'):
                # First read # of atoms
                line = self.readline().split()
                na = int(line[0])
                
                # currently line[1] is unused!
                for i in range(na):
                    line = self.readline().split()
                    atom.append(int(line[0]))
                    xyz.append(map(float, line[1:]))

        return Geometry(xyz, atom=atom, sc=SuperCell(cell))


    def ArgumentParser(self, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geom().ArgumentParser(*args, **newkw)


add_sile('xsf', XSFSile, case=False, gzip=True)
