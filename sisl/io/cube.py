"""
Sile object for reading/writing CUBE files
"""

from __future__ import print_function

import numpy as np

# Import sile objects
from sisl.io.sile import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell, Grid
from sisl.units import unit_convert

__all__ = ['CUBESile']

Ang2Bohr = unit_convert('Ang', 'Bohr')


class CUBESile(Sile):
    """ CUBE file object """

    def _setup(self):
        """ Setup the `CUBESile` after initialization """
        self._comment = []

    @Sile_fh_open
    def write_geom(
            self,
            geom,
            size=None,
            fmt='15.10e',
            origo=None,
            *args,
            **kwargs):
        """ Writes the `Geometry` object attached to this grid """
        sile_raise_write(self)

        # Write header
        self._write('\n')
        self._write('sisl --- CUBE file\n')

        if size is None:
            size = np.ones([3], np.int32)
        if origo is None:
            origo = np.zeros([3], np.float64)
            
        _fmt = '{:d} {:15.10e} {:15.10e} {:15.10e}\n'

        # Add #-of atoms and origo
        self._write(_fmt.format(len(geom), *origo * Ang2Bohr))

        # Write the cell and voxels
        dcell = np.empty([3, 3], np.float64)
        for ix in range(3):
            dcell[ix, :] = geom.cell[ix, :] / size[ix] * Ang2Bohr
        self._write(_fmt.format(size[0], *dcell[0, :]))
        self._write(_fmt.format(size[1], *dcell[1, :]))
        self._write(_fmt.format(size[2], *dcell[2, :]))

        tmp = ' {:' + fmt + '}'
        _fmt = '{:d} 0.0' + tmp + tmp + tmp + '\n'
        for ia in geom:
            self._write(_fmt.format(geom.atom[ia].Z, *geom.xyz[ia, :] * Ang2Bohr))

    @Sile_fh_open
    def write_grid(self, grid, fmt='%.5e', *args, **kwargs):
        """ Writes the geometry to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

        # Write the geometry
        self.write_geom(grid.geom, size=grid.grid.shape, *args, **kwargs)

        g_size = np.copy(grid.grid.shape)

        grid.grid.shape = (-1,)

        # Write the grid
        np.savetxt(self.fh, grid.grid[:], fmt)

        # Add a finishing line to ensure empty ending
        self._write('\n')

        grid.grid.shape = g_size

    @Sile_fh_open
    def read_sc(self, na=False):
        """ Returns `SuperCell` object from the CUBE file

        If `na=True` it will return a tuple (na,SuperCell)
        """

        self.readline()  # header 1
        self.readline()  # header 2
        tmp = self.readline().split()  # origo
        origo = [float(x) for x in tmp[1:4]]
        na = int(tmp[0])

        cell = np.empty([3, 3], np.float64)
        for i in [0, 1, 2]:
            tmp = self.readline().split()
            s = int(tmp[0])
            tmp = tmp[1:]
            for j in [0, 1, 2]:
                cell[i, j] = float(tmp[j]) * s

        cell = cell / Ang2Bohr
        if na:
            return na, SuperCell(cell)
        return SuperCell(cell)

    @Sile_fh_open
    def read_geom(self):
        """ Returns `Geometry` object from the CUBE file """
        na, sc = self.read_sc(na=True)

        # Start reading the geometry
        xyz = np.empty([na, 3], np.float64)
        atom = []
        for ia in range(na):
            tmp = self.readline().split()
            atom.append( Atom( int(tmp[0]) ) )
            xyz[ia, 0] = float(tmp[2])
            xyz[ia, 1] = float(tmp[3])
            xyz[ia, 2] = float(tmp[4])

        xyz /= Ang2Bohr
        return Geometry(xyz, atom, sc=sc)

    @Sile_fh_open
    def read_grid(self):
        """ Returns `Grid` object from the CUBE file """
        geom = self.read_geom()

        # Now seek behind to read grid sizes
        self.fh.seek(0)

        # Skip headers and origo
        self.readline()
        self.readline()
        na = int(self.readline().split()[0])

        ngrid = [0] * 3
        for i in [0, 1, 2]:
            tmp = self.readline().split()
            ngrid[i] = int(tmp[0])

        # Read past the atoms
        for i in range(na):
            self.readline()

        grid = Grid(ngrid, dtype=np.float32, geom=geom)
        grid.grid = np.loadtxt(self.fh, dtype=grid.dtype)
        grid.grid.shape = ngrid

        return grid


add_sile('cube', CUBESile, case=False, gzip=True)
