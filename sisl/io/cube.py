from __future__ import print_function

import numpy as np

# Import sile objects
from sisl.io.sile import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell, Grid, SislError
from sisl.unit import unit_convert

__all__ = ['cubeSile']

Ang2Bohr = unit_convert('Ang', 'Bohr')


class cubeSile(Sile):
    """ CUBE file object """

    @sile_fh_open()
    def write_supercell(self, sc, fmt='15.10e', size=None, origo=None,
                        *args, **kwargs):
        """ Writes `SuperCell` object attached to this grid

        Parameters
        ----------
        sc : SuperCell
            supercell to be written
        fmt : str, optional
            floating point format for stored values
        size : (3, ), optional
            shape of the stored grid (``[1, 1, 1]``)
        origo : (3, ), optional
            origo of the cell (``[0, 0, 0]``)
        """
        sile_raise_write(self)

        # Write header
        self._write('\n')
        self._write('sisl --- CUBE file\n')

        if size is None:
            size = np.ones([3], np.int32)
        if origo is None:
            origo = sc.origo[:]

        _fmt = '{:d} {:15.10e} {:15.10e} {:15.10e}\n'

        # Add #-of atoms and origo
        self._write(_fmt.format(1, *(origo * Ang2Bohr)))

        # Write the cell and voxels
        for ix in range(3):
            dcell = sc.cell[ix, :] / size[ix] * Ang2Bohr
            self._write(_fmt.format(size[ix], *dcell))

        self._write('1 0. 0. 0. 0.\n')

    @sile_fh_open()
    def write_geometry(self, geometry, fmt='15.10e', size=None, origo=None,
            *args, **kwargs):
        """ Writes `Geometry` object attached to this grid

        Parameters
        ----------
        geometry : Geometry
            geometry to be written
        fmt : str, optional
            floating point format for stored values
        size : (3, ), optional
            shape of the stored grid (``[1, 1, 1]``)
        origo : (3, ), optional
            origo of the cell (``[0, 0, 0]``)
        """
        sile_raise_write(self)

        # Write header
        self._write('\n')
        self._write('sisl --- CUBE file\n')

        if size is None:
            size = np.ones([3], np.int32)
        if origo is None:
            origo = geometry.origo[:]

        _fmt = '{:d} {:15.10e} {:15.10e} {:15.10e}\n'

        valid_Z = (geometry.atoms.Z > 0).nonzero()[0]
        geometry = geometry.sub(valid_Z)

        # Add #-of atoms and origo
        self._write(_fmt.format(len(geometry), *(origo * Ang2Bohr)))

        # Write the cell and voxels
        for ix in range(3):
            dcell = geometry.cell[ix, :] / size[ix] * Ang2Bohr
            self._write(_fmt.format(size[ix], *dcell))

        tmp = ' {:' + fmt + '}'
        _fmt = '{:d} 0.0' + tmp + tmp + tmp + '\n'
        for ia in geometry:
            self._write(_fmt.format(geometry.atoms[ia].Z, *geometry.xyz[ia, :] * Ang2Bohr))

    @sile_fh_open()
    def write_grid(self, grid, fmt='.5e', imag=False, *args, **kwargs):
        """ Write `Grid` to the contained file

        Parameters
        ----------
        grid : Grid
           the grid to be written in the CUBE file
        fmt : str, optional
           format used for precision output
        imag : bool, optional
           write only imaginary part of the grid, default to only writing the
           real part.
        buffersize : int, optional
           size of the buffer while writing the data, (6144)
        """
        # Check that we can write to the file
        sile_raise_write(self)

        if grid.geometry is None:
            self.write_supercell(grid.sc, size=grid.shape, *args, **kwargs)
        else:
            self.write_geometry(grid.geometry, size=grid.shape, *args, **kwargs)

        buffersize = kwargs.get('buffersize', min(6144, grid.grid.size))
        buffersize += buffersize % 6 # ensure multiple of 6

        # A CUBE file contains grid-points aligned like this:
        # for x
        #   for y
        #     for z
        #       write...
        _fmt1 = '{:' + fmt + '} '
        _fmt6 = (_fmt1 * 6)[:-1] + '\n'
        __fmt = _fmt6 * (buffersize // 6)

        if imag:
            for z in np.nditer(np.asarray(grid.grid.imag, order='C').reshape(-1), flags=['external_loop', 'buffered'],
                               op_flags=[['readonly']], order='C', buffersize=buffersize):
                if z.shape[0] != buffersize:
                    s = z.shape[0]
                    __fmt = _fmt6 * (s // 6) + _fmt1 * (s % 6) + '\n'
                self._write(__fmt.format(*z.tolist()))
        else:
            for z in np.nditer(np.asarray(grid.grid.real, order='C').reshape(-1), flags=['external_loop', 'buffered'],
                               op_flags=[['readonly']], order='C', buffersize=buffersize):
                if z.shape[0] != buffersize:
                    s = z.shape[0]
                    __fmt = _fmt6 * (s // 6) + _fmt1 * (s % 6) + '\n'
                self._write(__fmt.format(*z.tolist()))

        # Add a finishing line to ensure empty ending
        self._write('\n')

    @sile_fh_open()
    def read_supercell(self, na=False):
        """ Returns `SuperCell` object from the CUBE file

        Parameters
        ----------
        na : bool, optional
           whether to also return the number of atoms in the geometry
        """
        self.readline()  # header 1
        self.readline()  # header 2
        origo = self.readline().split() # origo
        lna = int(origo[0])
        origo = np.array(list(map(float, origo[1:])), np.float64)

        cell = np.empty([3, 3], np.float64)
        for i in [0, 1, 2]:
            tmp = self.readline().split()
            s = int(tmp[0])
            tmp = tmp[1:]
            for j in [0, 1, 2]:
                cell[i, j] = float(tmp[j]) * s

        cell = cell / Ang2Bohr
        origo = origo / Ang2Bohr
        if na:
            return lna, SuperCell(cell, origo=origo)
        return SuperCell(cell, origo=origo)

    @sile_fh_open()
    def read_geometry(self):
        """ Returns `Geometry` object from the CUBE file """
        na, sc = self.read_supercell(na=True)

        if na == 0:
            return None

        # Start reading the geometry
        xyz = np.empty([na, 3], np.float64)
        atom = []
        for ia in range(na):
            tmp = self.readline().split()
            atom.append(Atom(int(tmp[0])))
            xyz[ia, 0] = float(tmp[2])
            xyz[ia, 1] = float(tmp[3])
            xyz[ia, 2] = float(tmp[4])

        xyz /= Ang2Bohr
        return Geometry(xyz, atom, sc=sc)

    @sile_fh_open()
    def read_grid(self, imag=None):
        """ Returns `Grid` object from the CUBE file

        Parameters
        ----------
        imag : str or Sile or Grid
            the imaginary part of the grid. If the geometries does not match
            an error will be raised.
        """
        if not imag is None:
            if not isinstance(imag, Grid):
                imag = Grid.read(imag)
        geom = self.read_geometry()
        if geom is None:
            self.fh.seek(0)
            sc = self.read_supercell()
        else:
            sc = geom.sc

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

        if geom is None:
            grid = Grid(ngrid, dtype=np.float64, sc=sc)
        else:
            grid = Grid(ngrid, dtype=np.float64, geometry=geom)
        grid.grid.shape = (-1,)

        # TODO check performance of this
        # We are currently doing this to enable reading
        #  1-column data and 6-column data.
        lines = [item for sublist in self.fh.readlines() for item in sublist.split()]
        grid.grid[:] = np.array(lines).astype(grid.dtype)
        grid.grid.shape = ngrid

        if imag is None:
            return grid

        # We are expecting an imaginary part
        if not grid.geometry.equal(imag.geometry):
            raise SislError(str(self) + ' and its imaginary part does not have the same '
                            'geometry. Hence a combined complex Grid cannot be formed.')
        if grid != imag:
            raise SislError(str(self) + ' and its imaginary part does not have the same '
                            'shape. Hence a combined complex Grid cannot be formed.')

        # Now we have a complex grid
        grid.grid = grid.grid + 1j * imag.grid

        return grid


add_sile('cube', cubeSile, case=False, gzip=True)
