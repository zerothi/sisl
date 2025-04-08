# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np

from sisl import Atom, Geometry, Grid, Lattice, SislError
from sisl._internal import set_module

# Import sile objects
from sisl.io.sile import *
from sisl.messages import deprecate_argument
from sisl.unit import unit_convert

from ._help import header_to_dict

__all__ = ["cubeSile"]


@set_module("sisl.io")
class cubeSile(Sile):
    """CUBE file object

    By default the cube file is written using Bohr units.
    one can define the units by passing a respective unit argument.
    Note that the grid data is assumed unit-less and thus no conversion
    will be done for this data, only atomic coordinates and lattice vectors.
    """

    @sile_fh_open()
    @deprecate_argument("sc", "lattice", "use lattice= instead of sc=", "0.15", "0.17")
    def write_lattice(
        self,
        lattice: Lattice,
        fmt: str = "15.10e",
        size=None,
        origin=None,
        unit: str = "Bohr",
        *args,
        **kwargs,
    ):
        """Writes `Lattice` object attached to this grid

        Parameters
        ----------
        lattice :
            lattice to be written
        fmt :
            floating point format for stored values
        size : (3, ), optional
            shape of the stored grid (``[1, 1, 1]``)
        origin : (3, ), optional
            origin of the cell (``[0, 0, 0]``)
        unit:
            what length unit should the cube file data be written in
        """
        sile_raise_write(self)

        Ang2unit = unit_convert("Ang", unit)

        # Write header
        self._write("\n")
        self._write(f"sisl-version=1 unit={unit}\n")

        if size is None:
            size = np.ones([3], np.int32)
        if origin is None:
            origin = lattice.origin[:]

        _fmt = "{:d} {:15.10e} {:15.10e} {:15.10e}\n"

        # Add #-of atoms and origin
        self._write(_fmt.format(1, *(origin * Ang2unit)))

        # Write the cell and voxels
        for ix in range(3):
            dcell = lattice.cell[ix, :] / size[ix] * Ang2unit
            self._write(_fmt.format(size[ix], *dcell))

        self._write("1 0. 0. 0. 0.\n")

    @sile_fh_open()
    def write_geometry(
        self,
        geometry: Geometry,
        fmt: str = "15.10e",
        size=None,
        origin=None,
        unit: str = "Bohr",
        *args,
        **kwargs,
    ):
        """Writes `Geometry` object attached to this grid

        Parameters
        ----------
        geometry :
            geometry to be written
        fmt :
            floating point format for stored values
        size : (3, ), optional
            shape of the stored grid (``[1, 1, 1]``)
        origin : (3, ), optional
            origin of the cell (``[0, 0, 0]``)
        unit:
            what length unit should the cube file data be written in
        """
        sile_raise_write(self)

        Ang2unit = unit_convert("Ang", unit)

        # Write header
        self._write("\n")
        self._write(f"sisl-version=1 unit={unit}\n")

        if size is None:
            size = np.ones([3], np.int32)
        if origin is None:
            origin = geometry.origin[:]

        _fmt = "{:d} {:15.10e} {:15.10e} {:15.10e}\n"

        valid_Z = (geometry.atoms.Z > 0).nonzero()[0]
        geometry = geometry.sub(valid_Z)

        # Add #-of atoms and origin
        self._write(_fmt.format(len(geometry), *(origin * Ang2unit)))

        # Write the cell and voxels
        for ix in range(3):
            dcell = geometry.cell[ix, :] / size[ix] * Ang2unit
            self._write(_fmt.format(size[ix], *dcell))

        tmp = " {:" + fmt + "}"
        _fmt = "{:d} 0.0" + tmp + tmp + tmp + "\n"
        for ia in geometry:
            self._write(
                _fmt.format(geometry.atoms[ia].Z, *geometry.xyz[ia, :] * Ang2unit)
            )

    @sile_fh_open()
    def write_grid(
        self,
        grid: Grid,
        fmt: str = ".5e",
        imag: bool = False,
        unit: str = "Bohr",
        *args,
        **kwargs,
    ):
        """Write `Grid` to the contained file

        Parameters
        ----------
        grid :
           the grid to be written in the CUBE file
        fmt :
           format used for precision output
        imag :
           write only imaginary part of the grid, default to only writing the
           real part.
        unit:
            what length unit should the cube file data be written in.
            The grid data is assumed to be unit-less, this unit only refers
            to the lattice vectors and atomic coordinates.
        buffersize : int, optional
           size of the buffer while writing the data, (6144)
        """
        # Check that we can write to the file
        sile_raise_write(self)

        if grid.geometry is None:
            self.write_lattice(
                grid.lattice, size=grid.shape, unit=unit, *args, **kwargs
            )
        else:
            self.write_geometry(
                grid.geometry, size=grid.shape, unit=unit, *args, **kwargs
            )

        buffersize = kwargs.get("buffersize", min(6144, grid.grid.size))
        buffersize += buffersize % 6  # ensure multiple of 6

        # A CUBE file contains grid-points aligned like this:
        # for x
        #   for y
        #     for z
        #       write...
        _fmt1 = "{:" + fmt + "} "
        _fmt6 = (_fmt1 * 6)[:-1] + "\n"
        __fmt = _fmt6 * (buffersize // 6)

        if imag:
            for z in np.nditer(
                np.asarray(grid.grid.imag, order="C").reshape(-1),
                flags=["external_loop", "buffered"],
                op_flags=[["readonly"]],
                order="C",
                buffersize=buffersize,
            ):
                if z.shape[0] != buffersize:
                    s = z.shape[0]
                    __fmt = _fmt6 * (s // 6) + _fmt1 * (s % 6) + "\n"
                self._write(__fmt.format(*z.tolist()))
        else:
            for z in np.nditer(
                np.asarray(grid.grid.real, order="C").reshape(-1),
                flags=["external_loop", "buffered"],
                op_flags=[["readonly"]],
                order="C",
                buffersize=buffersize,
            ):
                if z.shape[0] != buffersize:
                    s = z.shape[0]
                    __fmt = _fmt6 * (s // 6) + _fmt1 * (s % 6) + "\n"
                self._write(__fmt.format(*z.tolist()))

        # Add a finishing line to ensure empty ending
        self._write("\n")

    def _r_header_dict(self):
        """Reads the header of the file"""
        self.fh.seek(0)
        self.readline()
        header = header_to_dict(self.readline())
        header["unit"] = unit_convert(header.get("unit", "Bohr"), "Ang")
        return header

    def read_basis(self) -> Atoms:
        """Reads the `Atoms` object from the CUBE file"""
        return self.read_geometry().atoms

    @sile_fh_open()
    def read_lattice(self, ret_na: bool = False) -> Lattice:
        """Returns `Lattice` object from the CUBE file

        Parameters
        ----------
        ret_na : bool, optional
           whether to also return the number of atoms in the geometry

        Returns
        -------
        lattice: Lattice
            the lattice object
        na : int
            number of atoms (only if `ret_na`)
        """
        unit2Ang = self._r_header_dict()["unit"]

        origin = self.readline().split()  # origin
        na = int(origin[0])
        origin = np.fromiter(map(float, origin[1:]), np.float64)

        cell = np.empty([3, 3], np.float64)
        for i in [0, 1, 2]:
            tmp = self.readline().split()
            s = int(tmp[0])
            tmp = tmp[1:]
            for j in [0, 1, 2]:
                cell[i, j] = float(tmp[j]) * s

        cell = cell * unit2Ang
        origin = origin * unit2Ang
        if ret_na:
            return Lattice(cell, origin=origin), na
        return Lattice(cell, origin=origin)

    @sile_fh_open()
    def read_geometry(self) -> Geometry:
        """Returns `Geometry` object from the CUBE file"""
        unit2Ang = self._r_header_dict()["unit"]
        lattice, na = self.read_lattice(ret_na=True)

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

        return Geometry(xyz * unit2Ang, atom, lattice=lattice)

    @sile_fh_open()
    def read_grid(self, imag=None) -> Grid:
        """Returns `Grid` object from the CUBE file

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
            lattice = self.read_lattice()
        else:
            lattice = geom.lattice

        # read headers (and seek to start)
        self._r_header_dict()
        na = int(self.readline().split()[0])

        ngrid = [0] * 3
        for i in (0, 1, 2):
            tmp = self.readline().split()
            ngrid[i] = int(tmp[0])

        # Read past the atoms
        for i in range(na):
            self.readline()

        if geom is None:
            grid = Grid(ngrid, dtype=np.float64, lattice=lattice)
        else:
            grid = Grid(ngrid, dtype=np.float64, geometry=geom)
        grid.grid.shape = (-1,)

        # TODO check performance of this
        # We are currently doing this to enable reading
        #  1-column data and 6-column data.
        lines = [item for sublist in self.fh.readlines() for item in sublist.split()]
        grid.grid[:] = np.array(lines).astype(grid.dtype, copy=False)
        grid.grid.shape = ngrid

        if imag is None:
            return grid

        # We are expecting an imaginary part
        if not grid.geometry.equal(imag.geometry):
            raise SislError(
                f"{self!s} and its imaginary part does not have the same "
                "geometry. Hence a combined complex Grid cannot be formed."
            )
        if grid != imag:
            raise SislError(
                f"{self!s} and its imaginary part does not have the same "
                "shape. Hence a combined complex Grid cannot be formed."
            )

        # Now we have a complex grid
        grid.grid = grid.grid + 1j * imag.grid

        return grid


add_sile("cube", cubeSile, case=False, gzip=True)
