# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import os.path as osp
from numbers import Integral
from typing import Optional

import numpy as np

from sisl import Grid, Lattice
from sisl._internal import set_module
from sisl.messages import deprecate_argument, info
from sisl.unit.siesta import unit_convert

from .._help import grid_reduce_indices
from ..sile import add_sile, sile_raise_write
from .sile import SileCDFSiesta

__all__ = ["gridncSileSiesta"]


Bohr2Ang = unit_convert("Bohr", "Ang")
Ry2eV = unit_convert("Ry", "eV")


@set_module("sisl.io.siesta")
class gridncSileSiesta(SileCDFSiesta):
    """NetCDF real-space grid file

    The grid sile will automatically convert the units from Siesta units (Bohr, Ry) to sisl units (Ang, eV) provided the correct extension is present.
    """

    def read_lattice(self) -> Lattice:
        """Returns a Lattice object from a Siesta.grid.nc file"""
        cell = np.array(self._value("cell"), np.float64)
        # Yes, this is ugly, I really should implement my unit-conversion tool
        cell *= Bohr2Ang
        cell.shape = (3, 3)

        return Lattice(cell)

    @deprecate_argument("sc", "lattice", "use lattice= instead of sc=", "0.15", "0.17")
    def write_lattice(self, lattice: Lattice) -> None:
        """Write a supercell to the grid.nc file"""
        sile_raise_write(self)

        # Create initial dimensions
        self._crt_dim(self, "xyz", 3)
        self._crt_dim(self, "abc", 3)

        v = self._crt_var(self, "cell", "f8", ("abc", "xyz"))
        v.info = "Unit cell"
        v.unit = "Bohr"
        v[:, :] = lattice.cell[:, :] / Bohr2Ang

    def read_grid(self, index=0, name: str = "gridfunc", **kwargs) -> Grid:
        """Reads a grid in the current Siesta.grid.nc file

        Enables the reading and processing of the grids created by Siesta

        Parameters
        ----------
        index : int or array_like, optional
           the spin-index for retrieving one of the components. If a vector
           is passed it refers to the fraction per indexed component. I.e.
           ``[0.5, 0.5]`` will return sum of half the first two components.
           Default to the first component.
        name :
            the name for the grid-function (do not supply for standard Siesta output)
        geometry: Geometry, optional
            add the Geometry to the Grid
        spin : optional
           same as `index` argument. `spin` argument has precedence.
        """
        # Default to *index* variable
        index = kwargs.get("spin", index)
        # Determine the name of this file
        f = osp.basename(self.file)

        # File names are made up of
        #  ElectrostaticPotential.grid.nc
        # So the first one should be ElectrostaticPotential
        try:
            # <>.grid.nc
            base = f.split(".")[-3]
        except Exception:
            base = "None"

        # Unit-conversion
        BohrC2AngC = Bohr2Ang**3

        unit = {
            "Rho": 1.0 / BohrC2AngC,
            "DeltaRho": 1.0 / BohrC2AngC,
            "RhoXC": 1.0 / BohrC2AngC,
            "RhoInit": 1.0 / BohrC2AngC,
            "Chlocal": 1.0 / BohrC2AngC,
            "TotalCharge": 1.0 / BohrC2AngC,
            "BaderCharge": 1.0 / BohrC2AngC,
            "ElectrostaticPotential": Ry2eV,
            "TotalPotential": Ry2eV,
            "Vna": Ry2eV,
        }.get(base, None)

        # Fall-back
        if unit is None:
            unit = 1.0
            show_info = True
        else:
            show_info = False

        # Swap as we swap back in the end
        lattice = self.read_lattice().swapaxes(0, 2)

        # Create the grid
        nx = len(self._dimension("n1"))
        ny = len(self._dimension("n2"))
        nz = len(self._dimension("n3"))

        if name is None:
            raise ValueError(
                f"{self.__class__.__name__}.read_grid does not allow 'name=None'"
            )
        else:
            v = self._variable(name)

        # Create the grid, Siesta uses periodic, always
        lattice.set_boundary_condition(Grid.PERIODIC)
        grid = Grid(
            [nz, ny, nx],
            lattice=lattice,
            dtype=v.dtype,
            geometry=kwargs.get("geometry", None),
        )

        if v.ndim == 3:
            grid.grid[:, :, :] = v[:, :, :] * unit
        elif isinstance(index, Integral):
            grid.grid[:, :, :] = v[index, :, :, :] * unit
        else:
            grid_reduce_indices(v, np.array(index) * unit, axis=0, out=grid.grid)

        if show_info:
            info(
                f"{self.__class__.__name__}.read_grid cannot determine the units of the grid. "
                "The units may not be in sisl units."
            )

        # Read the grid, we want the z-axis to be the fastest
        # looping direction, hence x,y,z == 0,1,2
        return grid.swapaxes(0, 2)

    def write_grid(
        self, grid: Grid, spin: int = 0, nspin: Optional[int] = None, **kwargs
    ) -> None:
        """Write a grid to the grid.nc file

        Parameters
        ----------
        grid :
            the grid to write to NetCDF file.
        spin :
            integer index for the spin component of the written grid function.
        nspin :
            size of the spin-dimension, each `spin` index requires a separate
            `write_grid` call.
        name : str
            the name of the variable in the NetCDF file.
            Defaults to ``gridfunc``.
        info : str
            Information written to the variable attribute ``info``.
            Defaults to ``Grid function``.
        unit : str
            A unit specifier added to the attributes of the variable in the
            NetCDF file.
        """
        sile_raise_write(self)

        # Default to *index* variable
        spin = kwargs.get("index", spin)

        self.write_lattice(grid.lattice)

        if nspin is not None:
            self._crt_dim(self, "spin", nspin)

        self._crt_dim(self, "n1", grid.shape[0])
        self._crt_dim(self, "n2", grid.shape[1])
        self._crt_dim(self, "n3", grid.shape[2])

        name = kwargs.get("name", "gridfunc")

        shape = ("n3", "n2", "n1")
        if nspin is not None:
            shape = ("spin",) + shape

        v = self._crt_var(self, name, grid.dtype, shape)
        v.info = kwargs.get("info", "Grid function")
        if "unit" in kwargs:
            v.unit = kwargs["unit"]

        if nspin is None:
            v[:, :, :] = np.swapaxes(grid.grid, 0, 2)
        else:
            v[spin, :, :, :] = np.swapaxes(grid.grid, 0, 2)


add_sile("grid.nc", gridncSileSiesta)
