from __future__ import print_function

import os.path as osp
from numbers import Integral
import numpy as np

from .sile import SileCDFSiesta
from ..sile import *

from sisl.messages import info
from sisl import SuperCell, Grid
from sisl.unit.siesta import unit_convert

__all__ = ['gridncSileSiesta']

Bohr2Ang = unit_convert('Bohr', 'Ang')
Ry2eV = unit_convert('Ry', 'eV')


class gridncSileSiesta(SileCDFSiesta):
    """ NetCDF real-space grid file """

    def read_supercell(self):
        """ Returns a SuperCell object from a Siesta.grid.nc file
        """
        cell = np.array(self._value('cell'), np.float64)
        # Yes, this is ugly, I really should implement my unit-conversion tool
        cell *= Bohr2Ang
        cell.shape = (3, 3)

        return SuperCell(cell)

    def write_supercell(self, sc):
        """ Write a supercell to the grid.nc file """
        sile_raise_write(self)

        # Create initial dimensions
        self._crt_dim(self, 'xyz', 3)
        self._crt_dim(self, 'abc', 3)

        v = self._crt_var(self, 'cell', 'f8', ('abc', 'xyz'))
        v.info = 'Unit cell'
        v.unit = 'Bohr'
        v[:, :] = sc.cell[:, :] / Bohr2Ang

    def read_grid(self, spin=0, name='gridfunc', *args, **kwargs):
        """ Reads a grid in the current Siesta.grid.nc file

        Enables the reading and processing of the grids created by Siesta

        Parameters
        ----------
        spin : int or array_like, optional
            specify the retrieved values
        name : str, optional
            the name for the grid-function (do not supply for standard Siesta output)
        """
        # Determine the name of this file
        f = osp.basename(self.file)

        # File names are made up of
        #  ElectrostaticPotential.grid.nc
        # So the first one should be ElectrostaticPotential
        base = f.split('.')[0]

        # Unit-conversion
        BohrC2AngC = Bohr2Ang ** 3

        unit = {'Rho': 1. / BohrC2AngC,
                'DeltaRho': 1. / BohrC2AngC,
                'RhoXC': 1. / BohrC2AngC,
                'RhoInit': 1. / BohrC2AngC,
                'Chlocal': 1. / BohrC2AngC,
                'TotalCharge': 1. / BohrC2AngC,
                'BaderCharge': 1. / BohrC2AngC,
                'ElectrostaticPotential': Ry2eV,
                'TotalPotential': Ry2eV,
                'Vna': Ry2eV,
        }.get(base, None)

        # Fall-back
        if unit is None:
            unit = 1.
            show_info = True
        else:
            show_info = False

        # Swap as we swap back in the end
        sc = self.read_supercell().swapaxes(0, 2)

        # Create the grid
        nx = len(self._dimension('n1'))
        ny = len(self._dimension('n2'))
        nz = len(self._dimension('n3'))

        if name is None:
            v = self._variable('gridfunc')
        else:
            v = self._variable(name)

        # Create the grid, Siesta uses periodic, always
        grid = Grid([nz, ny, nx], bc=Grid.PERIODIC, sc=sc, dtype=v.dtype)

        if v.ndim == 3:
            grid.grid[:, :, :] = v[:, :, :] * unit
        elif isinstance(spin, Integral):
            grid.grid[:, :, :] = v[spin, :, :, :] * unit
        else:
            if len(spin) > v.shape[0]:
                raise SileError(self.__class__.__name__ + '.read_grid requires spin to be an integer or '
                                'an array of length equal to the number of spin components.')
            grid.grid[:, :, :] = v[0, :, :, :] * spin[0] * unit
            for i, scale in enumerate(spin[1:]):
                grid.grid[:, :, :] += v[1+i, :, :, :] * scale * unit
        if show_info:
            info(self.__class__.__name__ + '.read_grid cannot determine the units of the grid. '
                 'The units may not be in sisl units.')

        # Read the grid, we want the z-axis to be the fastest
        # looping direction, hence x,y,z == 0,1,2
        return grid.swapaxes(0, 2)

    def write_grid(self, grid, spin=0, nspin=None):
        """ Write a grid to the grid.nc file """
        sile_raise_write(self)

        self.write_supercell(grid.sc)

        if nspin is not None:
            self._crt_dim(self, 'spin', nspin)

        self._crt_dim(self, 'n1', grid.shape[0])
        self._crt_dim(self, 'n2', grid.shape[1])
        self._crt_dim(self, 'n3', grid.shape[2])

        if nspin is None:
            v = self._crt_var(self, 'gridfunc', grid.dtype, ('n3', 'n2', 'n1'))
        else:
            v = self._crt_var(self, 'gridfunc', grid.dtype, ('spin', 'n3', 'n2', 'n1'))
        v.info = 'Grid function'

        if nspin is None:
            v[:, :, :] = np.swapaxes(grid.grid, 0, 2)
        else:
            v[spin, :, :, :] = np.swapaxes(grid.grid, 0, 2)


add_sile('grid.nc', gridncSileSiesta)
