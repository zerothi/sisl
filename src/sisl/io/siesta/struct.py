# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np

from sisl import Atom, AtomGhost, Atoms, AtomUnknown, Geometry, Lattice
from sisl._internal import set_module
from sisl.messages import deprecate_argument
from sisl.unit.siesta import unit_convert

from ..sile import add_sile, sile_fh_open, sile_raise_write
from .sile import SileSiesta

__all__ = ["structSileSiesta"]


Bohr2Ang = unit_convert("Bohr", "Ang")


@set_module("sisl.io.siesta")
class structSileSiesta(SileSiesta):
    """Geometry file"""

    @sile_fh_open()
    def write_geometry(self, geometry: Geometry, fmt: str = ".9f"):
        """Writes the geometry to the contained file

        Parameters
        ----------
        geometry :
           geometry to write in the STRUCT file
        fmt :
           the precision used for writing the coordinates in the file
        """
        # Check that we can write to the file
        sile_raise_write(self)

        # Create format string for the cell-parameters
        fmt_str = "   " + ("{:" + fmt + "} ") * 3 + "\n"
        for i in range(3):
            self._write(fmt_str.format(*geometry.cell[i]))
        self._write(f"{geometry.na:12d}\n")

        # Create format string for the atomic coordinates
        fxyz = geometry.fxyz
        fmt_str = "{:3d}{:6d} "
        fmt_str += ("{:" + fmt + "} ") * 3 + "\n"
        for ia, a, ips in geometry.iter_species():
            if isinstance(a, AtomGhost):
                self._write(fmt_str.format(ips + 1, -a.Z, *fxyz[ia]))
            else:
                self._write(fmt_str.format(ips + 1, a.Z, *fxyz[ia]))

    @sile_fh_open()
    def read_lattice(self) -> Lattice:
        """Returns `Lattice` object from the STRUCT file"""

        cell = np.empty([3, 3], np.float64)
        for i in range(3):
            cell[i, :] = list(map(float, self.readline().split()[:3]))

        return Lattice(cell)

    @sile_fh_open()
    @deprecate_argument(
        "species_Z",
        "species_as_Z",
        "use species_as_Z= instead of species_Z=",
        "0.15",
        "0.16",
    )
    def read_geometry(self, species_as_Z: bool = False) -> Geometry:
        """Returns a `Geometry` object from the ``STRUCT`` file

        Parameters
        ----------
        species_as_Z : bool, optional
           if ``True`` the atomic numbers are the species indices (useful when
           reading the ChemicalSpeciesLabel block simultaneously).

        Returns
        -------
        Geometry
        """
        lattice = self.read_lattice()

        # Read number of atoms
        na = int(self.readline())
        xyz = np.empty([na, 3], np.float64)
        atms = [None] * na
        sp = np.empty([na], np.int32)
        for ia in range(na):
            line = self.readline().split()
            sp[ia] = int(line[0])
            Z = int(line[1])

            if species_as_Z:
                atms[ia] = Atom(sp[ia])
            else:
                atms[ia] = Atom(Z)

            xyz[ia, :] = line[2:5]

        xyz = xyz @ lattice.cell

        # Ensure correct sorting
        max_s = sp.max()
        sp -= 1
        # Ensure we can remove the atom after having aligned them
        atms2 = Atoms(AtomUnknown(1000), na=na)
        for i in range(max_s):
            idx = (sp[:] == i).nonzero()[0]
            if len(idx) == 0:
                # Always ensure we have "something" for the unoccupied places
                atms2[idx] = AtomUnknown(1000 + i)
            else:
                atms2[idx] = atms[idx[0]]

        return Geometry(xyz, atms2.reduce(), lattice=lattice)

    def ArgumentParser(self, p=None, *args, **kwargs):
        """Returns the arguments that is available for this Sile"""
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geometry().ArgumentParser(p, *args, **newkw)


add_sile("STRUCT_IN", structSileSiesta, gzip=True)
add_sile("STRUCT_NEXT_ITER", structSileSiesta, gzip=True)
add_sile("STRUCT_OUT", structSileSiesta, gzip=True)
