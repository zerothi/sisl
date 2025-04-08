# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np

from sisl import Atom, AtomGhost, Atoms, Geometry, Lattice
from sisl._internal import set_module
from sisl.messages import deprecate_argument
from sisl.unit.siesta import unit_convert

from .._help import _fill_basis_empty
from ..sile import SileError, add_sile, sile_fh_open, sile_raise_write
from .sile import SileSiesta

__all__ = ["xvSileSiesta"]


Bohr2Ang = unit_convert("Bohr", "Ang")


@set_module("sisl.io.siesta")
class xvSileSiesta(SileSiesta):
    """Geometry file"""

    @sile_fh_open()
    def write_geometry(self, geometry: Geometry, fmt: str = ".9f", velocity=None):
        """Writes the geometry to the contained file

        Parameters
        ----------
        geometry :
           geometry to write in the XV file
        fmt :
           the precision used for writing the XV file
        velocity : numpy.ndarray, optional
           velocities to write in the XV file (will be zero if not specified).
           Units input must be in Ang/fs.
        """
        # Check that we can write to the file
        sile_raise_write(self)

        if velocity is None:
            velocity = np.zeros([geometry.na, 3], np.float32)
        if geometry.xyz.shape != velocity.shape:
            raise SileError(
                f"{self}.write_geometry requires the input"
                "velocity to have equal length to the input geometry."
            )

        # Write unit-cell
        tmp = np.zeros(6, np.float64)

        # Create format string for the cell-parameters
        fmt_str = ("   " + ("{:" + fmt + "} ") * 3) * 2 + "\n"
        for i in range(3):
            tmp[0:3] = geometry.cell[i, :] / Bohr2Ang
            self._write(fmt_str.format(*tmp))
        self._write(f"{geometry.na:12d}\n")

        # Create format string for the atomic coordinates
        fmt_str = "{:3d}{:6d} "
        fmt_str += ("{:" + fmt + "} ") * 3 + "   "
        fmt_str += ("{:" + fmt + "} ") * 3 + "\n"
        for ia, a, ips in geometry.iter_species():
            tmp[0:3] = geometry.xyz[ia, :] / Bohr2Ang
            tmp[3:] = velocity[ia, :] / Bohr2Ang
            if isinstance(a, AtomGhost):
                self._write(fmt_str.format(ips + 1, -a.Z, *tmp))
            else:
                self._write(fmt_str.format(ips + 1, a.Z, *tmp))

    @sile_fh_open()
    def read_lattice(self) -> Lattice:
        """Returns `Lattice` object from the XV file"""

        cell = np.empty([3, 3], np.float64)
        for i in range(3):
            cell[i, :] = list(map(float, self.readline().split()[:3]))
        cell *= Bohr2Ang

        return Lattice(cell)

    @sile_fh_open()
    @deprecate_argument(
        "velocity",
        "ret_velocity",
        "use ret_velocity= instead of velocity=",
        "0.15",
        "0.17",
    )
    @deprecate_argument(
        "species_Z",
        "species_as_Z",
        "species_ arguments (species_Z and species_as_Z) are deprecated in favor of atoms=",
        "0.15",
        "0.17",
    )
    @deprecate_argument(
        "species_as_Z",
        None,
        "species_as_Z= is deprecated, please pass an Atoms object with the basis information as atoms=",
        "0.15",
        "0.17",
    )
    def read_geometry(
        self,
        ret_velocity: bool = False,
        atoms: Optional[Atoms, Geometry] = None,
        species_as_Z: bool = False,
    ) -> Geometry:
        """Returns a `Geometry` object from the XV file

        Parameters
        ----------
        ret_velocity :
            also return the velocities in the file
        atoms :
            an object containing the basis information, is useful to overwrite
            the atoms object contained in the geometry.
        species_as_Z :
            Deprecated, it does nothing!

        Returns
        -------
        geometry: Geometry
            the geometry in the XV file
        velocity: numpy.ndarray
            only if `ret_velocity` is true.
        """
        lattice = self.read_lattice()

        # Read number of atoms
        na = int(self.readline())
        xyz = np.empty([na, 3], np.float64)
        vel = np.empty([na, 3], np.float64)
        atms = [None] * na
        sp = np.empty([na], np.int32)
        for ia in range(na):
            line = self.readline().split()
            sp[ia] = int(line[0])
            Z = int(line[1])

            atms[ia] = Atom(Z)

            xyz[ia, :] = line[2:5]
            vel[ia, :] = line[5:8]

        xyz *= Bohr2Ang
        vel *= Bohr2Ang

        # Ensure correct sorting
        # This is important when users creates geometries with
        # basis information for atoms that are not present.
        # E.g. a graphene flake with a H basis information as the first species

        if atoms is None:
            atoms = atms
        # ensure correct sorting
        atms2 = _fill_basis_empty(sp - 1, atoms)

        geom = Geometry(xyz, atms2, lattice=lattice)
        if ret_velocity:
            return geom, vel
        return geom

    @sile_fh_open()
    def read_velocity(self) -> np.ndarray:
        """Returns an array with the velocities from the XV file

        Returns
        -------
        numpy.ndarray
        """
        self.read_lattice()
        na = int(self.readline())
        vel = np.empty([na, 3], np.float64)
        for ia in range(na):
            line = list(map(float, self.readline().split()[:8]))
            vel[ia, :] = line[5:8]

        vel *= Bohr2Ang
        return vel

    read_data = read_velocity

    def ArgumentParser(self, p=None, *args, **kwargs):
        """Returns the arguments that is available for this Sile"""
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geometry().ArgumentParser(p, *args, **newkw)


add_sile("XV", xvSileSiesta, gzip=True)
