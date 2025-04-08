# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""
Sile object for reading/writing FHI-aims geometry files
"""
import numpy as np

from sisl import Geometry, Lattice
from sisl._internal import set_module
from sisl.messages import deprecate_argument

from ..sile import SileError, add_sile, sile_fh_open, sile_raise_write
from .sile import SileFHIaims

__all__ = ["inSileFHIaims"]


@set_module("sisl.io.fhiaims")
class inSileFHIaims(SileFHIaims):
    """FHI-aims ``geometry.in`` file object"""

    @sile_fh_open()
    @deprecate_argument("sc", "lattice", "use lattice= instead of sc=", "0.15", "0.17")
    def write_lattice(self, lattice: Lattice, fmt: str = ".8f"):
        """Writes the supercell to the contained file

        Parameters
        ----------
        lattice :
           the supercell to be written
        fmt :
           used format for the precision of the data
        """
        sile_raise_write(self)
        _fmt = f"lattice_vector {{:{fmt}}} {{:{fmt}}} {{:{fmt}}}\n"
        self._write(_fmt.format(*lattice.cell[0]))
        self._write(_fmt.format(*lattice.cell[1]))
        self._write(_fmt.format(*lattice.cell[2]))

    @sile_fh_open()
    def write_geometry(
        self,
        geometry: Geometry,
        fmt: str = ".8f",
        as_frac: bool = False,
        velocity=None,
        moment=None,
    ):
        """Writes the geometry to the contained file

        Parameters
        ----------
        geometry :
           the geometry to be written
        fmt :
           used format for the precision of the data
        as_frac :
           whether coordinates are written as fractional coordinates
        velocity: array_like, optional
           also write the velocity fields in [Ang/ps]
        moment : array_like, optional
           also write an initial moment for each atom
        """
        # Check that we can write to the file
        sile_raise_write(self)

        self.write_lattice(geometry.lattice, fmt)

        if as_frac:
            xyz = geometry.fxyz
            prefix = "atom_frac"
        else:
            xyz = geometry.xyz
            prefix = "atom"

        _fmt = f"{prefix} {{1:{fmt}}} {{2:{fmt}}} {{3:{fmt}}} {{0:s}}\n"
        _fmtv = f"velocity {{:{fmt}}} {{:{fmt}}} {{:{fmt}}}\n"
        _fmtm = f"initial_moment {{:{fmt}}}\n"
        for ia, atom in enumerate(geometry.atoms):
            s = {"fa": "Ds"}.get(atom.symbol, atom.symbol)
            self._write(_fmt.format(s, *xyz[ia]))
            if velocity is not None:
                self._write(_fmtv.format(*velocity[ia]))
            if moment is not None:
                self._write(_fmtm.format(moment[ia]))

    @sile_fh_open()
    def read_lattice(self) -> Lattice:
        """Reads supercell object from the file"""
        self.fh.seek(0)

        # read until "lattice_vector" is found
        cell = []
        for line in self:
            if line.startswith("lattice_vector"):
                cell.append([float(f) for f in line.split()[1:]])

        return Lattice(cell)

    @sile_fh_open()
    @deprecate_argument(
        "velocity",
        "ret_velocity",
        "use ret_velocity= instead of veloticy=",
        "0.15",
        "0.17",
    )
    @deprecate_argument(
        "moment",
        "ret_moment",
        "use ret_moment= instead of moment=",
        "0.15",
        "0.17",
    )
    def read_geometry(
        self, ret_velocity: bool = False, ret_moment: bool = False
    ) -> Geometry:
        """Reads Geometry object from the file

        Parameters
        ----------
        ret_velocity: bool, optional
           also return the velocities in the file, if not present, it will
           return a 0 array
        ret_moment: bool, optional
           also return the moments specified in the file, if not present, it will
           return a 0 array

        Returns
        -------
        geometry : Geometry
            geometry found in file
        velocity : numpy.ndarray
            array of velocities in Ang/ps for each atom, will only be returned if `ret_velocity` is true
        moment : numpy.ndarray
            array of initial moments of each atom, will only be returned if `ret_moment` is true
        """
        lattice = self.read_lattice()

        self.fh.seek(0)
        sp = []
        xyz = []
        v = []
        m = []

        def ensure_length(l, length, add):
            if length < 0:
                raise SileError(
                    "Found a velocity/initial_moment entry before an atom entry?"
                )
            while len(l) < length:
                l.append(add)

        for line in self:
            line = line.split()
            if line[0] == "atom":
                xyz.append([float(f) for f in line[1:4]])
            elif line[0] == "atom_frac":
                xyz.append([float(f) for f in line[1:4]] @ lattice.cell)
            elif line[0] == "velocity":
                # ensure xyz and v are same length
                ensure_length(v, len(xyz) - 1, [0, 0, 0])
                v.append([float(f) for f in line[1:4]])
                continue
            elif line[0] == "initial_moment":
                # ensure xyz and v are same length
                ensure_length(m, len(xyz) - 1, 0)
                m.append(float(line[1]))
                continue
            else:
                continue

            # we found an atom
            sp.append(line[4])

        ret = (Geometry(xyz, atoms=sp, lattice=lattice),)
        if not ret_velocity and not ret_moment:
            return ret[0]

        if ret_velocity:
            ret = ret + (np.array(v),)
        if ret_moment:
            ret = ret + (np.array(m),)
        return ret

    def read_velocity(self) -> np.ndarray:
        """Reads velocity in the file"""
        return self.read_geometry(ret_velocity=True)[1]

    def read_moment(self) -> np.ndarray:
        """Reads initial moment in the file"""
        return self.read_geometry(ret_moment=True)[1]

    def ArgumentParser(self, p=None, *args, **kwargs):
        """Returns the arguments that is available for this Sile"""
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geometry().ArgumentParser(p, *args, **newkw)


add_sile("geometry.in", inSileFHIaims, case=False, gzip=True)
add_sile("aims", inSileFHIaims, case=False, gzip=True)
