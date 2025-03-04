# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np

import sisl._array as _a
from sisl import Atom, Geometry, Lattice, PeriodicTable
from sisl._internal import set_module
from sisl.messages import warn

from ..sile import add_sile, sile_fh_open, sile_raise_write

# Import sile objects
from .sile import SileVASP

__all__ = ["carSileVASP"]


def _search_name(info, instance, line):
    """Names for CAR files are always in the first line"""
    return instance._line == 1


@set_module("sisl.io.vasp")
class carSileVASP(SileVASP):
    """CAR VASP files for defining geometries

    This file-object handles both POSCAR and CONTCAR files
    """

    _info_attributes_ = [
        dict(
            name="name",
            searcher=_search_name,
            default="<unknown>",
            not_found="error",
        ),
    ]

    def _setup(self, *args, **kwargs):
        """Setup the `carSile` after initialization"""
        super()._setup(*args, **kwargs)
        self._scale = 1.0

    @sile_fh_open()
    def write_geometry(
        self, geometry: Geometry, dynamic=True, group_species: bool = False
    ):
        r"""Writes the geometry to the contained file

        Parameters
        ----------
        geometry :
           geometry to be written to the file
        dynamic : None, bool or list, optional
           define which atoms are dynamic in the VASP run (default is True,
           which means all atoms are dynamic).
           If None, the resulting file will not contain any dynamic flags
        group_species:
           before writing `geometry` first re-order species to
           have species in consecutive blocks (see `geometry_group`)

        Examples
        --------
        >>> car = carSileVASP('POSCAR', 'w')
        >>> geom = geom.graphene()
        >>> geom.write(car) # regular car without Selective Dynamics
        >>> geom.write(car, dynamic=False) # fix all atoms
        >>> geom.write(car, dynamic=[False, (True, False, True)]) # fix 1st and y coordinate of 2nd

        See Also
        --------
        geometry_group: method used to group atoms together according to their species
        """
        # Check that we can write to the file
        sile_raise_write(self)

        if group_species:
            geometry, idx = self.geometry_group(geometry, ret_index=True)
        else:
            # small hack to allow dynamic
            idx = _a.arangei(len(geometry))

        # LABEL
        self._write("sisl output\n")

        # Scale
        self._write("  1.\n")

        # Write unit-cell
        fmt = ("   " + "{:18.9f}" * 3) + "\n"
        for i in range(3):
            self._write(fmt.format(*geometry.cell[i]))

        # Figure out how many species
        pt = PeriodicTable()
        s, d = [], []
        ia = 0
        while ia < geometry.na:
            atom = geometry.atoms[ia]
            # species = geometry.atoms.species[ia]
            ia_end = (np.diff(geometry.atoms.species[ia:]) != 0).nonzero()[0]
            if len(ia_end) == 0:
                # remaining atoms
                ia_end = geometry.na
            else:
                ia_end = ia + ia_end[0] + 1
            s.append(pt.Z_label(atom.Z))
            d.append(ia_end - ia)
            ia += d[-1]

        fmt = " {:s}" * len(d) + "\n"
        self._write(fmt.format(*s))
        fmt = " {:d}" * len(d) + "\n"
        self._write(fmt.format(*d))
        if dynamic is None:
            # We write in direct mode
            dynamic = [None] * len(geometry)

            def todyn(fix):
                return "\n"

        else:
            self._write("Selective dynamics\n")
            b2s = {True: "T", False: "F"}

            def todyn(fix):
                if isinstance(fix, bool):
                    return " {0} {0} {0}\n".format(b2s[fix])
                return " {} {} {}\n".format(b2s[fix[0]], b2s[fix[1]], b2s[fix[2]])

        self._write("Cartesian\n")

        if isinstance(dynamic, bool):
            dynamic = [dynamic] * len(geometry)

        fmt = "{:18.9f}" * 3
        for ia in geometry:
            self._write(fmt.format(*geometry.xyz[ia, :]) + todyn(dynamic[idx[ia]]))

    @sile_fh_open(True)
    def read_lattice(self) -> Lattice:
        """Returns `Lattice` object from this Sile"""

        # read first line
        self.readline()  # LABEL
        # Update scale-factor
        self._scale = float(self.readline())

        # Read cell vectors
        cell = np.empty([3, 3], np.float64)
        for i in range(3):
            cell[i] = list(map(float, self.readline().split()[:3]))
        cell *= self._scale

        return Lattice(cell)

    @sile_fh_open()
    def read_geometry(self, ret_dynamic: bool = False) -> Geometry:
        r"""Returns Geometry object from this Sile

        Possibly also return the dynamics (if present).

        Parameters
        ----------
        ret_dynamic : bool, optional
           also return selective dynamics (if present), if not, None will
           be returned.

        Returns
        -------
        Geometry
            the contained geometry
        numpy.ndarray
            which Cartesian directions are allowed to move (only if `ret_dynamic`)
        """
        lattice = self.read_lattice()

        # The species labels are not always included in *CAR
        line1 = self.readline().split()
        opt = self.readline().split()
        try:
            species = line1
            species_count = np.array(opt, np.int32)
        except Exception:
            species_count = np.array(line1, np.int32)
            # We have no species...
            # We default to consecutive elements in the
            # periodic table.
            species = [i + 1 for i in range(len(species_count))]
            err = "\n".join(
                [
                    "POSCAR best format:",
                    "  <Specie-1> <Specie-2>",
                    "  <#Specie-1> <#Specie-2>",
                    "Format not found, the species are defaulted to the first elements of the periodic table.",
                ]
            )
            warn(err)

        # Create list of atoms to be used subsequently
        atom = [
            Atom[spec] for spec, nsp in zip(species, species_count) for i in range(nsp)
        ]

        # Number of atoms
        na = len(atom)

        # check whether this is Selective Dynamics
        opt = self.readline()
        if opt[0] in "Ss":
            dynamics = True
            # pre-create the dynamic list
            dynamic = np.empty([na, 3], dtype=np.bool_)
            opt = self.readline()
        else:
            dynamics = False
            dynamic = None

        # Check whether this is in fractional or direct
        # coordinates (Direct == fractional)
        cart = False
        if opt[0] in "CcKk":
            cart = True

        xyz = _a.emptyd([na, 3])
        for ia in range(na):
            line = self.readline().split()
            xyz[ia, :] = list(map(float, line[:3]))
            if dynamics:
                dynamic[ia] = list(map(lambda x: x.lower() == "t", line[3:6]))

        if cart:
            # The unit of the coordinates are cartesian
            xyz *= self._scale
        else:
            xyz = xyz.dot(lattice.cell)

        # The POT/CONT-CAR does not contain information on the atomic species
        geom = Geometry(xyz=xyz, atoms=atom, lattice=lattice)
        if ret_dynamic:
            return geom, dynamic
        return geom

    def ArgumentParser(self, p=None, *args, **kwargs):
        """Returns the arguments that is available for this Sile"""
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geometry().ArgumentParser(p, *args, **newkw)


add_sile("CAR", carSileVASP, gzip=True)
add_sile("POSCAR", carSileVASP, gzip=True)
add_sile("CONTCAR", carSileVASP, gzip=True)
