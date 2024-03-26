# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""
Sile object for reading/writing ascii files from BigDFT
"""

import numpy as np

from sisl import Atom, Geometry, Lattice
from sisl._internal import set_module
from sisl.unit import unit_convert

from ..sile import *
from .sile import SileBigDFT

__all__ = ["asciiSileBigDFT"]


Bohr2Ang = unit_convert("Bohr", "Ang")


@set_module("sisl.io.bigdft")
class asciiSileBigDFT(SileBigDFT):
    """ASCII file object for BigDFT"""

    def _setup(self, *args, **kwargs):
        """Initialize for `asciiSileBigDFT`"""
        super()._setup(*args, **kwargs)
        self._comment = ["#", "!"]

    @sile_fh_open()
    def read_geometry(self) -> Geometry:
        """Reads a supercell from the Sile"""

        # 1st line is arbitrary
        self.readline(True)
        # Read dxx, dyx, dyy
        dxx, dyx, dyy = map(float, self.readline().split()[:3])
        # Read dzx, dzy, dzz
        dzx, dzy, dzz = map(float, self.readline().split()[:3])

        # options for the ASCII format
        is_frac = False
        is_angdeg = False
        is_bohr = False

        xyz = []
        spec = []

        # Now we need to read through and find keywords
        try:
            while True:
                # Read line also with comment
                l = self.readline(True)

                # Empty line, means EOF
                if l == "":
                    break
                # too short, continue
                if len(l) < 1:
                    continue

                # Check for keyword
                if l[1:].startswith("keyword:"):
                    if "reduced" in l:
                        is_frac = True
                    if "angdeg" in l:
                        is_angdeg = True
                    if "bohr" in l or "atomic" in l:
                        is_bohr = True
                    continue

                elif l[0] in self._comment:
                    # this is a comment, cycle
                    continue

                # Read atomic coordinates
                ls = l.split()
                if len(ls) < 3:
                    continue

                # The first three are the coordinates
                xyz.append([float(x) for x in ls[:3]])
                # The 4th is the specie, [5th is tag]
                s = ls[3]
                t = s
                if len(ls) > 4:
                    t = ls[4]
                spec.append(Atom(s, tag=t))

        except OSError as e:
            print(f"I/O error({e.errno}): {e.strerror}")
        except Exception:
            # Allowed pass due to pythonic reading
            pass

        if is_bohr:
            dxx *= Bohr2Ang
            dyx *= Bohr2Ang
            dyy *= Bohr2Ang
            if not is_angdeg:
                dzx *= Bohr2Ang
                dzy *= Bohr2Ang
                dzz *= Bohr2Ang

        # Create the supercell
        if is_angdeg:
            # The input is in skewed axis
            lattice = Lattice([dxx, dyx, dyy, dzx, dzy, dzz])
        else:
            lattice = Lattice([[dxx, 0.0, 0.0], [dyx, dyy, 0.0], [dzx, dzy, dzz]])

        # Now create the geometry
        xyz = np.array(xyz, np.float64)

        if is_frac:
            # Transform from fractional to actual
            # coordinates
            xyz = np.dot(xyz, lattice.cell.T)

        elif is_bohr:
            # Not when fractional coordinates are used
            # the supercell conversion takes care of
            # correct unit
            xyz *= Bohr2Ang

        return Geometry(xyz, spec, lattice=lattice)

    @sile_fh_open()
    def write_geometry(self, geometry: Geometry, fmt: str = ".8f"):
        """Writes the geometry to the contained file"""
        # Check that we can write to the file
        sile_raise_write(self)

        # Write out the cell
        self._write("# Created by sisl\n")
        # We write the cell coordinates as the cell coordinates
        fmt_str = f"{{:{fmt}}} " * 3 + "\n"
        self._write(
            fmt_str.format(
                geometry.cell[0, 0], geometry.cell[1, 0], geometry.cell[1, 1]
            )
        )
        self._write(fmt_str.format(*geometry.cell[2, :]))

        # This also denotes
        self._write("#keyword: angstroem\n")

        self._write("# Geometry containing: " + str(len(geometry)) + " atoms\n")

        f1_str = "{{1:{0}}}  {{2:{0}}}  {{3:{0}}} {{0:2s}}\n".format(fmt)
        f2_str = "{{2:{0}}}  {{3:{0}}}  {{4:{0}}} {{0:2s}} {{1:s}}\n".format(fmt)

        for ia, a, _ in geometry.iter_species():
            if a.symbol != a.tag:
                self._write(f2_str.format(a.symbol, a.tag, *geometry.xyz[ia, :]))
            else:
                self._write(f1_str.format(a.symbol, *geometry.xyz[ia, :]))
        # Add a single new line
        self._write("\n")

    def ArgumentParser(self, p=None, *args, **kwargs):
        """Returns the arguments that is available for this Sile"""
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geometry().ArgumentParser(p, *args, **newkw)


add_sile("ascii", asciiSileBigDFT, case=False, gzip=True)
