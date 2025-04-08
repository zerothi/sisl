# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Import sile objects
from __future__ import annotations

from sisl import Geometry
from sisl._internal import set_module
from sisl.messages import deprecate_argument

from .sile import *

__all__ = ["moldenSile"]


@set_module("sisl.io")
class moldenSile(Sile):
    """Molden file object"""

    @sile_fh_open()
    @deprecate_argument("sc", "lattice", "use lattice= instead of sc=", "0.15", "0.17")
    def write_lattice(self, lattice):
        """Writes the supercell to the contained file"""
        # Check that we can write to the file
        sile_raise_write(self)

        # Write the number of atoms in the geometry
        self._write("[Molden Format]\n")

        # Sadly, MOLDEN does not read this information...

    @sile_fh_open()
    def write_geometry(self, geometry, fmt=".8f"):
        """Writes the geometry to the contained file"""
        # Check that we can write to the file
        sile_raise_write(self)

        # Be sure to write the supercell
        self.write_lattice(geometry.lattice)

        # Write in ATOM mode
        self._write("[Atoms] Angs\n")

        # Write out the cell information in the comment field
        # This contains the cell vectors in a single vector (3 + 3 + 3)
        # quantities, plus the number of supercells (3 ints)

        fmt_str = (
            "{{0:2s}} {{1:4d}} {{2:4d}}  {{3:{0}}}  {{4:{0}}}  {{5:{0}}}\n".format(fmt)
        )
        for ia, a, _ in geometry.iter_species():
            self._write(fmt_str.format(a.symbol, ia, a.Z, *geometry.xyz[ia, :]))

    def ArgumentParser(self, p=None, *args, **kwargs):
        """Returns the arguments that is available for this Sile"""
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geometry().ArgumentParser(p, *args, **newkw)


add_sile("molf", moldenSile, case=False, gzip=True)
