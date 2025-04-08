# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Optional

import numpy as np

from sisl._internal import set_module
from sisl.messages import deprecation, warn
from sisl.unit.siesta import unit_convert

from ..sile import add_sile, sile_fh_open
from .sile import SileSiesta

__all__ = ["fcSileSiesta"]


@set_module("sisl.io.siesta")
class fcSileSiesta(SileSiesta):
    """Force constant file"""

    @sile_fh_open()
    def read_force(
        self, displacement: Optional[float] = None, na: Optional[int] = None
    ) -> np.ndarray:
        """Reads all displacement forces by multiplying with the displacement value

        Since the force constant file does not contain the non-displaced configuration
        this will only return forces on the displaced configurations minus the forces from
        the non-displaced configuration.

        This may be used in conjunction with phonopy by noticing that Siesta FC-runs does
        the displacements in reverse order (-x/+x vs. +x/-x). In this case one should reorder
        the elements like this:

        >>> fc = np.roll(fc, 1, axis=2)

        Parameters
        ----------
        displacement :
           the used displacement in the calculation, since Siesta 4.1-b4 this value
           is written in the FC file and hence not required.
           If prior Siesta versions are used and this is not supplied the 0.04 Bohr displacement
           will be assumed.
        na :
           number of atoms in geometry (for returning correct number of atoms), since Siesta 4.1-b4
           this value is written in the FC file and hence not required.
           If prior Siesta versions are used then the file is expected to only contain 1-atom displacement.

        Returns
        -------
        numpy.ndarray : (displaced atoms, d[xyz], [-+], total atoms, xyz)
             force constant matrix times the displacement, see `read_force_constant` for details regarding
             data layout.
        """
        if displacement is None:
            line = self.readline().split()
            self.fh.seek(0)
            try:
                displacement = float(line[-1])
            except Exception:
                warn(
                    f"{self.__class__.__name__}.read_force assumes displacement=0.04 Bohr!"
                )
                displacement = 0.04 * unit_convert("Bohr", "Ang")

        # Since the displacements changes sign (starting with a negative sign)
        # we can convert using this scheme
        displacement = np.repeat(displacement, 6).ravel()
        displacement[1::2] *= -1
        return self.read_hessian(na) * displacement.reshape(1, 3, 2, 1, 1)

    @sile_fh_open()
    def read_hessian(self, na: Optional[int] = None):
        """Reads the Hessian/force constant stored in the FC file

        Parameters
        ----------
        na :
           number of atoms in the unit-cell, if not specified it will guess on only
           one atom displacement.

        Returns
        -------
        numpy.ndarray : (displacement, d[xyz], [-+], atoms, xyz)
             force constant matrix containing all forces. The 2nd dimension contains
             contains the directions, 3rd dimension contains -/+ displacements.
        """
        # Force constants matrix
        line = self.readline().split()
        if na is None:
            try:
                na = int(line[-2])
            except Exception:
                na = None

        fc = list()
        while True:
            line = self.readline()
            if line == "":
                # empty line or nothing
                break
            fc.append(list(map(float, line.split())))

        # Units are already eV / Ang ** 2
        fc = np.array(fc)

        # Slice to correct size
        if na is None:
            na = fc.size // 6 // 3

        # Correct shape of matrix
        fc.shape = (-1, 3, 2, na, 3)

        return fc

    read_force_constant = deprecation(
        "read_force_constant is deprecated in favor of read_hessian", "0.15", "0.17"
    )(read_hessian)


add_sile("FC", fcSileSiesta, gzip=True)
add_sile("FCC", fcSileSiesta, gzip=True)
