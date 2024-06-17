# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np

from sisl._internal import set_module

from ..sile import add_sile, sile_fh_open, sile_raise_write
from .sile import SileSiesta

__all__ = ["faSileSiesta"]


@set_module("sisl.io.siesta")
class faSileSiesta(SileSiesta):
    """Forces file"""

    @sile_fh_open()
    def read_force(self) -> np.ndarray:
        """Reads the forces from the file"""
        na = int(self.readline())

        f = np.empty([na, 3], np.float64)
        for ia in range(na):
            f[ia, :] = list(map(float, self.readline().split()[1:]))

        # Units are already eV / Ang
        return f

    @sile_fh_open()
    def write_force(self, f, fmt: str = ".9e"):
        """Write forces to file

        Parameters
        ----------
        fmt : str, optional
           precision of written forces
        """
        sile_raise_write(self)
        na = len(f)
        self._write(f"{na}\n")
        _fmt = ("{:d}" + (" {:" + fmt + "}") * 3) + "\n"

        for ia in range(na):
            self._write(_fmt.format(ia + 1, *f[ia, :]))

    # Short-cut
    read_data = read_force
    write_data = write_force


add_sile("FA", faSileSiesta, gzip=True)
add_sile("FAC", faSileSiesta, gzip=True)
add_sile("TSFA", faSileSiesta, gzip=True)
add_sile("TSFAC", faSileSiesta, gzip=True)
