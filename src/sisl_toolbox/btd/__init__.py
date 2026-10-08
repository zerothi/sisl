# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Block-tri-diagonal Green function calculations for any number of electrodes

Developer: Nick Papior
Contact: nickpapior <at> gmail.com
sisl-version: >=0.11.0
tbtrans-version: >=siesta-4.1.5

Calculates Green function quantities of N-terminal devices using the
block-tri-diagonal (BTD) structure of the device matrices.
This gives access to quantities not currently available in TBtrans, e.g.
spectral functions, scattering states, scattering matrices and eigenchannels.

The toolbox contains:

- `DeviceGreen`: the BTD Green function calculator of the device region,
  typically created from TBtrans input/output via `DeviceGreen.from_fdf`
- `PivotSelfEnergy` and `DownfoldSelfEnergy`: electrode self-energies placed,
  or downfolded, into the device region
- `BlockMatrix`: container for block matrices
- `btd_partition` and `PermuteBTD`: pivoting tables and block sizes for
  BTD matrices, see also the `~sisl_toolbox.btd.pivot` module
"""

from __future__ import annotations

from ._btd import *
from ._electrode import *
from ._green import *
