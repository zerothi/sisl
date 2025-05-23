# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Eigenchannel calculator for any number of electrodes

Developer: Nick Papior
Contact: nickpapior <at> gmail.com
sisl-version: >=0.11.0
tbtrans-version: >=siesta-4.1.5

This eigenchannel calculater uses TBtrans output to calculate the eigenchannels
for N-terminal systems. In the future this will get transferred to the TBtrans code
but for now this may be used for arbitrary geometries.

It requires two inputs and has several optional flags.

- The siesta.TBT.nc file which contains the geometry that is to be calculated for
  The reason for using the siesta.TBT.nc file is the ease of use:

    The siesta.TBT.nc contains electrode atoms and device atoms. Hence it
    becomes easy to read in the electrode atomic positions.
    Note that since you'll always do a 0 V calculation this isn't making
    any implications for the requirement of the TBT.nc file.
"""
from __future__ import annotations

from ._btd import *
from ._electrode import *
from ._green import *
