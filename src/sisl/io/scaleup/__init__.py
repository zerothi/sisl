# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""
ScaleUp
=======

The interaction between sisl and `ScaleUp`_ allows constructing large TB models
to be post-processed in the NEGF code `TBtrans`_.

   orboccSileScaleUp - orbital information
   refSileScaleUp - reference coordinates
   rhamSileScaleUp - Hamiltonian file

"""
from .sile import *  # isort: split
from .orbocc import *
from .ref import *
from .rham import *
