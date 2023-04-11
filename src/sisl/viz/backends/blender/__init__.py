# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
r"""Blender
==========

Blender is an open source general use 3D software. 

In science, we can profit from its excellent features to generate very nice images!

Currently, the following plots have a blender drawing backend implemented:
   GridPlot
"""

import bpy

from .backend import BlenderBackend, BlenderMultiplePlotBackend
from ._plots import *
from ._helpers import delete_all_objects
