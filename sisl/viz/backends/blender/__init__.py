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
