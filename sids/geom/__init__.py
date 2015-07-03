"""
Geometry package for obtaining/retaining geometrical structures.

It enables the creation, extension and in general mingling with 
a geometry consisting of a super-cell and atomic coordinates.

It also enables the rotation of the coordinates using quaternions
"""

# Easy import all sub-classes
from sids.geom.atom import *
from sids.geom.geometry import *
from sids.geom.quaternion import *

# Import default creation routines
from sids.geom.default import *

