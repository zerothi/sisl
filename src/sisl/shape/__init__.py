# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""
Shapes
======

A variety of default shapes.

All shapes inherit the `Shape` class.

All shapes in sisl allows one to perform arithmetic on them.
I.e. one may *add* two shapes to accomblish what would be equivalent
to an ``&`` operation. The resulting shape will be a ``CompositeShape`` which
implements the necessary routines to ensure correct operation.

Currently these mathematical/boolean operators are implemented:

`A & B`
    intersection of shapes

`A | B` or `A + B`
    union of shapes

`A ^ B`
    the disjunction union

`A - B`
    complementary shape

   Shape - base class
   Cuboid - 3d cube
   Cube - 3d box
   Ellipsoid
   Sphere
   EllipticalCylinder
   NullShape

"""
from ._cylinder import *
from .base import *
from .ellipsoid import *
from .prism4 import *

__all__ = [s for s in dir() if not s.startswith("_")]
