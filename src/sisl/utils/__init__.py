# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""
Utility routines
================

Several utility functions are used throughout sisl.

Range routines
==============

   strmap
   strseq
   lstranges
   erange
   list2str
   fileindex

Miscellaneous routines
======================

   str_spec
   direction - abc/012 -> 012
   angle - radian to degree
   listify
   iter_shape
   math_eval
   batched_indices

"""
from . import mathematics as math
from ._arrays import *
from .cmd import *
from .misc import *
from .ranges import *
