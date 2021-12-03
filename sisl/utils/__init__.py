# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Utility routines
================

Several utility functions are used throughout sisl.

Range routines
==============

   array_arange - fast creation of sub-aranges
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
   iter_shape
   math_eval

"""
from .cmd import *
from .misc import *
from .ranges import *
from . import mathematics as math
