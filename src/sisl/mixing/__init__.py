# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""
Mixing objects
==============

Implementation of various mixing methods.

Mixing methods are typical objects used for speeding up convergence
for self-consistent methods.

sisl implements a variety of the standard methods used in the electronic
structure community and its easy implementations allows easy testing for
new methods.
"""

from .base import *
from .diis import *
from .linear import *
