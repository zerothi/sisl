# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""
Linear algebra
==============

Although `numpy` and `scipy` provides a large set of
linear algebra routines, sisl re-implements some of them with
a reduced memory and/or computational effort. This is because
`numpy.linalg` and `scipy.linalg` routines are defaulting
to a large variety of checks to assert the input matrices.

sisl implements its own variants which has interfaces much
like `numpy` and `scipy`.

   inv
   solve
   eig
   eigh
   svd
   eigs
   eigsh

"""
from .base import *
from .special import *
