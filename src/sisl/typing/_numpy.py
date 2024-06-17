# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from numpy import __version__

if tuple(map(int, __version__.split("."))) >= (1, 21, 0):
    # NDArray entered in 1.21.
    # numpy.typing entered in 1.20.0
    # we have numpy typing
    from numpy.typing import *
else:
    ArrayLike = "ArrayLike"
    NDArray = "NDArray"
    DTypeLike = "DTypeLike"
