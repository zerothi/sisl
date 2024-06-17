# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import scipy.sparse as sps

# To prevent import cycles place any internal imports in the branch below
# and use a string literal forward reference to it in subsequent types
# https://mypy.readthedocs.io/en/latest/common_issues.html#import-cycles
if TYPE_CHECKING:
    from sisl import BaseSile

__all__ = [
    "SileLike",
    "GeometryLike",
    "LatticeLike",
    "LatticeOrGeometryLike",
    "GridLike",
]


SileLike = Union[
    str,
    "BaseSile",
    Path,
]
"""Anything that can be intercepted as a `Sile`"""

GeometryLike = Union[
    SileLike,
    "Geometry",
]
"""Data-types that can be converted to a `Geometry`"""

LatticeLike = Union[
    SileLike,
    "Lattice",
]
"""Data-types that can be converted to a `Lattice`"""

GridLike = Union[
    SileLike,
    "Grid",
]
"""Data-types that can be converted to a `Grid`"""


LatticeOrGeometryLike = Union[
    LatticeLike,
    GeometryLike,
]
"""Data-types that can be either a `Lattice` or `Geometry`"""
