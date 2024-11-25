# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Literal, Union

__all__ = [
    "GaugeType",
    "ProjectionType",
    "ProjectionTypeTrace",
    "ProjectionTypeDiag",
    "ProjectionTypeMatrix",
    "ProjectionTypeHadamard",
    "ProjectionTypeHadamardAtoms",
]

GaugeType = Literal["cell", "atom"]

ProjectionTypeMatrix = Literal["matrix", "ij"]
ProjectionTypeTrace = Literal["trace", "sum"]
ProjectionTypeDiag = Literal["diagonal", "diag", "ii"]
ProjectionTypeHadamard = Literal["hadamard", "basis"]
ProjectionTypeHadamardAtoms = Literal["hadamard:atoms", "atoms"]
ProjectionType = Union[ProjectionTypeMatrix, ProjectionTypeDiag, ProjectionTypeTrace]
