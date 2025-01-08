# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Callable, Literal, Union

import numpy as np
import numpy.typing as npt

__all__ = [
    "GaugeType",
    "ProjectionType",
    "ProjectionTypeTrace",
    "ProjectionTypeDiag",
    "ProjectionTypeMatrix",
    "ProjectionTypeHadamard",
    "ProjectionTypeHadamardAtoms",
    "DistributionFunc",
    "DistributionStr",
    "DistributionType",
]

GaugeType = Literal["lattice", "atomic"]

ProjectionTypeMatrix = Literal["matrix", "ij"]
ProjectionTypeTrace = Literal["trace", "sum"]
ProjectionTypeDiag = Literal["diagonal", "diag", "ii"]
ProjectionTypeHadamard = Literal["hadamard", "basis"]
ProjectionTypeHadamardAtoms = Literal["hadamard:atoms", "atoms"]
ProjectionType = Union[ProjectionTypeMatrix, ProjectionTypeDiag, ProjectionTypeTrace]

# The distribution method, can be a string, or a Callable
DistributionFunc = Callable[[npt.ArrayLike], np.ndarray]
DistributionStr = Literal[
    "gaussian",
    "lorentzian",
    "fermi",
    "bose-einstein",
    "cold",
    "step-function",
    "heaviside",
]
DistributionType = Union[DistributionStr, DistributionFunc]
