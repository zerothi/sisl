# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, Union

import numpy as np
import numpy.typing as npt

from ._common import SparseMatrixGeometry

if TYPE_CHECKING:
    from sisl.physics import SparseOrbitalBZ, SparseOrbitalBZSpin, Spin

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
    "SpinLiteral",
    "SpinType",
    "SparseMatrixPhysical",
]

GaugeType = Literal["lattice", "atomic"]

SpinLiteral = Literal[
    "",
    "unpolarized",
    "colinear",
    "collinear",
    "polarized",
    "p",
    "pol",
    "noncolinear",
    "noncollinear",
    "non-colinear",
    "non-collinear",
    "nc",
    "spinorbit",
    "spin-orbit",
    "so",
    "soc",
    "nambu",
    "bdg",
]

SpinType = Union["Spin", SpinLiteral]

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

SparseMatrixPhysical = Union[
    SparseMatrixGeometry, "SparseOrbitalBZ", "SparseOrbitalBZSpin"
]
