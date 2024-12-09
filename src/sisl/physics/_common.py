# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from sisl.typing import GaugeType, ProjectionType

__all__ = ["comply_gauge", "comply_projection"]


def comply_gauge(gauge: GaugeType) -> str:
    """Comply the gauge to one of two words: atom | cell"""
    return {
        "R": "lattice",
        "cell": "lattice",
        "lattice": "lattice",
        "r": "atomic",
        "orbital": "atomic",
        "orbitals": "atomic",
        "atom": "atomic",
        "atoms": "atomic",
        "atomic": "atomic",
    }[gauge]


def comply_projection(projection: ProjectionType) -> str:
    """Comply the projection to one of the allowed variants"""
    return {
        "matrix": "matrix",
        "ij": "matrix",
        "trace": "trace",
        "sum": "trace",
        "diagonal": "diagonal",
        "diag": "diagonal",
        "ii": "diagonal",
        "hadamard": "hadamard",
        "basis": "hadamard",
        "orbital": "hadamard",
        "orbitals": "hadamard",
        "hadamard:atoms": "hadamard:atoms",
        "atoms": "hadamard:atoms",
        "atom": "hadamard:atoms",
    }[projection]
