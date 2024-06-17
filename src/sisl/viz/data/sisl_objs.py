# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Any, get_type_hints

from sisl import Geometry, Grid, Hamiltonian

from .data import Data


class SislObjData(Data):
    """Base class for sisl objects"""

    def __instancecheck__(self, instance: Any) -> bool:
        expected_type = get_type_hints(self.__class__)["_data"]
        return isinstance(instance, expected_type)

    def __subclasscheck__(self, subclass: Any) -> bool:
        expected_type = get_type_hints(self.__class__)["_data"]
        return issubclass(subclass, expected_type)


class GeometryData(Data):
    """Geometry data class"""

    _data: Geometry


class GridData(Data):
    """Grid data class"""

    _data: Grid


class HamiltonianData(Data):
    """Hamiltonian data class"""

    _data: Hamiltonian
