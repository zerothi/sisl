# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Mapping, Sequence, Union

__all__ = ["UnitsVar"]

UnitsVar = Union[str, Mapping[str, str], Sequence[str]]
