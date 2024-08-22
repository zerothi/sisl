# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Union

# To prevent import cycles place any internal imports in the branch below
# and use a string literal forward reference to it in subsequent types
# https://mypy.readthedocs.io/en/latest/common_issues.html#import-cycles
if TYPE_CHECKING:
    from sisl import Atom, Atoms

__all__ = [
    "AtomLike",
    "AtomsLike",
]


# A definition of a type that can be parsed to an Atom object
# It should be anything that the Atom can accept instead of Z
AtomLike = Union[int, str, "Atom"]
AtomsLike = Union[AtomLike, Sequence[AtomLike], "Atoms"]
