# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from sisl._internal import set_module

from ._except_base import *

__all__ = ["MissingFermiLevelWarning"]


@set_module("sisl.io")
class MissingFermiLevelWarning(SileWarning):
    """The Fermi level is not present in the file."""
