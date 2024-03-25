# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from pathlib import Path


def path_rel_or_abs(path, base=None):
    path = Path(path)
    if path.is_absolute():
        return path
    if base is None:
        base = Path.cwd()
    return base / path


def path_abs(path, base=None):
    path = Path(path)
    if path.is_absolute():
        return path
    if base is None:
        base = Path.cwd()
    return base / path
