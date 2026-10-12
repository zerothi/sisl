# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

""" Local sisl fixtures """
import pytest


@pytest.fixture(scope="module", autouse=True)
def ensure_nodify():
    pytest.importorskip("nodify")
