# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import pytest

from sisl.unit.codata import read_codata

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("year", (2010, 2014, 2018, 2022))
def test_codata(year):
    codata = read_codata(year)
    assert codata["year"] == f"{year}"
