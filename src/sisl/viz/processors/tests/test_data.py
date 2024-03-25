# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import pytest

from sisl.viz.data import Data
from sisl.viz.processors.data import accept_data, extract_data

pytestmark = [pytest.mark.viz, pytest.mark.processors]


class FakeData(Data):
    def __init__(self, valid: bool = True):
        self._data = valid

    def sanity_check(self):
        assert self._data == True


class OtherData(Data):
    pass


@pytest.mark.parametrize("valid", [True, False])
def test_accept_data(valid):
    data = FakeData(valid)

    # If the input is an instance of an invalid class
    with pytest.raises(TypeError):
        accept_data(data, OtherData)

    # Perform a sanity check on data
    if valid:
        assert accept_data(data, FakeData) is data
    else:
        with pytest.raises(AssertionError):
            accept_data(data, FakeData)

    # Don't perform a sanity check on data
    assert accept_data(data, FakeData, check=False) is data


@pytest.mark.parametrize("valid", [True, False])
def test_extract_data(valid):
    data = FakeData(valid)

    # If the input is an instance of an invalid class
    with pytest.raises(TypeError):
        extract_data(data, OtherData)

    # Perform a sanity check on data
    if valid:
        assert extract_data(data, FakeData) is data._data
    else:
        with pytest.raises(AssertionError):
            extract_data(data, FakeData)

    # Don't perform a sanity check on data
    assert extract_data(data, FakeData, check=False) is data._data
