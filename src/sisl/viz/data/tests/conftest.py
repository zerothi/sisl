# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import os.path as osp

import pytest


@pytest.fixture(scope="session")
def siesta_test_files(sisl_files):
    def _siesta_test_files(path):
        return sisl_files(osp.join("sisl", "io", "siesta", path))

    return _siesta_test_files
