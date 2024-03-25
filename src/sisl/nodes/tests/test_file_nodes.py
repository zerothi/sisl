# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from sisl.nodes import FileNode


def test_file_node_return():
    n = FileNode("test.txt")

    assert n.get() == Path("test.txt")


def test_file_node_update():
    pytest.importorskip("watchdog")

    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        n = FileNode(f.name)

        assert n.get() == Path(f.name)

        assert not n._outdated

        f.write("test")

    time.sleep(0.2)

    assert n._outdated
