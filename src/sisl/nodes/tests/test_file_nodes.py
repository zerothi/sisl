import tempfile
from pathlib import Path
import time

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
