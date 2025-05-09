# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import tempfile
import zipfile

import sisl
from sisl.io._zipfile import ZipPath


def test_zipfile_preserved():
    """Test that the zipfile is preserved through the sile framework

    This is VERY important, because the lookup for paths will only be
    fast if the index is already built. If each time a new sile is
    created we need to create a new zipfile, it will be very slow.
    """
    # Write a temporary zipfile
    tempfile_path = tempfile.mktemp(suffix=".zip")
    with zipfile.ZipFile(tempfile_path, "w") as f:
        ...

    f = zipfile.ZipFile(tempfile_path)

    root_path = zipfile.Path(f, "")

    fdf = sisl.get_sile(root_path / "test.fdf")

    assert isinstance(fdf.file, ZipPath)
    assert fdf.file.root is f
