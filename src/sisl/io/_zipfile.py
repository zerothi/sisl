# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import zipfile
from pathlib import Path
from typing import Literal


class ZipPath(zipfile.Path):
    """Extension of the zipfile.Path class to mimic the pathlib.Path class.

    The main goal is this extension of zipfile.Path can be used within the
    sile framework as a drop in replacement for pathlib.Path. By enabling
    that, all the reading/writing functionality also works with zipfiles.

    The extension has two purposes:

    1. Some methods were missing in the original zipfile.Path class if we
    wanted to use it as a Path object:

        - ``with_suffix``

    2. When writing a file in a zipfile, it is important to not only close the
    file handle for the written file, but also to close the zipfile itself, so
    that it writes the changes to disk.

    To this end, when `open` is called, we wrap the `close` method of the returned
    file handle so that we can close the zipfile if necessary.
    """

    #: Class variable to keep track of all the open files for a given
    #: zipfile.
    _open_files: dict[str, dict[str, Literal[True]]] = {}

    #: Absolute path to the root zipfile.
    _zip_abs_path: str = None

    def __init__(self, zipfile, *args, **kwargs):
        """Initialize the ZipPath with a zipfile object and a path"""
        super().__init__(zipfile, *args, **kwargs)

        self._zip_abs_path = Path(zipfile.filename).resolve()
        if self._zip_abs_path not in self._open_files:
            self._open_files[self._zip_abs_path] = {}

    @property
    def suffix(self):
        """Override the suffix property to ensure it returns a string"""
        return Path(str(self)).suffix

    def with_suffix(self, *args, **kwargs):
        """Override the with_suffix method to ensure it returns a ZipPath"""

        file_path = Path(str(self)).relative_to(self.root.filename)

        return self.__class__(self.root, str(file_path.with_suffix(*args, **kwargs)))

    def open(self, *args, **kwargs):
        """Override the open method to ensure it returns a file handle"""
        fh = super().open(*args, **kwargs)

        self._open_files[self._zip_abs_path][fh] = True

        close = fh.close

        def _close(*args, **kwargs):
            close(*args, **kwargs)

            self._open_files[self._zip_abs_path].pop(fh, None)

            if not self._open_files[self._zip_abs_path]:
                self.root.close()

        fh.close = _close
        return fh

    @classmethod
    def from_zipfile_path(cls, zipfile_path):
        """Create a ZipPath from a zipfile object"""

        file_path = Path(str(zipfile_path)).relative_to(zipfile_path.root.filename)

        return cls(zipfile_path.root, str(file_path))

    @classmethod
    def from_path(cls, path: Path, mode: str = "r"):
        """Create a ZipPath from a Path object.

        Given a path object, scans
        """
        for i, part in enumerate(path.parts[:-1]):
            if part.endswith(".zip"):
                zip_path = Path(*path.parts[: i + 1])
                if zip_path.is_file():
                    zipfile_mode = "r" if mode.startswith("r") else "a"
                    root_zip = zipfile.ZipFile(zip_path, zipfile_mode)
                    return cls(root_zip, str(path.relative_to(zip_path)))
        else:
            raise FileNotFoundError(f"Could not find zip file in {path}")
