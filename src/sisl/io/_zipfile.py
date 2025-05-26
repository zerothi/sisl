# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import functools
import sys
import zipfile
from pathlib import Path

_is_python_old = sys.version_info < (3, 10)


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

    Parameters
    ----------
    *args :
        Arguments passed to the zipfile.Path constructor.
    close_zipfile :
        Whether to close the root zipfile when closing an open file handle
        produced by this path.
        Closing the zipfile is necessary for the changes to be written to disk.
        However, if the zipfile is passed by the user, we do not want to close
        the zipfile, because we don't know if the user wants to continue writing
        to it.
        Therefore, the default is ``False``, but if the ``ZipPath`` is created
        from a normal path (i.e. the user didn't input the zipfile, it was created
        on the fly), we set it to ``True`` (see ``ZipPath.from_path``).
    **kwargs :
        Keyword arguments passed to the zipfile.Path constructor.
    """

    #: Whether to close the root zipfile when closing the file handles.
    close_zipfile: bool = False

    def __init__(self, *args, close_zipfile: bool = False, **kwargs):
        """Initialize the ZipPath with a zipfile object and a path"""
        if _is_python_old:
            raise RuntimeError(
                "Zip file functionality in sisl requires Python 3.10 or newer. "
                "Upgrade your Python version if you want to use it."
            )

        super().__init__(*args, **kwargs)

        self.close_zipfile = close_zipfile

    @property
    def suffix(self):
        """Override the suffix property to ensure it returns a string"""
        return Path(str(self)).suffix

    def with_suffix(self, *args, **kwargs):
        """Override the with_suffix method to ensure it returns a ZipPath"""

        file_path = Path(str(self)).relative_to(self.root.filename)

        return self.__class__(self.root, str(file_path.with_suffix(*args, **kwargs)))

    def open(self, *args, **kwargs):
        """Override the open method to patch the file handle if necessary"""

        fh = super().open(*args, **kwargs)

        if self.close_zipfile:
            # The root zipfile needs to be closed when closing the file handle,
            # so we wrap the close method of the file handle
            close = fh.close

            @functools.wraps(close)
            def _close(*args, **kwargs):
                close(*args, **kwargs)
                self.root.close()

            fh.close = _close

        return fh

    @classmethod
    def from_zipfile_path(cls, zipfile_path: zipfile.Path, close_zipfile: bool = False):
        """Create a ZipPath from a zipfile.Path object"""

        file_path = Path(str(zipfile_path)).relative_to(zipfile_path.root.filename)

        return cls(zipfile_path.root, str(file_path), close_zipfile=close_zipfile)

    @classmethod
    def from_path(cls, path: Path, mode: str = "r", close_zipfile: bool = True):
        """Create a ZipPath from a Path object.

        Given a path object, scans the path to find the first zip file in the path.
        If a zip file is found, a ZipPath object is created.

        This function initializes a new ``zipfile.ZipFile`` object to use as the root.

        Parameters
        ----------
        path :
            The path to the file or directory.
        mode :
            The mode in which you will want to open paths. This
            determines the mode in which the zipfile is opened.
        close_zipfile :
            Whether to close the zipfile when closing the file handle.
            This is important because if you write to a zipfile, you need
            to close it to write the changes to disk.
            Since the zipfile is created internally, the default is to close
            it when we are done reading/writing to the path.
        """
        for i, part in enumerate(path.parts[:-1]):
            if part.endswith(".zip"):
                zip_path = Path(*path.parts[: i + 1])
                if zip_path.is_file():
                    zipfile_mode = "r" if mode.startswith("r") else "a"
                    root_zip = zipfile.ZipFile(zip_path, zipfile_mode)
                    return cls(
                        root_zip,
                        str(path.relative_to(zip_path)),
                        close_zipfile=close_zipfile,
                    )

        # If we got here it is because we did not find a zip file in the path,
        # so we raise an error
        raise FileNotFoundError(f"Could not find zip file in {path}")
