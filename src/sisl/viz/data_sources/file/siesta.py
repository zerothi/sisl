# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from pathlib import Path

import sisl

from .file_source import FileData  # noqa: F401


def get_sile(path=None, fdf=None, cls=None, **kwargs):
    """Wrapper around FileData.get_sile that infers files from the root fdf

    Parameters
    ----------
    path : str or Path, optional
        the path to the file to be read.
    cls : sisl.io.SileSiesta, optional
        if `path` is not provided, we try to infer it from the root fdf file,
        looking for files that fullfill this class' rules.

    Returns
    ---------
    Sile:
        The sile object.
    """
    if fdf is not None and isinstance(fdf, (str, Path)):
        fdf = get_sile(path=fdf)

    if path is None:
        if cls is None:
            raise ValueError(f"Either a path or a class must be provided to get_sile")
        if fdf is None:
            raise ValueError(
                f"We can not look for files of a sile type without a root fdf file."
            )

        for rule in sisl.get_sile_rules(cls=cls):
            filename = fdf.get("SystemLabel", default="siesta") + f".{rule.suffix}"
            try:
                path = fdf.dir_file(filename)
                return get_sile(path=path, **kwargs)
            except:
                pass
        else:
            raise FileNotFoundError(
                f"Tried to find a {cls} from the root fdf ({fdf.file}), "
                f"but didn't find any."
            )

    return sisl.get_sile(path, **kwargs)


def FileDataSIESTA(path=None, fdf=None, cls=None, **kwargs):
    if isinstance(path, sisl.io.BaseSile):
        path = path.file
    return get_sile(path=path, fdf=fdf, cls=cls, **kwargs)
