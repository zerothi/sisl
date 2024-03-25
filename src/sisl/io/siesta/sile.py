# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

try:
    from . import _siesta

    has_fortran_module = True
except ImportError:
    has_fortran_module = False

from sisl._internal import set_module

from ..sile import (
    MissingInputSileError,
    MissingInputSileInfo,
    Sile,
    SileBin,
    SileCDF,
    SileError,
    missing_input,
)

__all__ = [
    "SileSiesta",
    "SileCDFSiesta",
    "SileBinSiesta",
    "MissingFDFSiestaError",
    "MissingFDFSiestaInfo",
    "missing_input_fdf",
]


@set_module("sisl.io.siesta")
class MissingFDFSiestaError(MissingInputSileError):
    pass


@set_module("sisl.io.siesta")
class MissingFDFSiestaInfo(MissingInputSileInfo):
    pass


@set_module("sisl.io.siesta")
def missing_input_fdf(
    inputs, executable: str = "siesta", when_exception: Exception = KeyError
):
    return missing_input(executable, inputs, MissingFDFSiestaError, when_exception)


@set_module("sisl.io.siesta")
class SileSiesta(Sile):
    pass


@set_module("sisl.io.siesta")
class SileCDFSiesta(SileCDF):
    # all netcdf output should not be masked
    def _setup(self, *args, **kwargs):
        super()._setup(*args, **kwargs)

        # all NetCDF routines actually returns masked arrays
        # this is to prevent Siesta CDF files from doing this.
        if hasattr(self, "fh"):
            self.fh.set_auto_mask(False)


@set_module("sisl.io.siesta")
class SileBinSiesta(SileBin):
    def _setup(self, *args, **kwargs):
        """We set up everything to handle the fortran I/O unit"""
        super()._setup(*args, **kwargs)
        self._iu = -1

    def _fortran_check(self, method, message, ret_msg=False):
        ierr = _siesta.io_m.iostat_query()
        msg = ""
        if ierr != 0:
            msg = f"{self!s}.{method} {message} (ierr={ierr})"
            if not ret_msg:
                raise SileError(msg)
        if ret_msg:
            return msg

    def _fortran_is_open(self):
        return self._iu != -1

    def _fortran_open(self, mode, rewind=False):
        if self._fortran_is_open() and mode == self._mode:
            if rewind:
                _siesta.io_m.rewind_file(self._iu)
            else:
                # retain indices
                return
        else:
            if mode == "r":
                self._iu = _siesta.io_m.open_file_read(self.file)
            elif mode == "w":
                self._iu = _siesta.io_m.open_file_write(self.file)
            else:
                raise SileError(
                    f"Mode '{mode}' is not an accepted mode to open a fortran file unit. Use only 'r' or 'w'"
                )
        self._fortran_check(
            "_fortran_open",
            "could not open for {}.".format({"r": "reading", "w": "writing"}[mode]),
        )

    def _fortran_close(self):
        if not self._fortran_is_open():
            return
        # Close it
        _siesta.io_m.close_file(self._iu)
        self._iu = -1
