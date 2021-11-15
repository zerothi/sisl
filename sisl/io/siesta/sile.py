# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

try:
    from . import _siesta
    found_bin_module = True
except Exception as e:
    print(e)
    found_bin_module = False

from sisl._internal import set_module
from ..sile import Sile, SileCDF, SileBin, SileError

__all__ = ['SileSiesta', 'SileCDFSiesta', 'SileBinSiesta']


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

    def _fortran_check(self, method, message):
        ierr = _siesta.io_m.iostat_query()
        if ierr != 0:
            raise SileError(f'{self!s}.{method} {message} (ierr={ierr})')

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
            if mode == 'r':
                self._iu = _siesta.io_m.open_file_read(self.file)
            elif mode == 'w':
                self._iu = _siesta.io_m.open_file_write(self.file)
            else:
                raise SileError(f"Mode '{mode}' is not an accepted mode to open a fortran file unit. Use only 'r' or 'w'")
        self._fortran_check('_fortran_open', 'could not open for {}.'.format({'r': 'reading', 'w': 'writing'}[mode]))

    def _fortran_close(self):
        if not self._fortran_is_open():
            return
        # Close it
        _siesta.io_m.close_file(self._iu)
        self._iu = -1
