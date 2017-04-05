"""
Sile object for reading/writing a file with tabular data
"""

from __future__ import print_function

import numpy as np

# Import sile objects
from .sile import *

__all__ = ['TableSile']


class TableSile(Sile):
    """ Table file object """

    def _setup(self):
        """ Setup the `TableSile` after initialization """
        self._comment = ['#']

    def write_data(self, data, header=None, footer=None, newline='\n', fmt='%.5e', comment=None, delimiter='\t'):
        """ Write tabular data to the file with optional header. """
        C = self._comment[0]

        if comment is None:
            comment = ''
        elif isinstance(comment, (list, tuple)):
            comment = newline.join(comment)
        if comment is not None:
            comment = comment + newline

        if header is None:
            header = ''
        elif isinstance(header, (list, tuple)):
            header = delimiter.join(header)
        header = comment + header

        if footer is None:
            footer = ''
        elif isinstance(footer, (list, tuple)):
            footer = newline.join(footer)

        # Use numpy to store the txt data
        np.savetxt(self.file, data.T, header=header, footer=footer,
                   fmt=fmt, delimiter=delimiter)

    # Specify the default write function
    _write_default = write_data


add_sile('table', TableSile, case=False, gzip=True)
