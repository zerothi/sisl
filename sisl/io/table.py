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

    def write(self, data, header=None, fmt='%.5e', comment=None, delimiter='\t'):
        """ Write tabular data to the file with optional header. """

        if comment is None:
            comment = ''
        else:
            comment = self._comment[0] + comment.replace('\n','\n'+self._comment[0]) + '\n'
            
        if header is None:
            header = ''
        else:
            header = self._comment[0] + delimiter.join(header)

        # Use numpy to store the txt data
        np.savetxt(self.file, data.T, header=header, fmt=fmt, comments=comment, delimiter=delimiter)


add_sile('table', TableSile, case=False, gzip=True)
