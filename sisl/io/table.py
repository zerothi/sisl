from __future__ import print_function

import numpy as np

# Import sile objects
from .sile import *

__all__ = ['TableSile']


class TableSile(Sile):
    """ ASCII tabular formatted data

    Intrinsically this uses the `numpy.savetxt` routine to store the content.
    """

    def _setup(self, *args, **kwargs):
        """ Setup the `TableSile` after initialization """
        self._comment = ['#']

    @Sile_fh_open
    def write_data(self, *args, **kwargs):
        """ Write tabular data to the file with optional header. 

        Parameters
        ----------
        *args : array_like or list of array_like
            the different columns in the tabular data.
            This may either be several 1D data, or 2D arrays.

            Internally the data is stacked via `numpy.vstack` and for
            any dimension higher than 2 it gets separated by two newline
            characters (like gnuplot acceptable data).
        fmt : str, optional
            The formatting string, defaults to ``'.5e'``.
        newline : str, optional
            Defaults to ``'\n'``.
        delimiter : str, optional
            Defaults to ``'\t'``.
        comment : str or list of str, optional
            A pre-header text at the top of the file.
            This comment is automatically prepended a ``'#'`` at the
            start of new lines, for lists each item corresponds to a newline
            which is automatically appended.
        header : str or list of str, optional
            A header for the data.
        footer : str or list of str, optional
            A footer written at the end of the file.

        Examples
        --------

        >>> tbl = sisl.io.TableSile('test.dat')
        >>> tbl.write_data(np.arange(2), np.arange(2) + 1, comment='This is nothing', header=['index', 'value'])
        >>> print(open('test.dat').readlines())
        # This is nothing
        index    value
        0        1
        1        2
        """
        fmt = kwargs.get('fmt', '.5e')
        newline = kwargs.get('newline', '\n')
        delimiter = kwargs.get('delimiter', '\t')

        comment = kwargs.get('comment', None)
        if comment is None:
            comment = ''
        elif isinstance(comment, (list, tuple)):
            comment = (newline + self._comment[0] + ' ').join(comment)
        if len(comment) > 0:
            comment = self._comment[0] + ' ' + comment + newline

        header = kwargs.get('header', None)
        if header is None:
            header = ''
        elif isinstance(header, (list, tuple)):
            header = delimiter.join(header)
        header = comment + header

        footer = kwargs.get('footer', None)
        if footer is None:
            footer = ''
        elif isinstance(footer, (list, tuple)):
            footer = newline.join(footer)

        # Create a unified data table
        dat = np.vstack(args)

        _fmt = '{:' + fmt + '}'

        # Reshape
        if len(dat.shape) > 2:
            dat.shape = (-1, dat.shape[-2], dat.shape[-1])

        # Now we are ready to write
        self._write(header)

        if len(dat.shape) > 2:
            _fmt = (_fmt + delimiter) * (dat.shape[1] - 1) + _fmt + newline
            for i in range(dat.shape[0]):
                for j in range(dat.shape[2]):
                    self._write(_fmt.format(*dat[i, :, j]))
                self._write(newline * 2)
        else:
            _fmt = (_fmt + delimiter) * (dat.shape[0] - 1) + _fmt + newline
            for i in range(dat.shape[1]):
                self._write(_fmt.format(*dat[:, i]))

        if len(footer) > 0:
            self._write(newline * 2 + footer)

    # Specify the default write function
    _write_default = write_data


add_sile('table', TableSile, case=False, gzip=True)
add_sile('dat', TableSile, case=False, gzip=True)
