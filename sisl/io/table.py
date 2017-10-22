from __future__ import print_function

import numpy as np

# Import sile objects
from sisl._help import ensure_array
from .sile import Sile, add_sile, Sile_fh_open, sile_raise_write

__all__ = ['TableSile']


class TableSile(Sile):
    """ ASCII tabular formatted data """

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
        fmts : str, optional
            The formatting string (for all columns), defaults to ``fmt * len(args)`.
            `fmts` has precedence over `fmt`.
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
            A header for the data (this will be prepended a comment if not present).
        footer : str or list of str, optional
            A footer written at the end of the file.

        Examples
        --------

        >>> tbl = TableSile('test.dat', 'w')
        >>> tbl.write_data(range(2), range(1, 3), comment='A comment', header=['index', 'value'])
        >>> print(open('test.dat').readlines())
        # A comment
        #index    value
        0        1
        1        2

        """
        sile_raise_write(self)

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
        if header is not None:
            # Ensure no "dangling" spaces
            header = header.strip()
            if not header.startswith(self._comment[0]):
                header = self._comment[0] + header
            if not header.endswith('\n'):
                header += '\n'
        header = comment + header

        footer = kwargs.get('footer', None)
        if footer is None:
            footer = ''
        elif isinstance(footer, (list, tuple)):
            footer = newline.join(footer)

        # Create a unified data table
        # This is rather difficult because we have to guess the
        # input size
        if len(args) == 1:
            dat = np.stack(args[0])
        else:
            dat = np.stack(args)

        _fmt = '{:' + fmt + '}'

        # Reshape
        if len(dat.shape) > 2:
            dat.shape = (-1, dat.shape[-2], dat.shape[-1])

        # Now we are ready to write
        self._write(header)

        if len(dat.shape) > 2:
            _fmt = kwargs.get('fmts', (_fmt + delimiter) * (dat.shape[1] - 1) + _fmt + newline)
            for i in range(dat.shape[0]):
                for j in range(dat.shape[2]):
                    self._write(_fmt.format(*dat[i, :, j]))
                self._write(newline * 2)
        else:
            _fmt = kwargs.get('fmts', (_fmt + delimiter) * (dat.shape[0] - 1) + _fmt + newline)
            for i in range(dat.shape[1]):
                self._write(_fmt.format(*dat[:, i]))

        if len(footer) > 0:
            self._write(newline * 2 + footer)

    @Sile_fh_open
    def read_data(self, *args, **kwargs):
        """ Read tabular data from the file.

        Parameters
        ----------
        columns : list of int, optional
            only return the indices of the columns that are provided
        delimiter : str, optional
            the delimiter used in the file, will automatically try to guess if not specified
        ret_comment : bool, optional
            also return the comments at the top of the file (if queried)
        ret_header : bool, optional
            also return the header information (if queried)
        comment : str, optional
            lines starting with this are discarded as comments
        """
        # Override the comment in the file
        self._comment = [kwargs.get('comment', self._comment[0])]

        # Skip to next line
        lines = []
        comment = []
        header = ''

        # Also read comments
        line = self.readline(True)
        while line.startswith(self._comment[0] + ' '):
            comment.append(line)
            line = self.readline(True)

        if line.startswith(self._comment[0]):
            header = line

        # Now we are ready to read the data
        dat = [[]]
        line = self.readline()

        # First we need to figure out the separator:
        len_sep = 0
        sep = kwargs.get('delimiter', '')
        if len(sep) == 0:
            for cur_sep in ['\t', ' ', ',']:
                s = line.split(cur_sep)
                if len(s) > len_sep:
                    len_sep = len(s)
                    sep = cur_sep
            if len(sep) == 0:
                raise ValueError(self.__class__.__name__ + '.read_data could not determine '
                                 'column separator...')

        while len(line) > 0:
            # If we start a line by a comment, or a newline
            # then we have a new data set
            if line.startswith('\n'):
                dat[-1] = ensure_array(dat[-1], np.float64)
                dat.append([])
            else:
                line = [l for l in line.split(sep) if len(l) > 0]
                dat[-1].append(ensure_array(map(float, line), np.float64))

            line = self.readline()
        dat[-1] = ensure_array(dat[-1], np.float64)

        # Ensure we have no false positives
        dat = [d for d in dat if len(d) > 0]
        dat = ensure_array(dat, np.float64)
        if dat.shape[0] == 1:
            s = list(dat.shape)
            s.pop(0)
            dat.shape = tuple(s)

        dat = np.swapaxes(dat, -2, -1)
        if kwargs.get('ret_comment', False):
            if kwargs.get('ret_header', False):
                return dat, comment, header
            return dat, comment
        return dat

    # Specify the default write function
    _write_default = write_data


add_sile('table', TableSile, case=False, gzip=True)
add_sile('dat', TableSile, case=False, gzip=True)
