import re
import numpy as np

from sisl._internal import set_module
import sisl._array as _a
from .sile import Sile, add_sile, sile_fh_open, sile_raise_write


__all__ = ['tableSile', 'TableSile']


@set_module("sisl.io")
class tableSile(Sile):
    """ ASCII tabular formatted data

    A generic table data which will easily accommodate the most common write-outs of data

    Examples
    --------

    >>> a = np.arange(6).reshape(3, 2)
    >>> tbl = tableSile('test.dat', 'w')
    >>> tbl.write_data(a)
    >>> print(''.join(open('test.dat').readlines())) # doctest: +NORMALIZE_WHITESPACE
    0.00000e+00	2.00000e+00	4.00000e+00
    1.00000e+00	3.00000e+00	5.00000e+00
    <BLANKLINE>

    >>> a = np.arange(6).reshape(3, 2)
    >>> tbl = tableSile('test.dat', 'w')
    >>> tbl.write_data(a, comment='Hello')
    >>> print(''.join(open('test.dat').readlines())) # doctest: +NORMALIZE_WHITESPACE
    # Hello
    0.00000e+00	2.00000e+00	4.00000e+00
    1.00000e+00	3.00000e+00	5.00000e+00
    <BLANKLINE>

    >>> a = np.arange(6).reshape(3, 2)
    >>> tbl = tableSile('test.dat', 'w')
    >>> tbl.write_data(a, header=['x', 'y'])
    >>> print(''.join(open('test.dat').readlines())) # doctest: +NORMALIZE_WHITESPACE
    #x	y
    0.00000e+00	2.00000e+00	4.00000e+00
    1.00000e+00	3.00000e+00	5.00000e+00
    <BLANKLINE>

    >>> a = np.arange(6).reshape(3, 2)
    >>> tbl = tableSile('test.dat', 'w')
    >>> tbl.write_data(a.T)
    >>> print(''.join(open('test.dat').readlines())) # doctest: +NORMALIZE_WHITESPACE
    0.00000e+00	1.00000e+00
    2.00000e+00	3.00000e+00
    4.00000e+00	5.00000e+00
    <BLANKLINE>

    >>> x = np.arange(4)
    >>> y = np.arange(8).reshape(2, 4) + 4
    >>> tbl = tableSile('test.dat', 'w')
    >>> tbl.write_data(x, y)
    >>> print(''.join(open('test.dat').readlines())) # doctest: +NORMALIZE_WHITESPACE
    0.00000e+00	4.00000e+00	8.00000e+00
    1.00000e+00	5.00000e+00	9.00000e+00
    2.00000e+00	6.00000e+00	1.00000e+01
    3.00000e+00	7.00000e+00	1.10000e+01
    <BLANKLINE>

    """

    def _setup(self, *args, **kwargs):
        """ Setup the `tableSile` after initialization """
        self._comment = ['#']

    @sile_fh_open()
    def write_data(self, *args, **kwargs):
        """ Write tabular data to the file with optional header.

        Parameters
        ----------
        args : array_like or list of array_like
            the different columns in the tabular data.
            This may either be several 1D data, or 2D arrays.
            Internally the data is stacked via `numpy.vstack` and for
            any dimension higher than 2 it gets separated by two newline
            characters (like gnuplot acceptable data).
        fmt : str, optional
            The formatting string, defaults to ``.5e``.
        fmts : str, optional
            The formatting string (for all columns), defaults to ``fmt * len(args)``.
            `fmts` has precedence over `fmt`.
        newline : str, optional
            Defaults to ``\\n``.
        delimiter : str, optional
            Defaults to ``\\t``.
        comment : str or list of str, optional
            A pre-header text at the top of the file.
            This comment is automatically prepended a ``#`` at the
            start of new lines, for lists each item corresponds to a newline
            which is automatically appended.
        header : str or list of str, optional
            A header for the data (this will be prepended a comment if not present).
        footer : str or list of str, optional
            A footer written at the end of the file.

        Examples
        --------

        >>> tbl = tableSile('test.dat', 'w')
        >>> tbl.write_data(range(2), range(1, 3), comment='A comment', header=['index', 'value'])
        >>> print(''.join(open('test.dat').readlines())) # doctest: +NORMALIZE_WHITESPACE
        # A comment
        #index    value
        0.00000e+00 1.00000e+00
        1.00000e+00 2.00000e+00
        <BLANKLINE>
        """
        sile_raise_write(self)

        fmt = kwargs.get('fmt', '.5e')
        newline = kwargs.get('newline', '\n')
        delimiter = kwargs.get('delimiter', '\t')
        _com = self._comment[0]

        def comment_newline(line, prefix=''):
            """ Converts a list of str arguments into nicely formatted commented
            and newlined output """
            nonlocal _com
            line = map(lambda s: s.strip(), line.strip().split(newline))
            # always append a newline
            line = newline.join([s if s.startswith(_com) else f"{_com}{prefix}{s}" for s in line]) + newline
            return line

        comment = kwargs.get('comment', None)
        if comment is None:
            comment = ''
        elif isinstance(comment, str):
            comment = comment_newline(comment, ' ')
        else:
            comment = comment_newline(newline.join(comment), ' ')

        header = kwargs.get('header', None)
        if header is None:
            header = ''
        elif isinstance(header, str):
            header = comment_newline(header)
        else:
            header = comment_newline(delimiter.join(header))

        # Finalize output
        header = comment + header

        footer = kwargs.get('footer', None)
        if footer is None:
            footer = ''
        elif isinstance(footer, str):
            pass
        else:
            footer = newline.join(footer)

        # Now we are ready to write
        if len(header) > 0:
            self._write(header)

        # Create a unified data table
        # This is rather difficult because we have to guess the
        # input size
        if len(args) == 0:
            # This may be used in cases where one wishes to accummulate content
            # in a file.
            return
        elif len(args) == 1:
            if isinstance(args[0], np.ndarray):
                # Ensure that when we change the shape we are doing it on a view
                dat = args[0].view()
            else:
                # Probably a tuple/list passed
                dat = np.vstack(args[0])
        else:
            dat = np.vstack(args)

        _fmt = '{:' + fmt + '}'

        # Reshape such that it becomes easy
        ndim = dat.ndim
        if ndim > 2:
            dat.shape = (-1, dat.shape[-2], dat.shape[-1])
        elif ndim == 1:
            dat.shape = (1, -1)

        if ndim > 2:
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

    @sile_fh_open()
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
        comment = []
        header = ''

        # Also read comments
        line = self.readline(True)
        while line.startswith(self._comment[0] + ' '):
            comment.append(line)
            line = self.readline(True)

        if line.startswith(self._comment[0]):
            header = line
            line = self.readline()

        # Now we are ready to read the data
        dat = [[]]

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

        empty = re.compile(r'\s*\n')
        while len(line) > 0:
            # If we start a line by a comment, or a newline
            # then we have a new data set
            if empty.match(line) is not None:
                if len(dat[-1]) > 0:
                    dat[-1] = _a.asarrayd(dat[-1])
                    dat.append([])
            else:
                line = [l for l in line.split(sep) if len(l) > 0]
                dat[-1].append(_a.fromiterd(map(float, line)))

            line = self.readline()
        if len(dat[-1]) > 0:
            dat[-1] = _a.asarrayd(dat[-1])

        # Ensure we have no false positives
        dat = _a.asarrayd([d for d in dat if len(d) > 0])
        if dat.shape[0] == 1:
            s = list(dat.shape)
            s.pop(0)
            dat.shape = tuple(s)

        if dat.ndim == 2:
            if dat.shape[1] == 1:
                # surely a 1D data
                dat.shape = (-1,)

        # For 2D data we need to transpose because the data is
        # read row wise, but stored column wise
        if dat.ndim > 1:
            dat = np.swapaxes(dat, -2, -1)

        ret_comment = kwargs.get('ret_comment', False)
        ret_header = kwargs.get('ret_header', False)
        if ret_comment:
            if ret_header:
                return dat, comment, header
            return dat, comment
        elif ret_header:
            return dat, header
        return dat

    # Specify the default write function
    _write_default = write_data


TableSile = tableSile

add_sile('table', tableSile, case=False, gzip=True)
add_sile('dat', tableSile, case=False, gzip=True)
