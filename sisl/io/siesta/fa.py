import numpy as np

from ..sile import add_sile, sile_fh_open, sile_raise_write
from .sile import SileSiesta

from sisl._internal import set_module


__all__ = ['faSileSiesta']


@set_module("sisl.io.siesta")
class faSileSiesta(SileSiesta):
    """ Forces file """

    @sile_fh_open()
    def read_force(self):
        """ Reads the forces from the file """
        na = int(self.readline())

        f = np.empty([na, 3], np.float64)
        for ia in range(na):
            f[ia, :] = list(map(float, self.readline().split()[1:]))

        # Units are already eV / Ang
        return f

    @sile_fh_open()
    def write_force(self, f, fmt='.9e'):
        """ Write forces to file

        Parameters
        ----------
        fmt : str, optional
           precision of written forces
        """
        sile_raise_write(self)
        na = len(f)
        self._write(f'{na}\n')
        _fmt = ('{:d}' + (' {:' + fmt + '}') * 3) + '\n'

        for ia in range(na):
            self._write(_fmt.format(ia + 1, *f[ia, :]))

    # Short-cut
    read_data = read_force
    write_data = write_force


add_sile('FA', faSileSiesta, gzip=True)
add_sile('FAC', faSileSiesta, gzip=True)
