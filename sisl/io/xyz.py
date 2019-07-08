"""
Sile object for reading/writing XYZ files
"""
from __future__ import print_function

from collections import defaultdict
import re
import numpy as np

# Import sile objects
from .sile import *

# Import the geometry object
from sisl import Geometry, SuperCell

# Warnings
from sisl.messages import warn


__all__ = ['xyzSile']


class xyzSile(Sile):
    """ XYZ file object """

    @sile_fh_open()
    def write_geometry(self, geom, fmt='.8f'):
        """ Writes the geometry to the contained file

        Parameters
        ----------
        geom : Geometry
           the geometry to be written
        fmt : str, optional
           used format for the precision of the data
        """
        # Check that we can write to the file
        sile_raise_write(self)

        # Write the number of atoms in the geometry
        self._write('   {}\n'.format(len(geom)))

        # Write out the cell information in the comment field
        # This contains the cell vectors in a single vector (3 + 3 + 3)
        # quantities, plus the number of supercells (3 ints)
        fmt_str = 'cell= ' + '{{:{0}}} '.format(fmt) * 9 + ' nsc= {} {} {}\n'.format(*geom.nsc[:])
        self._write(fmt_str.format(*geom.cell.flatten()))

        fmt_str = '{{0:2s}}  {{1:{0}}}  {{2:{0}}}  {{3:{0}}}\n'.format(fmt)
        for ia, a, _ in geom.iter_species():
            s = {'fa': 'Ds'}.get(a.symbol, a.symbol)
            self._write(fmt_str.format(s, *geom.xyz[ia, :]))
        # Add a single new line
        self._write('\n')

    @sile_fh_open()
    def read_geometry(self):
        """ Returns Geometry object from the XYZ file """

        cell = np.asarray(np.diagflat([1] * 3), np.float64)
        nsc = [1, 1, 1]
        l = self.readline()
        na = int(l)

        # Attempt parsing comment line for key=value pairs
        l = self.readline()
        l = l.split()
        keymatch = re.compile(r"(\w+)=(.*)")
        keyval = defaultdict(lambda: [])
        cur_key = "nokey"
        for w in l:
            key = keymatch.match(w)
            if key is not None:
                cur_key = key.group(1).lower()
                w = key.group(2)
            w = w.strip().strip("\"")
            keyval[cur_key].append(w)
        keyval = {k: " ".join(v) for k, v in keyval.items()}

        # Now set cell if given
        cell_set = False
        for cellkey in ("lattice", "cell"):
            try:
                values = keyval[cellkey].split()
                if len(values) != 9:
                    raise ValueError(
                        "There are not exactly 9 values associated to the"
                        " {cellkey} values".format(cellkey=cellkey))
                cell.shape = (9,)
                for i, il in enumerate(values):
                    cell[i] = float(il)
                cell_set = True
                cell.shape = (3, 3)
            except KeyError:  # cellkey not in keyvals
                continue
            except ValueError as e:
                warn(
                    "Found a key indicating {cellkey} in xyz file, but "
                    "could not parse it: {e!s}".format(cellkey=cellkey, e=e))

        # Now set nsc if its there
        try:
            values = keyval["nsc"].split()
            if len(values) != 3:
                raise ValueError(
                    "There are not exactly 3 values associated to the"
                    " cell/lattice values")
            for i, il in enumerate(values):
                nsc[i] = int(il)
        except KeyError:
            pass
        except ValueError as e:
            warn(
                "Found a key indicating nsc in xyz file, but "
                "could not parse it: {e!s}".format(e=e))

        sp = [None] * na
        xyz = np.empty([na, 3], np.float64)
        for ia in range(na):
            l = self.readline().split()
            sp[ia] = l.pop(0)
            xyz[ia, :] = [float(k) for k in l[:3]]

        # Fix the maximum size of the supercell
        # by adding 10 A vacuum
        if not cell_set:
            cell = xyz.max(0) - xyz.min(0) + 10.

        return Geometry(xyz, atom=sp, sc=SuperCell(cell, nsc=nsc))

    def ArgumentParser(self, p=None, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geometry().ArgumentParser(p, *args, **newkw)


add_sile('xyz', xyzSile, case=False, gzip=True)
