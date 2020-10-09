"""
Sile object for reading/writing XYZ files
"""

from re import compile as re_compile
import numpy as np

# Import sile objects
from .sile import *

from sisl._internal import set_module
from sisl import Geometry, SuperCell
from sisl.messages import warn
import sisl._array as _a


__all__ = ['xyzSile']


def _header_to_dict(header):
    """ Convert a header line with 'key=val key1=val1' sequences to a single dictionary """
    e = re_compile(r"(\S+)=")

    # 1. Remove *any* entry with 0 length
    # 2. Ensure it is a list
    # 3. Reverse the list order (for popping)
    kv = list(filter(lambda x: len(x.strip()) > 0, e.split(header)))[::-1]

    # Now create the dictionary
    d = {}
    while len(kv) >= 2:
        # We have reversed the list
        key = kv.pop().strip(' =') # remove white-space *and* =
        val = kv.pop().strip() # remove outer whitespace
        d[key] = val

    return d


@set_module("sisl.io")
class xyzSile(Sile):
    """ XYZ file object """

    @sile_fh_open()
    def write_geometry(self, geom, fmt='.8f', comment=None):
        """ Writes the geometry to the contained file

        Parameters
        ----------
        geom : Geometry
           the geometry to be written
        fmt : str, optional
           used format for the precision of the data
        comment : str, optional
           if None, a sisl made comment that can be used for parsing the unit-cell is used
           else this comment will be written at the 2nd line.
        """
        # Check that we can write to the file
        sile_raise_write(self)

        # Write the number of atoms in the geometry
        self._write('   {}\n'.format(len(geom)))

        # Write out the cell information in the comment field
        # This contains the cell vectors in a single vector (3 + 3 + 3)
        # quantities, plus the number of supercells (3 ints)
        if comment is None:
            fmt_str = 'sisl-version=1 cell= ' + f'{{:{fmt}}} ' * 9 + ' nsc= {} {} {}\n'.format(*geom.nsc[:])
            self._write(fmt_str.format(*geom.cell.flatten()))
        else:
            self._write(f"{comment}\n")

        fmt_str = '{{0:2s}}  {{1:{0}}}  {{2:{0}}}  {{3:{0}}}\n'.format(fmt)
        for ia, a, _ in geom.iter_species():
            s = {'fa': 'Ds'}.get(a.symbol, a.symbol)
            self._write(fmt_str.format(s, *geom.xyz[ia, :]))

    def _r_geometry_sisl(self, na, header, sp, xyz):
        """ Read the geometry as though it was created with sisl """
        # Default version of the header is 1
        v = header.get("sisl-version", 1)
        nsc = list(map(int, header.pop("nsc").split()))
        cell = _a.fromiterd(header.pop("cell").split()).reshape(3, 3)

        return Geometry(xyz, atoms=sp, sc=SuperCell(cell, nsc=nsc))

    def _r_geometry_ase(self, na, header, sp, xyz):
        """ Read the geometry as though it was created with ASE """
        # Convert F T to nsc
        #  F = 1
        #  T = 3
        nsc = list(map(lambda x: "FT".index(x) * 2 + 1, header.pop("pbc").strip('"').split()))
        cell = _a.fromiterd(header.pop("Lattice").strip('"').split()).reshape(3, 3)

        return Geometry(xyz, atoms=sp, sc=SuperCell(cell, nsc=nsc))

    def _r_geometry(self, na, sp, xyz):
        """ Read the geometry for a generic xyz file (not sisl, nor ASE) """
        # The cell dimensions isn't defined, we are going to create a molecule box
        cell = xyz.max(0) - xyz.min(0) + 10.
        return Geometry(xyz, atoms=sp, sc=SuperCell(cell, nsc=[1] * 3))

    @sile_fh_open()
    def read_geometry(self):
        """ Returns Geometry object from the XYZ file """
        # Read number of atoms
        na = int(self.readline())

        # Read header, and try and convert to dictionary
        header = self.readline()
        kv = _header_to_dict(header)

        # Read atoms and coordinates
        sp = [None] * na
        xyz = np.empty([na, 3], np.float64)
        for ia in range(na):
            l = self.readline().split()
            sp[ia] = l.pop(0)
            xyz[ia, :] = [float(k) for k in l[:3]]

        def _has_keys(d, *keys):
            for key in keys:
                if not key in d:
                    return False
            return True

        if _has_keys(kv, "cell", "nsc"):
            return self._r_geometry_sisl(na, kv, sp, xyz)
        elif _has_keys(kv, "Properties", "Lattice", "pbc"):
            return self._r_geometry_ase(na, kv, sp, xyz)
        return self._r_geometry(na, sp, xyz)

    def ArgumentParser(self, p=None, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geometry().ArgumentParser(p, *args, **newkw)


add_sile('xyz', xyzSile, case=False, gzip=True)
