import os.path as osp
import numpy as np

# Import sile objects
from .sile import *

from sisl._internal import set_module
from sisl import Geometry, Atom, SuperCell
from sisl.utils import str_spec


__all__ = ["vestaSile"]


@set_module("sisl.io")
class vestaSile(Sile):
    """ VESTA file for VESTA """

    @sile_fh_open()
    def write_header(self):
        """ Write header information (basically a file information) """

        # Check that we can write to the file
        sile_raise_write(self)
        
        # Write out top-header stuff
        from time import gmtime, strftime
        self._write("# File created by: sisl {}\n".format(strftime("%Y-%m-%d", gmtime())))
        self._write("#VESTA_FORMAT_VERSION 3.0.0\n")

    @sile_fh_open()
    def write_supercell(self, sc, fmt=".8f"):
        """ Writes the supercell to the contained file

        Parameters
        ----------
        sc : SuperCell
           the supercell to be written
        fmt : str, optional
           used format for the precision of the data
        """
        # Implementation notice!
        # The XSF files are compatible with Vesta, but ONLY
        # if there are no empty lines!

        # Check that we can write to the file
        sile_raise_write(self)

        self.write_header()

        self._write("CRYSTAL\n")
        self._write("# Cell in Angstroem and degrees\n")
        self._write("CELLP\n")
        # We write the cell coordinates as the cell coordinates
        fmt_str = f"{{:{fmt}}} " * 6 + "\n"
        self._write(fmt_str.format(*sc.parameters()))

    @sile_fh_open()
    def write_geometry(self, geometry, fmt=".8f"):
        """ Writes the geometry to the contained file

        Parameters
        ----------
        geometry : Geometry
           the geometry to be written
        fmt : str, optional
           used format for the precision of the data
        """
        self.write_supercell(geometry.sc, fmt)


add_sile("vesta", vestaSile, case=False, gzip=True)
