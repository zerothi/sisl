# Import sile objects
from .sile import *

# Import the geometry object
from sisl import Geometry


__all__ = ['moldenSile']


class moldenSile(Sile):
    """ Molden file object """

    @sile_fh_open()
    def write_supercell(self, sc):
        """ Writes the supercell to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

        # Write the number of atoms in the geometry
        self._write('[Molden Format]\n')

        # Sadly, MOLDEN does not read this information...

    @sile_fh_open()
    def write_geometry(self, geom, fmt='.8f'):
        """ Writes the geometry to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

        # Be sure to write the supercell
        self.write_supercell(geom.sc)

        # Write in ATOM mode
        self._write('[Atoms] Angs\n')

        # Write out the cell information in the comment field
        # This contains the cell vectors in a single vector (3 + 3 + 3)
        # quantities, plus the number of supercells (3 ints)

        fmt_str = '{{0:2s}} {{1:4d}} {{2:4d}}  {{3:{0}}}  {{4:{0}}}  {{5:{0}}}\n'.format(fmt)
        for ia, a, _ in geom.iter_species():
            self._write(fmt_str.format(a.symbol, ia, a.Z, *geom.xyz[ia, :]))

    def ArgumentParser(self, p=None, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geometry().ArgumentParser(p, *args, **newkw)


add_sile('molf', moldenSile, case=False, gzip=True)
