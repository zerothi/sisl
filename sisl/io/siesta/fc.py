import numpy as np

from ..sile import add_sile, sile_fh_open
from .sile import SileSiesta

from sisl._internal import set_module
from sisl.messages import warn
from sisl.unit.siesta import unit_convert


__all__ = ['fcSileSiesta']


@set_module("sisl.io.siesta")
class fcSileSiesta(SileSiesta):
    """ Force constant file """

    @sile_fh_open()
    def read_force(self, displacement=None, na=None):
        """ Reads all displacement forces by multiplying with the displacement value

        Since the force constant file does not contain the non-displaced configuration
        this will only return forces on the displaced configurations minus the forces from
        the non-displaced configuration.

        This may be used in conjunction with phonopy by noticing that Siesta FC-runs does
        the displacements in reverse order (-x/+x vs. +x/-x). In this case one should reorder
        the elements like this:

        >>> fc = np.roll(fc, 1, axis=2)

        Parameters
        ----------
        displacement : float, optional
           the used displacement in the calculation, since Siesta 4.1-b4 this value
           is written in the FC file and hence not required.
           If prior Siesta versions are used and this is not supplied the 0.04 Bohr displacement
           will be assumed.
        na : int, optional
           number of atoms in geometry (for returning correct number of atoms), since Siesta 4.1-b4
           this value is written in the FC file and hence not required.
           If prior Siesta versions are used then the file is expected to only contain 1-atom displacement.

        Returns
        -------
        numpy.ndarray : (displaced atoms, d[xyz], [-+], total atoms, xyz)
             force constant matrix times the displacement, see `read_force_constant` for details regarding
             data layout.
        """
        if displacement is None:
            line = self.readline().split()
            self.fh.seek(0)
            try:
                displacement = float(line[-1])
            except:
                warn(f"{self.__class__.__name__}.read_force assumes displacement=0.04 Bohr!")
                displacement = 0.04 * unit_convert('Bohr', 'Ang')

        # Since the displacements changes sign (starting with a negative sign)
        # we can convert using this scheme
        displacement = np.repeat(displacement, 6).ravel()
        displacement[1::2] *= -1
        return self.read_force_constant(na) * displacement.reshape(1, 3, 2, 1, 1)

    @sile_fh_open()
    def read_force_constant(self, na=None):
        """ Reads the force-constant stored in the FC file

        Parameters
        ----------
        na : int, optional
           number of atoms in the unit-cell, if not specified it will guess on only
           one atom displacement.

        Returns
        -------
        numpy.ndarray : (displacement, d[xyz], [-+], atoms, xyz)
             force constant matrix containing all forces. The 2nd dimension contains
             contains the directions, 3rd dimension contains -/+ displacements.
        """
        # Force constants matrix
        line = self.readline().split()
        if na is None:
            try:
                na = int(line[-2])
            except:
                na = None

        fc = list()
        while True:
            line = self.readline()
            if line == '':
                # empty line or nothing
                break
            fc.append(list(map(float, line.split())))

        # Units are already eV / Ang ** 2
        fc = np.array(fc)

        # Slice to correct size
        if na is None:
            na = fc.size // 6 // 3

        # Correct shape of matrix
        fc.shape = (-1, 3, 2, na, 3)

        return fc


add_sile('FC', fcSileSiesta, gzip=True)
add_sile('FCC', fcSileSiesta, gzip=True)
