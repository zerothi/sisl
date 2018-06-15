from __future__ import print_function, division

import numpy as np

# Import sile objects
from ..sile import add_sile, Sile_fh_open
from .sile import *
from sisl.unit.siesta import unit_convert


__all__ = ['fcSileSiesta']


class fcSileSiesta(SileSiesta):
    """ Force constants Siesta file object """

    @Sile_fh_open
    def read_force(self, na=None, displacement=None):
        """ Reads all displacement forces by multiplying with the displacement value

        I.e. this is equivalent to requesting the forces on a per-displacement calculation.

        Since the force constant file does not contain the non-displaced configuration
        this will only return forces on the displaced configurations minus the non-displaced forces.

        Parameters
        ----------
        na : int, optional
           number of atoms (for returning correct number of atoms), since Siesta 4.1-b4 this value
           is written in the FC file and hence not required.
           If prior Siesta versions are used then the file is expected to only contain 1-atom displacement.
        displacement : float, optional
           the used displacement in the calculation, since Siesta 4.1-b4 this value
           is written in the FC file and hence not required.
           If prior Siesta versions are used and this is not supplied the 0.04 Bohr displacement
           will be assumed.

        Returns
        -------
        forces : numpy.ndarray with 4 dimensions containing all the forces. The 2nd dimensions contains
                 -x/+x/-y/+y/-z/+z displacements.
        """
        # Force constants matrix
        line = self.readline().split()
        if displacement is None:
            try:
                na = int(line[-2])
                displacement = float(line[-1])
            except:
                na = None
                displacement = 0.04 * unit_convert('Bohr', 'Ang')

        fc = list()
        while True:
            line = self.readline()
            if line == '':
                # empty line or nothing
                break
            fc.append(list(map(float, line.split())))

        # Units are already eV / Ang ** 2
        fc = - np.array(fc) * displacement
        # Slice to correct size
        if na is None:
            na = fc.size // 6 // 3
        fc.shape = (-1, 6, na, 3)
        return fc

    @Sile_fh_open
    def read_hessian(self, geometry, ):
        """ Returns Hessian from the file """
        fc = - self.read_force(displacement=1.) # to not do anything
        return fc


add_sile('FC', fcSileSiesta, case=False, gzip=True)
add_sile('FCC', fcSileSiesta, case=False, gzip=True)
