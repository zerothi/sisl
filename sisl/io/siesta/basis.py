from __future__ import print_function, division

from sisl._help import xml_parse
from sisl.atom import Atom
from sisl.orbital import SphericalOrbital
from sisl.io import add_sile
from sisl._array import arrayd, aranged
from sisl.unit.siesta import unit_convert
from .sile import SileSiesta, SileCDFSiesta


__all__ = ['ionxmlSileSiesta', 'ionncSileSiesta']


class ionxmlSileSiesta(SileSiesta):
    """ Basis set information in xml format

    Note that the ``ion`` files are equivalent to the ``ion.xml`` files.
    """

    def read_basis(self):
        """ Returns data associated with the ion.xml file """
        # Get the element-tree
        root = xml_parse(self.file).getroot()

        # Get number of orbitals
        label = root.find('label').text.strip()
        Z = int(root.find('z').text) # atomic number, negative for floating
        mass = float(root.find('mass').text)

        # Read in the PAO's
        paos = root.find('paos')

        # Now loop over all orbitals
        orbital = []

        # All orbital data
        Bohr2Ang = unit_convert('Bohr', 'Ang')
        for orb in paos:

            n = int(orb.get('n'))
            l = int(orb.get('l'))
            z = int(orb.get('z')) # zeta

            q0 = float(orb.get('population'))

            P = not int(orb.get('ispol')) == 0

            # Radial components
            rad = orb.find('radfunc')
            npts = int(rad.find('npts').text)

            # Grid spacing in Bohr (conversion is done later
            # because the normalization is easier)
            delta = float(rad.find('delta').text)

            # Read in data to a list
            dat = list(map(float, rad.find('data').text.split()))

            # Since the readed data has fewer significant digits we
            # might as well re-create the table of the radial component.
            r = aranged(npts) * delta

            # To get it per Ang**3
            # TODO, check that this is correct.
            # The fact that we have to have it normalized means that we need
            # to convert psi /sqrt(Bohr**3) -> /sqrt(Ang**3)
            # \int psi^\dagger psi == 1
            psi = arrayd(dat[1::2]) * r ** l / Bohr2Ang ** (3./2.)

            # Create the sphericalorbital and then the atomicorbital
            sorb = SphericalOrbital(l, (r * Bohr2Ang, psi), q0)

            # This will be -l:l (this is the way siesta does it)
            orbital.extend(sorb.toAtomicOrbital(n=n, Z=z, P=P))

        # Now create the atom and return
        return Atom(Z, orbital, mass=mass, tag=label)


class ionncSileSiesta(SileCDFSiesta):
    """ Basis set information in NetCDF files

    Note that the ``ion.nc`` files are equivalent to the ``ion.xml`` files.
    """

    def read_basis(self):
        """ Returns data associated with the ion.xml file """
        no = len(self._dimension('norbs'))

        # Get number of orbitals
        label = self.Label.strip()
        Z = int(self.Atomic_number)
        mass = float(self.Mass)

        # Retrieve values
        orb_l = self._variable('orbnl_l')[:] # angular quantum number
        orb_n = self._variable('orbnl_n')[:] # principal quantum number
        orb_z = self._variable('orbnl_z')[:] # zeta
        orb_P = self._variable('orbnl_ispol')[:] > 0 # polarization shell, or not
        orb_q0 = self._variable('orbnl_pop')[:] # q0 for the orbitals
        orb_delta = self._variable('delta')[:] # delta for the functions
        orb_psi = self._variable('orb')[:, :]

        # Now loop over all orbitals
        orbital = []

        # All orbital data
        Bohr2Ang = unit_convert('Bohr', 'Ang')
        for io in range(no):

            n = orb_n[io]
            l = orb_l[io]
            z = orb_z[io]
            P = orb_P[io]

            # Grid spacing in Bohr (conversion is done later
            # because the normalization is easier)
            delta = orb_delta[io]

            # Since the readed data has fewer significant digits we
            # might as well re-create the table of the radial component.
            r = aranged(orb_psi.shape[1]) * delta

            # To get it per Ang**3
            # TODO, check that this is correct.
            # The fact that we have to have it normalized means that we need
            # to convert psi /sqrt(Bohr**3) -> /sqrt(Ang**3)
            # \int psi^\dagger psi == 1
            psi = orb_psi[io, :] * r ** l / Bohr2Ang ** (3./2.)

            # Create the sphericalorbital and then the atomicorbital
            sorb = SphericalOrbital(l, (r * Bohr2Ang, psi), orb_q0[io])

            # This will be -l:l (this is the way siesta does it)
            orbital.extend(sorb.toAtomicOrbital(n=n, Z=z, P=P))

        # Now create the atom and return
        return Atom(Z, orbital, mass=mass, tag=label)


add_sile('ion.xml', ionxmlSileSiesta, case=False, gzip=True)
add_sile('ion.nc', ionncSileSiesta, case=False)
