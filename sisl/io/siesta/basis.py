from __future__ import print_function, division

try:
    from defusedxml import ElementTree
except ImportError:
    from xml.etree.ElementTree import ElementTree

from sisl.atom import Atom
from sisl.orbital import SphericalOrbital, AtomicOrbital
from sisl.io import add_sile
from sisl._array import arrayd, aranged
from sisl.unit.siesta import unit_convert
from .sile import SileSiesta


__all__ = ['ionxmlSileSiesta']


class ionxmlSileSiesta(SileSiesta):
    """ ion.xml Siesta file object

    Note that the *.ion files are equivalent to the *.ion.xml files.
    However, the former has less precision and thus ion.xml files are
    preferred.
    """

    def read_basis(self):
        """ Returns data associated with the ion.xml file """
        # Get the element-tree
        ET = ElementTree(None, self.file)
        root = ET.getroot()

        # Get number of orbitals
        symbol = root.find('symbol').text.strip()
        label = root.find('label').text.strip()
        Z = int(root.find('z').text) # atomic number
        mass = float(root.find('mass').text)
        l_max = int(root.find('lmax_basis').text)
        norbs_nl = int(root.find('norbs_nl').text)
        l_max_projs = int(root.find('lmax_projs').text)
        nprojs_nl = int(root.find('nprojs_nl').text)

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

            P = not int(orb.get('ispol')) == 0

            # Radial components
            rad = orb.find('radfunc')
            npts = int(rad.find('npts').text)

            # Grid spacing in Ang
            delta = float(rad.find('delta').text)

            # Read in data to a list
            dat = map(float, rad.find('data').text.split())

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
            sorb = SphericalOrbital(l, (r * Bohr2Ang, psi))

            # This will be -l:l (this is the way siesta does it)
            orbital.extend(sorb.toAtomicOrbital(n=n, Z=z, P=P))

        # Now create the atom and return
        return Atom(Z, orbital, tag=label)

add_sile('ion.xml', ionxmlSileSiesta, case=False, gzip=True)
