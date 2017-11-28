from __future__ import print_function, division

try:
    from defusedxml import ElementTree
except ImportError:
    from xml.etree.ElementTree import ElementTree

from sisl.atom import Atom
from sisl.io import add_sile
from sisl._array import arrayd, aranged
from sisl.unit.siesta import unit_convert
from .sile import SileSiesta


__all__ = ['ionxmlSileSiesta']


class ionxmlSileSiesta(SileSiesta):
    """ ion.xml Siesta file object """

    def read_data(self):
        """ Returns data associated with the ion.xml file """
        # Get the element-tree
        ET = ElementTree(None, self.file)
        root = ET.getroot()

        # Get number of orbitals
        symbol = root.find('symbol').text.strip()
        label = root.find('label').text.strip()
        Z = int(root.find('z').text)
        mass = float(root.find('mass').text)
        l_max = int(root.find('lmax_basis').text)
        norbs_nl = int(root.find('norbs_nl').text)
        l_max_projs = int(root.find('lmax_projs').text)
        nprojs_nl = int(root.find('nprojs_nl').text)

        # Read in the PAO's
        paos = root.find('paos')

        # Now loop over all orbitals
        class Orbital(object):
            pass

        # All orbital data
        Bohr2Ang = unit_convert('Bohr', 'Ang')
        data = []
        for orb in paos:

            O = Orbital()
            # Get the atom
            O.atom = Atom(Z, mass=mass, tag=label)
            O.n = int(orb.get('n'))
            O.l = int(orb.get('l'))
            O.z = int(orb.get('z'))

            O.is_polarization = int(orb.get('ispol')) != 0
            O.population = float(orb.get('population'))

            # Radial components
            rad = orb.find('radfunc')
            npts = int(rad.find('npts').text)

            # Grid spacing in Ang
            O.delta = float(rad.find('delta').text) * Bohr2Ang
            # Cutoff of orbital
            O.cutoff = float(rad.find('cutoff').text) * Bohr2Ang

            # Read in data to a list
            dat = map(float, rad.find('data').text.split())

            # Since the readed data has fewer significant digits we
            # might as well re-create the table of the radial component.
            O.r = aranged(npts) * O.delta

            # To get it per Ang**3 (r is already converted)
            # TODO, check that this is correct. Inelastica does it differently
            # However, it seems Denchar does the below:
            O.psi = arrayd(dat[1::2]) * O.r ** O.l / Bohr2Ang ** 3

            data.append(O)

        return data

add_sile('ion.xml', ionxmlSileSiesta, case=False, gzip=True)
