from __future__ import print_function

try:
    from defusedxml import ElementTree
except ImportError:
    from xml.etree.ElementTree import ElementTree

# Import sile objects
from ..sile import add_sile
from .sile import SileSiesta
from sisl._array import arrayd


__all__ = ['pdosSileSiesta']


class pdosSileSiesta(SileSiesta):
    """ PDOS Siesta file object """

    def read_data(self):
        """ Returns data associated with the PDOS file

        The returned quantities are provided with an ``E`` array
        and a list of class objects which contain DOS.

        The fields of the objects are:

        n, l, m, z, index, atom_index, DOS
        """
        # Get the element-tree
        ET = ElementTree('pdos', self.file)
        root = ET.getroot()

        # Get number of orbitals
        nspin = int(root.find('nspin').text)
        no = int(root.find('norbitals').text)
        E = arrayd(map(float, root.find('energy_values').text.split()))

        # Now loop over all orbitals
        class Orbital(object):
            pass

        # All orbital data
        data = []
        for orb in root.findall('orbital'):
            O = Orbital()
            O.n = int(orb.get('n'))
            O.l = int(orb.get('l'))
            O.m = int(orb.get('m'))
            O.z = int(orb.get('z'))
            O.index = int(orb.get('index'))
            O.atom_index = int(orb.get('atom_index'))
            O.DOS = arrayd(map(float, orb.find('data').text.split()))
            data.append(O)

        return E, data

add_sile('PDOS', pdosSileSiesta, case=False, gzip=True)
