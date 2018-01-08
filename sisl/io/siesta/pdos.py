from __future__ import print_function

try:
    from defusedxml import ElementTree
except ImportError:
    from xml.etree.ElementTree import ElementTree

# Import sile objects
from ..sile import add_sile
from .sile import SileSiesta
from sisl._array import arrayd, emptyd
from sisl.atom import PeriodicTable, Atom, Atoms
from sisl.geometry import Geometry
from sisl.orbital import AtomicOrbital
from sisl.unit.siesta import unit_convert


__all__ = ['pdosSileSiesta']

Bohr2Ang = unit_convert('Bohr', 'Ang')


class pdosSileSiesta(SileSiesta):
    """ PDOS Siesta file object

    Data file containing the PDOS as calculated by Siesta.
    """

    def read_data(self):
        """ Returns data associated with the PDOS file

        Returns
        -------
        geom : Geometry instance with positions, atoms and orbitals. The
               orbitals of these atoms are `AtomicOrbital` instances.
        E : the energies at which the PDOS has been evaluated at
        PDOS : an array of DOS, for non-polarized calculations it has dimension ``(atom.no, len(E))``,
               else it has dimension ``(nspin, atom.no, len(E))``.
        """
        # Get the element-tree
        ET = ElementTree('pdos', self.file)
        root = ET.getroot()

        # Get number of orbitals
        nspin = int(root.find('nspin').text)
        E = arrayd(map(float, root.find('energy_values').text.split()))
        ne = len(E)

        # All coordinate, atoms and species data
        xyz = []
        atoms = []
        atom_species = []
        def ensure_size(ia):
            while len(atom_species) <= ia:
                atom_species.append(None)
                xyz.append(None)
        def ensure_size_orb(ia, i):
            while len(atoms) <= ia:
                atoms.append([])
            while len(atoms[ia]) <= i:
                atoms[ia].append(None)

        data = []
        for orb in root.findall('orbital'):
            def oi(name):
                return int(orb.get(name))

            # Get indices
            ia = oi('atom_index') - 1
            i = oi('index') - 1

            species = orb.get('species')

            # Create the atomic orbital
            try:
                Z = oi('Z')
            except:
                try:
                    Z = PeriodicTable().Z(species)
                except:
                    # Unknown
                    Z = -1

            ensure_size(ia)
            xyz[ia] = list(map(float, orb.get('position').split()))
            atom_species[ia] = Z

            # "sadly" it doesn't contain information about polarization
            try:
                P = orb.get('P') == 'true'
            except:
                P = False
            O = AtomicOrbital(n=oi('n'), l=oi('l'), m=oi('m'), Z=oi('z'), P=P)

            # We know that the index is far too high. However,
            # this ensures a consecutive orbital
            ensure_size_orb(ia, i)
            atoms[ia][i] = O

            # it is formed like : spin-1, spin-2 (however already in eV)
            d = arrayd(map(float, orb.find('data').text.split())).reshape(-1, nspin)
            # Transpose and copy to get the corret size
            data.append((i, d.T.copy(order='C').reshape(nspin, -1)))

        # Now we need to parse the data
        # First reduce the atom
        atoms = [[o for o in a if o] for a in atoms]
        atoms = Atoms([Atom(Z, os) for Z, os in zip(atom_species, atoms)])
        geom = Geometry(arrayd(xyz) * Bohr2Ang, atoms)
        pdos = emptyd([nspin, atoms.no, ne])
        for i, dos in data:
            pdos[:, i, :] = dos[:, :]

        if nspin == 1:
            pdos.shape = (atoms.no, ne)

        return geom, E, pdos

# PDOS files are:
# They contain the same file (same xml-data)
# However, pdos.xml is preferred because it has higher precision.
#  siesta.PDOS
add_sile('PDOS', pdosSileSiesta, case=False, gzip=True)
#  pdos.xml/siesta.PDOS.xml
add_sile('PDOS.xml', pdosSileSiesta, case=False, gzip=True)
