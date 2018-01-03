from __future__ import print_function, division

# Import sile objects
from .sile import SileSiesta
from ..sile import add_sile, Sile_fh_open

# Import the geometry object
from sisl import Orbital, AtomicOrbital
from sisl import PeriodicTable, Atom, Atoms
from sisl.unit.siesta import unit_convert

Bohr2Ang = unit_convert('Bohr', 'Ang')

__all__ = ['OrbIndxSileSiesta']


class OrbIndxSileSiesta(SileSiesta):
    """ .ORB_INDX file object """

    @Sile_fh_open
    def read_basis(self):
        """ Returns a set of atoms corresponding to the basis-sets in the ORB_INDX file

        The specie names have a short field in the ORB_INDX file, hence the name may
        not necessarily be the same as provided in the species block
        """

        # First line contains no no_s
        line = self.readline().split()
        no = int(line[0])

        # Read two new lines
        self.readline()
        self.readline()

        pt = PeriodicTable()

        def crt_atom(spec, orbs):
            i = pt.Z(spec)
            if isinstance(i, int):
                return Atom(i, orbs)
            else:
                return Atom(-1, orbs, tag=spec)

        # Now we begin by reading the atoms
        atom = []
        orbs = []
        specs = []
        ia = 1
        for io in range(no):
            line = self.readline().split()

            i_a = int(line[1])
            if i_a != ia:
                if i_s not in specs:
                    atom.append(crt_atom(spec, orbs))
                specs.append(i_s)
                ia = i_a
                orbs = []

            i_s = int(line[2]) - 1
            if i_s in specs:
                continue
            spec = line[3]
            nlmz = list(map(int, line[5:9]))
            P = line[9] == 'T'
            rc = float(line[11]) * Bohr2Ang
            # Create the orbital
            o = AtomicOrbital(n=nlmz[0], l=nlmz[1], m=nlmz[2], Z=nlmz[3], P=P,
                              spherical=Orbital(rc))
            orbs.append(o)

        if i_s not in specs:
            atom.append(crt_atom(spec, orbs))
        specs.append(i_s)

        # Now re-arrange the atoms and create the Atoms object
        return Atoms([atom[i] for i in specs])


add_sile('ORB_INDX', OrbIndxSileSiesta, gzip=True)
