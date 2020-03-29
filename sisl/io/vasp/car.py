import numpy as np

# Import sile objects
from .sile import SileVASP
from ..sile import *

# Import the geometry object
import sisl._array as _a
from sisl.messages import warn
from sisl import Geometry, PeriodicTable, Atom, SuperCell

__all__ = ['carSileVASP']


class carSileVASP(SileVASP):
    """ *CAR VASP files for defining geomtries

    This file-object handles both POSCAR and CONTCAR files
    """

    def _setup(self, *args, **kwargs):
        """ Setup the `carSile` after initialization """
        self._scale = 1.

    @sile_fh_open()
    def write_geometry(self, geometry, dynamic=True):
        r""" Writes the geometry to the contained file

        Parameters
        ----------
        geometry : Geometry
           geometry to be written to the file
        dynamic : bool or list, optional
           define which atoms are dynamic in the VASP run (default is True,
           which means all atoms are dynamic)

        Examples
        --------
        >>> car = carSileVASP('POSCAR', 'w')
        >>> geom = geom.graphene()
        >>> geom.write(car, dynamic=False) # fix all atoms
        >>> geom.write(car, dynamic=[False, (True, False, True)]) # fix 1st and y coordinate of 2nd
        """
        # Check that we can write to the file
        sile_raise_write(self)

        # LABEL
        self._write('sisl output\n')

        # Scale
        self._write('  1.\n')

        # Write unit-cell
        fmt = ('   ' + '{:18.9f}' * 3) + '\n'
        tmp = np.zeros([3], np.float64)
        for i in range(3):
            tmp[:3] = geometry.cell[i, :]
            self._write(fmt.format(*tmp))

        # Figure out how many species
        pt = PeriodicTable()
        s, d = [], []
        ia = 0
        while ia < geometry.na:
            atom = geometry.atoms[ia]
            specie = geometry.atoms.specie[ia]
            ia_end = (np.diff(geometry.atoms.specie[ia:]) != 0).nonzero()[0]
            if len(ia_end) == 0:
                # remaining atoms
                ia_end = geometry.na
            else:
                ia_end = ia + ia_end[0] + 1
            s.append(pt.Z_label(atom.Z))
            d.append(ia_end - ia)
            ia += d[-1]

        fmt = ' {:s}' * len(d) + '\n'
        self._write(fmt.format(*s))
        fmt = ' {:d}' * len(d) + '\n'
        self._write(fmt.format(*d))
        self._write('Selective dynamics\n')
        self._write('Cartesian\n')

        if isinstance(dynamic, bool):
            dynamic = [dynamic] * len(geometry)

        b2s = {True: 'T', False: 'F'}
        def todyn(fix):
            if isinstance(fix, bool):
                return '{0} {0} {0}\n'.format(b2s[fix])
            return '{} {} {}\n'.format(b2s[fix[0]], b2s[fix[1]], b2s[fix[2]])

        fmt = '{:18.9f} ' * 3
        for ia in geometry:
            self._write(fmt.format(*geometry.xyz[ia, :]) + todyn(dynamic[ia]))

    @sile_fh_open(True)
    def read_supercell(self):
        """ Returns `SuperCell` object from the CONTCAR/POSCAR file """

        # read first line
        self.readline()  # LABEL
        # Update scale-factor
        self._scale = float(self.readline())

        # Read cell vectors
        cell = np.empty([3, 3], np.float64)
        for i in range(3):
            cell[i, :] = list(map(float, self.readline().split()[:3]))
        cell *= self._scale

        return SuperCell(cell)

    @sile_fh_open()
    def read_geometry(self, ret_dynamic=False):
        r""" Returns Geometry object from the CONTCAR/POSCAR file

        Possibly also return the dynamics (if present)

        Parameters
        ----------
        ret_dynamic : bool, optional
           also read selective dynamics (if present), if not, a list of True will be returned
        """
        sc = self.read_supercell()

        # The species labels are not always included in *CAR
        line1 = self.readline().split()
        opt = self.readline().split()
        try:
            species = line1
            species_count = np.array(opt, np.int32)
        except:
            species_count = np.array(line1, np.int32)
            # We have no species...
            # We default to consecutive elements in the
            # periodic table.
            species = [i+1 for i in range(len(species_count))]
            err = '\n'.join([
                "POSCAR best format:",
                "  <Specie-1> <Specie-2>",
                "  <#Specie-1> <#Specie-2>",
                "Format not found, the species are defaulted to the first elements of the periodic table."])
            warn(err)

        # Create list of atoms to be used subsequently
        atom = [Atom[spec]
                for spec, nsp in zip(species, species_count)
                for i in range(nsp)]

        # Read whether this is selective or direct
        # Currently direct is not used
        opt = self.readline()
        dynamics = False
        if opt[0] in 'Ss':
            dynamics = True
            opt = self.readline()

        # Check whether this is in fractional or direct
        # coordinates
        cart = False
        if opt[0] in 'CcKk':
            cart = True

        # Number of atoms
        na = len(atom)
        # pre-create the dynamic list
        dynamic = [[False] * 3] * na

        xyz = _a.emptyd([na, 3])
        for ia in range(na):
            line = self.readline().split()
            xyz[ia, :] = list(map(float, line[:3]))
            if dynamics:
                dynamic[ia] = list(map(lambda x: x.lower() == 't', line[3:6]))

        if cart:
            # The unit of the coordinates are cartesian
            xyz *= self._scale
        else:
            xyz = xyz.dot(sc.cell)

        # The POT/CONT-CAR does not contain information on the atomic species
        geom = Geometry(xyz=xyz, atom=atom, sc=sc)
        if ret_dynamic:
            return geom, np.array(dynamic, dtype=np.bool_)
        return geom

    def ArgumentParser(self, p=None, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geometry().ArgumentParser(p, *args, **newkw)


add_sile('CAR', carSileVASP, gzip=True)
add_sile('POSCAR', carSileVASP, gzip=True)
add_sile('CONTCAR', carSileVASP, gzip=True)
