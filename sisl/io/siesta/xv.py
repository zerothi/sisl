import numpy as np

from ..sile import add_sile, sile_fh_open, sile_raise_write
from .sile import SileSiesta

from sisl._internal import set_module
from sisl import Geometry, Atom, AtomGhost, AtomUnknown, Atoms, SuperCell
from sisl.unit.siesta import unit_convert

__all__ = ['xvSileSiesta']


Bohr2Ang = unit_convert('Bohr', 'Ang')


@set_module("sisl.io.siesta")
class xvSileSiesta(SileSiesta):
    """ Geometry file """

    @sile_fh_open()
    def write_geometry(self, geom, fmt='.9f', velocity=None):
        """ Writes the geometry to the contained file

        Parameters
        ----------
        geom : Geometry
           geometry to write in the XV file
        fmt : str, optional
           the precision used for writing the XV file
        velocity : numpy.ndarray, optional
           velocities to write in the XV file (will be zero if not specified).
           Units input must be in Ang/fs.
        """
        # Check that we can write to the file
        sile_raise_write(self)

        if velocity is None:
            velocity = np.zeros([geom.na, 3], np.float32)
        if geom.xyz.shape != velocity.shape:
            raise SislError(str(self) + '.write_geometry requires the input'
                            'velocity to have equal length to the input geometry.')

        # Write unit-cell
        tmp = np.zeros(6, np.float64)

        # Create format string for the cell-parameters
        fmt_str = ('   ' + ('{:' + fmt + '} ') * 3) * 2 + '\n'
        for i in range(3):
            tmp[0:3] = geom.cell[i, :] / Bohr2Ang
            self._write(fmt_str.format(*tmp))
        self._write(f'{geom.na:12d}\n')

        # Create format string for the atomic coordinates
        fmt_str = '{:3d}{:6d} '
        fmt_str += ('{:' + fmt + '} ') * 3 + '   '
        fmt_str += ('{:' + fmt + '} ') * 3 + '\n'
        for ia, a, ips in geom.iter_species():
            tmp[0:3] = geom.xyz[ia, :] / Bohr2Ang
            tmp[3:] = velocity[ia, :] / Bohr2Ang
            if isinstance(a, AtomGhost):
                self._write(fmt_str.format(ips + 1, -a.Z, *tmp))
            else:
                self._write(fmt_str.format(ips + 1, a.Z, *tmp))

    @sile_fh_open()
    def read_supercell(self):
        """ Returns `SuperCell` object from the XV file """

        cell = np.empty([3, 3], np.float64)
        for i in range(3):
            cell[i, :] = list(map(float, self.readline().split()[:3]))
        cell *= Bohr2Ang

        return SuperCell(cell)

    @sile_fh_open()
    def read_geometry(self, velocity=False, species_Z=False):
        """ Returns a `Geometry` object from the XV file

        Parameters
        ----------
        species_Z : bool, optional
           if ``True`` the atomic numbers are the species indices (useful when
           reading the ChemicalSpeciesLabel block simultaneously).
        velocity : bool, optional
           also return the velocities in the file

        Returns
        -------
        Geometry
        velocity : only if `velocity` is true.
        """
        sc = self.read_supercell()

        # Read number of atoms
        na = int(self.readline())
        xyz = np.empty([na, 3], np.float64)
        vel = np.empty([na, 3], np.float64)
        atms = [None] * na
        sp = np.empty([na], np.int32)
        for ia in range(na):
            line = list(map(float, self.readline().split()[:8]))
            sp[ia] = int(line[0])
            if species_Z:
                atms[ia] = Atom(sp[ia])
            else:
                atms[ia] = Atom(int(line[1]))
            xyz[ia, :] = line[2:5]
            vel[ia, :] = line[5:8]

        xyz *= Bohr2Ang
        vel *= Bohr2Ang

        # Ensure correct sorting
        max_s = sp.max()
        sp -= 1
        # Ensure we can remove the atom after having aligned them
        atms2 = Atoms(AtomUnknown(1000), na=na)
        for i in range(max_s):
            idx = (sp[:] == i).nonzero()[0]
            if len(idx) == 0:
                # Always ensure we have "something" for the unoccupied places
                atms2[idx] = AtomUnknown(1000 + i)
            else:
                atms2[idx] = atms[idx[0]]

        geom = Geometry(xyz, atms2.reduce(), sc=sc)
        if velocity:
            return geom, vel
        return geom

    @sile_fh_open()
    def read_velocity(self):
        """ Returns an array with the velocities from the XV file

        Returns
        -------
        velocity : 
        """
        self.read_supercell()
        na = int(self.readline())
        vel = np.empty([na, 3], np.float64)
        for ia in range(na):
            line = list(map(float, self.readline().split()[:8]))
            vel[ia, :] = line[5:8]

        vel *= Bohr2Ang
        return vel

    read_data = read_velocity

    def ArgumentParser(self, p=None, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geometry().ArgumentParser(p, *args, **newkw)


add_sile('XV', xvSileSiesta, gzip=True)
