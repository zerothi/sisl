"""
Sile object for reading/writing Wannier90 in/output
"""
from pathlib import Path
import numpy as np
from scipy.sparse import lil_matrix

# Import sile objects
from .sile import SileWannier90
from ..sile import *

# Import the geometry object
from sisl import Geometry, SuperCell
from sisl.physics import Hamiltonian
from sisl.unit import unit_convert

__all__ = ['winSileWannier90']


class winSileWannier90(SileWannier90):
    """ Wannier seedname input file object

    This `Sile` enables easy interaction with the Wannier90 code.

    A seedname is the basis of reading all Wannier90 output because
    every file in Wannier90 is based of the name of the seed.

    Hence, if the correct flags are present in the seedname.win file,
    and the corresponding files are created, then the corresponding
    quantity may be read.

    For instance to read the Wannier-centres you *must* have this in your
    seedname.win:

    .. code:: bash

        write_xyz = true
        translate_home_cell = False

    while if you want to read the Wannier Hamiltonian you should have this:

    .. code:: bash

        write_xyz = true
        plot_hr = true
        translate_home_cell = False

    Examples
    --------
    >>> wan90 = get_sile('seedname.win')
    >>> H = wan90.read_hamiltonian()
    >>> H = wan90.read_hamiltonian(dtype=numpy.float64)
    >>> H = wan90.read_hamiltonian(cutoff=0.00001)
    """

    def _setup(self, *args, **kwargs):
        """ Setup `winSileWannier90` after initialization """
        self._comment = ['!', '#']
        self._seed = str(self.file).replace('.win', '')

    def _set_file(self, suffix=None):
        """ Update readed file """
        if suffix is None:
            self._file = Path(self._seed + '.win')
        else:
            self._file = Path(self._seed + suffix)

    @sile_fh_open()
    def _read_supercell(self):
        """ Deferred routine """

        f, l = self.step_to('unit_cell_cart', case=False)
        if not f:
            raise ValueError("The unit-cell vectors could not be found in the seed-file.")

        l = self.readline()
        lines = []
        while not l.startswith('end'):
            lines.append(l)
            l = self.readline()

        # Check whether the first element is a specification of the units
        pos_unit = lines[0].split()
        if len(pos_unit) > 2:
            unit = 1.
        else:
            unit = unit_convert(pos_unit[0].capitalize(), 'Ang')
            # Remove the line with the unit...
            lines.pop(0)

        # Create the cell
        cell = np.empty([3, 3], np.float64)
        for i in [0, 1, 2]:
            cell[i, :] = [float(x) for x in lines[i].split()]

        return SuperCell(cell * unit)

    def read_supercell(self):
        """ Reads a `SuperCell` and creates the Wannier90 cell """
        # Reset
        self._set_file()

        return self._read_supercell()

    @sile_fh_open()
    def _read_geometry_centres(self, *args, **kwargs):
        """ Defered routine """

        nc = int(self.readline())

        # Comment
        self.readline()

        na = 0
        sp = [None] * nc
        xyz = np.empty([nc, 3], np.float64)
        for ia in range(nc):
            l = self.readline().split()
            sp[ia] = l.pop(0)
            if sp[ia] == 'X':
                na = ia + 1
            xyz[ia, :] = [float(k) for k in l[:3]]

        return Geometry(xyz[:na, :], atoms='H')

    @sile_fh_open()
    def _read_geometry(self, sc, *args, **kwargs):
        """ Defered routine """

        is_frac = True
        f, _ = self.step_to('atoms_frac', case=False)
        if not f:
            is_frac = False
            self.fh.seek(0)
            f, _ = self.step_to('atoms_cart', case=False)

        if not f:
            raise ValueError("The geometry coordinates (atoms_frac/cart) could not be found in the seed-file.")

        # Species and coordinate list
        s = []
        xyz = []

        # Read the next line to determine the units
        if is_frac:
            unit = 1.
        else:
            unit = self.readline()
            if len(unit.split()) > 1:
                l = unit.split()
                s.append(l[0])
                xyz.append(list(map(float, l[1:4])))
                unit = 1.
            else:
                unit = unit_convert(unit.strip().capitalize(), 'Ang')

        l = self.readline()
        while not 'end' in l:
            # Get the species and
            l = l.split()
            s.append(l[0])
            xyz.append(list(map(float, l[1:4])))
            l = self.readline()

        # Convert
        xyz = np.array(xyz, np.float64) * unit

        if is_frac:
            xyz = np.dot(sc.cell.T, xyz.T).T

        return Geometry(xyz, atoms=s, sc=sc)

    def read_geometry(self, *args, **kwargs):
        """ Reads a `Geometry` and creates the Wannier90 cell """

        # Read in the super-cell
        sc = self.read_supercell()

        self._set_file('_centres.xyz')
        if self.exist():
            geom = self._read_geometry_centres()
        else:
            self._set_file()
            geom = self._read_geometry(sc, *args, **kwargs)

        # Reset file
        self._set_file()

        # Specify the supercell and return
        geom.set_sc(sc)
        return geom

    @sile_fh_open()
    def _write_supercell(self, sc, fmt='.8f', *args, **kwargs):
        """ Writes the supercel to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

        fmt_str = ' {{0:{0}}} {{1:{0}}} {{2:{0}}}\n'.format(fmt)

        self._write('begin unit_cell_cart\n')
        self._write(' Ang\n')
        self._write(fmt_str.format(*sc.cell[0, :]))
        self._write(fmt_str.format(*sc.cell[1, :]))
        self._write(fmt_str.format(*sc.cell[2, :]))
        self._write('end unit_cell_cart\n')

    def write_supercell(self, sc, fmt='.8f', *args, **kwargs):
        """ Writes the supercell to the contained file """
        self._set_file()
        self._write_supercell(sc, fmt, *args, **kwargs)

    @sile_fh_open()
    def _write_geometry(self, geom, fmt='.8f', *args, **kwargs):
        """ Writes the geometry to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

        # We have to have the _write_supercell here
        # due to the open function re-instantiating the mode,
        # and if it isn't 'a', then it cleans it... :(
        self._write_supercell(geom.sc, fmt, *args, **kwargs)

        fmt_str = ' {{1:2s}} {{2:{0}}} {{3:{0}}} {{4:{0}}} # {{0}}\n'.format(fmt)

        if kwargs.get('frac', False):
            # Get the fractional coordinates
            fxyz = geom.fxyz[:, :]

            self._write('begin atoms_frac\n')
            for ia, a, _ in geom.iter_species():
                self._write(fmt_str.format(ia + 1, a.symbol, *fxyz[ia, :]))
            self._write('end atoms_frac\n')
        else:
            self._write('begin atoms_cart\n')
            self._write(' Ang\n')
            for ia, a, _ in geom.iter_species():
                self._write(fmt_str.format(ia + 1, a.symbol, *geom.xyz[ia, :]))
            self._write('end atoms_cart\n')

    def write_geometry(self, geom, fmt='.8f', *args, **kwargs):
        """ Writes the geometry to the contained file """
        self._set_file()
        self._write_geometry(geom, fmt, *args, **kwargs)

    @sile_fh_open()
    def _read_hamiltonian(self, geom, dtype=np.float64, **kwargs):
        """ Reads a Hamiltonian

        Reads the Hamiltonian model
        """
        cutoff = kwargs.get('cutoff', 0.00001)

        # Rewind to ensure we can read the entire matrix structure
        self.fh.seek(0)

        # Time of creation
        self.readline()

        # Number of orbitals
        no = int(self.readline())
        if no != geom.no:
            raise ValueError(self.__class__.__name__ + '.read_hamiltonian has found inconsistent number '
                             'of orbitals in _hr.dat vs the geometry. Remember to re-run Wannier90?')

        # Number of Wigner-Seitz degeneracy points
        nrpts = int(self.readline())

        # First read across the Wigner-Seitz degeneracy
        # This is formatted with 15 per-line.
        if nrpts % 15 == 0:
            nlines = nrpts
        else:
            nlines = nrpts + 15 - nrpts % 15

        ws = []
        for _ in range(nlines // 15):
            ws.extend(list(map(int, self.readline().split())))

        # Convert to numpy array and invert (for weights)
        ws = 1. / np.array(ws, np.float64).flatten()

        # Figure out the number of supercells
        # and maintain the Hamiltonian in the ham list
        nsc = [0, 0, 0]

        # List for holding the Hamiltonian
        ham = []
        iws = -1

        while True:
            l = self.readline()
            if l == '':
                break

            # Split here...
            l = l.split()

            # Get super-cell, row and column
            iA, iB, iC, r, c = map(int, l[:5])

            nsc[0] = max(nsc[0], abs(iA))
            nsc[1] = max(nsc[1], abs(iB))
            nsc[2] = max(nsc[2], abs(iC))

            # Update index for degeneracy, if required
            if r + c == 2:
                iws += 1

            # Get degeneracy of this element
            f = ws[iws]

            # Store in the Hamiltonian array:
            #   isc
            #   row
            #   column
            #   Hr
            #   Hi
            ham.append(([iA, iB, iC], r-1, c-1, float(l[5]) * f, float(l[6]) * f))

        # Update number of super-cells
        geom.set_nsc([i * 2 + 1 for i in nsc])

        # With the geometry in place we can read in the entire matrix
        # Create a new sparse matrix
        Hr = lil_matrix((geom.no, geom.no_s), dtype=dtype)
        Hi = lil_matrix((geom.no, geom.no_s), dtype=dtype)

        # populate the Hamiltonian by examining the cutoff value
        for isc, r, c, hr, hi in ham:

            # Calculate the column corresponding to the
            # correct super-cell
            c = c + geom.sc_index(isc) * geom.no

            if abs(hr) > cutoff:
                Hr[r, c] = hr
            if abs(hi) > cutoff:
                Hi[r, c] = hi
        del ham

        if np.dtype(dtype).kind == 'c':
            Hr = Hr.tocsr()
            Hi = Hi.tocsr()
            Hr = Hr + 1j*Hi

        return Hamiltonian.fromsp(geom, Hr)

    def read_hamiltonian(self, *args, **kwargs):
        """ Read the electronic structure of the Wannier90 output

        Parameters
        ----------
        cutoff: float, optional
           the cutoff value for the zero Hamiltonian elements, default
           to 0.00001 eV.
        """
        # Retrieve the geometry...
        geom = self.read_geometry()

        # Set file
        self._set_file('_hr.dat')

        H = self._read_hamiltonian(geom, *args, **kwargs)
        self._set_file()
        return H

    def ArgumentParser(self, p=None, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geometry().ArgumentParser(p, *args, **newkw)


add_sile('win', winSileWannier90, gzip=True)
