# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Sile object for reading/writing Wannier90 in/output
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sps
from scipy.sparse import lil_matrix

import sisl._array as _a
from sisl import Geometry, Lattice
from sisl.messages import deprecate_argument
from sisl.physics import Hamiltonian
from sisl.unit import unit_convert

from ..sile import *

# Import sile objects
from .sile import SileWannier90

__all__ = ["winSileWannier90"]


def _construct_hamiltonian(geometry: Geometry, Hsc):
    # Get supercell size
    nsc = _a.zerosi(3)
    for isc in Hsc.keys():
        for i in (0, 1, 2):
            nsc[i] = max(abs(isc[i]), nsc[i])

    # Create the full supercell
    nsc = nsc * 2 + 1
    geometry.set_nsc(nsc)

    # Create the big matrix
    H = []
    for isc in geometry.lattice.sc_off:
        H.append(Hsc[tuple(isc)].tocsr())

    H = sps.hstack(H)

    return Hamiltonian.fromsp(geometry, H)


class winSileWannier90(SileWannier90):
    """Wannier seedname input file object

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
        """Setup `winSileWannier90` after initialization"""
        super()._setup(*args, **kwargs)
        self._comment = ["!", "#"]
        self._seed = str(self.file).replace(".win", "")

    def _set_file(self, suffix=None):
        """Update readed file"""
        if suffix is None:
            self._file = Path(self._seed + ".win")
        else:
            self._file = Path(self._seed + suffix)

    @sile_fh_open()
    def _r_lattice(self):
        """Deferred routine"""

        f, l = self.step_to("unit_cell_cart", case=False)
        if not f:
            raise ValueError(
                "The unit-cell vectors could not be found in the seed-file."
            )

        l = self.readline()
        lines = []
        while not l.startswith("end"):
            lines.append(l)
            l = self.readline()

        # Check whether the first element is a specification of the units
        pos_unit = lines[0].split()
        if len(pos_unit) > 2:
            unit = 1.0
        else:
            unit = unit_convert(pos_unit[0].capitalize(), "Ang")
            # Remove the line with the unit...
            lines.pop(0)

        # Create the cell
        cell = np.empty([3, 3], np.float64)
        for i in [0, 1, 2]:
            cell[i, :] = [float(x) for x in lines[i].split()]

        return Lattice(cell * unit)

    def read_lattice(self):
        """Reads a `Lattice` and creates the Wannier90 cell"""
        # Reset
        self._set_file()

        return self._r_lattice()

    @sile_fh_open()
    def _r_geometry_centres(self, *args, **kwargs):
        """Defered routine"""

        nc = int(self.readline())

        # Comment
        self.readline()

        na = 0
        sp = [None] * nc
        xyz = np.empty([nc, 3], np.float64)
        for ia in range(nc):
            l = self.readline().split()
            sp[ia] = l.pop(0)
            if sp[ia] == "X":
                na = ia + 1
            xyz[ia, :] = [float(k) for k in l[:3]]

        return Geometry(xyz[:na, :], atoms="H")

    @sile_fh_open()
    def _r_geometry(self, lattice, *args, **kwargs):
        """Defered routine"""

        is_frac = True
        f, _ = self.step_to("atoms_frac", case=False)
        if not f:
            is_frac = False
            self.fh.seek(0)
            f, _ = self.step_to("atoms_cart", case=False)

        if not f:
            raise ValueError(
                "The geometry coordinates (atoms_frac/cart) could not be found in the seed-file."
            )

        # Species and coordinate list
        s = []
        xyz = []

        # Read the next line to determine the units
        if is_frac:
            unit = 1.0
        else:
            unit = self.readline()
            if len(unit.split()) > 1:
                l = unit.split()
                s.append(l[0])
                xyz.append(list(map(float, l[1:4])))
                unit = 1.0
            else:
                unit = unit_convert(unit.strip().capitalize(), "Ang")

        l = self.readline()
        while not "end" in l:
            # Get the species and
            l = l.split()
            s.append(l[0])
            xyz.append(list(map(float, l[1:4])))
            l = self.readline()

        # Convert
        xyz = np.array(xyz, np.float64) * unit

        if is_frac:
            xyz = np.dot(lattice.cell.T, xyz.T).T

        return Geometry(xyz, atoms=s, lattice=lattice)

    @deprecate_argument(
        "sc", "lattice", "use lattice= instead of sc=", from_version="0.15"
    )
    def read_geometry(self, *args, **kwargs):
        """Reads a `Geometry` and creates the Wannier90 cell"""

        # Read in the super-cell
        lattice = self.read_lattice()

        self._set_file("_centres.xyz")
        if self.file.is_file():
            geom = self._r_geometry_centres()
        else:
            self._set_file()
            geom = self._r_geometry(lattice, *args, **kwargs)

        # Reset file
        self._set_file()

        # Specify the supercell and return
        geom.set_lattice(lattice)
        return geom

    @sile_fh_open()
    def _write_lattice(self, lattice, fmt=".8f", *args, **kwargs):
        """Writes the supercel to the contained file"""
        # Check that we can write to the file
        sile_raise_write(self)

        fmt_str = " {{0:{0}}} {{1:{0}}} {{2:{0}}}\n".format(fmt)

        self._write("begin unit_cell_cart\n")
        self._write(" Ang\n")
        self._write(fmt_str.format(*lattice.cell[0, :]))
        self._write(fmt_str.format(*lattice.cell[1, :]))
        self._write(fmt_str.format(*lattice.cell[2, :]))
        self._write("end unit_cell_cart\n")

    @deprecate_argument(
        "sc", "lattice", "use lattice= instead of sc=", from_version="0.15"
    )
    def write_lattice(self, lattice, fmt=".8f", *args, **kwargs):
        """Writes the supercell to the contained file"""
        self._set_file()
        self._write_lattice(lattice, fmt, *args, **kwargs)

    @sile_fh_open()
    def _write_geometry(self, geom, fmt=".8f", *args, **kwargs):
        """Writes the geometry to the contained file"""
        # Check that we can write to the file
        sile_raise_write(self)

        # We have to have the _write_lattice here
        # due to the open function re-instantiating the mode,
        # and if it isn't 'a', then it cleans it... :(
        self._write_lattice(geom.lattice, fmt, *args, **kwargs)

        fmt_str = " {{1:2s}} {{2:{0}}} {{3:{0}}} {{4:{0}}} # {{0}}\n".format(fmt)

        if kwargs.get("frac", False):
            # Get the fractional coordinates
            fxyz = geom.fxyz[:, :]

            self._write("begin atoms_frac\n")
            for ia, a, _ in geom.iter_species():
                self._write(fmt_str.format(ia + 1, a.symbol, *fxyz[ia, :]))
            self._write("end atoms_frac\n")
        else:
            self._write("begin atoms_cart\n")
            self._write(" Ang\n")
            for ia, a, _ in geom.iter_species():
                self._write(fmt_str.format(ia + 1, a.symbol, *geom.xyz[ia, :]))
            self._write("end atoms_cart\n")

    def write_geometry(self, geom, fmt=".8f", *args, **kwargs):
        """Writes the geometry to the contained file"""
        self._set_file()
        self._write_geometry(geom, fmt, *args, **kwargs)

    def _r_wigner_seitz_weights(self):
        # Number of Wigner-Seitz degeneracy points
        npts = int(self.readline())

        ws = []
        while len(ws) < npts:
            # both formats uses 15 points per line,
            # however, this should be usable if they decide to change
            # the number of counts per line.
            ws.extend(list(map(int, self.readline().split())))

        ws = 1.0 / _a.arrayd(ws)
        return ws

    @sile_fh_open()
    def _r_hamiltonian_hr(self, geometry, dtype=np.float64, **kwargs):
        """Reads a Hamiltonian model from the _hr.dat file"""
        cutoff = kwargs.get("cutoff", 0.00001)

        # Rewind to ensure we can read the entire matrix structure
        self.fh.seek(0)

        # Time of creation
        self.readline()

        # Number of orbitals
        no = int(self.readline())
        if no != geometry.no:
            raise ValueError(
                f"{self.__class__.__name__}"
                ".read_hamiltonian has found inconsistent number "
                "of orbitals in _hr.dat vs the geometry. Remember to re-run Wannier90?"
            )

        ws = self._r_wigner_seitz_weights()

        # List for holding the Hamiltonian
        Hsc = defaultdict(lambda: lil_matrix((geometry.no, geometry.no), dtype=dtype))
        is_complex = np.iscomplexobj(dtype(1))

        iws = -1
        isc = [0, 0, 0]
        while True:
            l = self.readline()
            if l == "":
                break

            # Split here...
            l = l.split()

            # Get super-cell, row and column
            isc[0], isc[1], isc[2], r, c = map(int, l[:5])

            # Update index for degeneracy, if required
            if r + c == 2:
                iws += 1

            # Get degeneracy of this element
            w = ws[iws]

            # Scale matrix elements
            hr = float(l[5]) * w
            if is_complex:
                h = hr + 1j * float(l[6]) * w
            else:
                h = hr

            if abs(h) > cutoff:
                Hsc[tuple(isc)][r - 1, c - 1] = h

        return _construct_hamiltonian(geometry, Hsc)

    @sile_fh_open()
    def _r_hamiltonian_tb(self, geometry, dtype=np.float64, **kwargs):
        """Reads a Hamiltonian model from the _tb.dat file"""
        cutoff = kwargs.get("cutoff", 0.00001)

        # Rewind to ensure we can read the entire matrix structure
        self.fh.seek(0)

        # Time of creation
        self.readline()

        #  Lattice vectors [Ang]
        for _ in range(3):
            self.readline()
        # TODO check against self.lattice?!

        # Number of orbitals
        no = int(self.readline())
        if no != geometry.no:
            raise ValueError(
                f"{self.__class__.__name__}"
                ".read_hamiltonian has found inconsistent number "
                "of orbitals in _hr.dat vs the geometry. Remember to re-run Wannier90?"
            )

        ws = self._r_wigner_seitz_weights()

        # List for holding the Hamiltonian
        Hsc = defaultdict(lambda: lil_matrix((geometry.no, geometry.no), dtype=dtype))
        is_complex = np.iscomplexobj(dtype(1))

        # Parse hamiltonian matrix elements
        for w in ws:
            l = self.readline()  # Skip empty line
            if not l.strip() == "":
                raise ValueError(
                    f"{self.__class__.__name__}"
                    ".read_hamiltonian unable to parse <>_tb.dat file, due to "
                    "error in file format."
                )

            # Get super-cell
            isc = map(int, self.readline().split())

            # Get Hamiltonian matrix elements
            Hr = Hsc[tuple(isc)]
            for _ in range(no**2):
                l = self.readline().split()

                # Get row and column):
                r, c = map(int, l[:2])

                # Scale matrix elements
                hr = float(l[2]) * w
                if is_complex:
                    h = hr + 1j * float(l[3]) * w
                else:
                    h = hr

                if abs(h) > cutoff:
                    Hr[r - 1, c - 1] = h

        return _construct_hamiltonian(geometry, Hsc)

    def read_hamiltonian(self, cutoff: float = 1e-5, *args, **kwargs):
        """Read the electronic structure of the Wannier90 output by reading the <>_tb.dat, <>_hr.dat

        Parameters
        ----------
        cutoff:
           the cutoff value for the zero Hamiltonian elements, default
           to 0.00001 eV.

        dtype: np.float64, optional
            the default data-type used for the matrix.
            Is mainly useful to check whether the TB model has imaginary
            components (it should not since it is a Wannier model).
        """
        # Retrieve the geometry...
        geom = self.read_geometry()

        order = kwargs.pop("order", ["tb", "hr"])
        for f in order:
            # Set file
            self._set_file(f"_{f.lower()}.dat")

            # only parse if the file exists
            if self._file.exists():
                H = getattr(self, f"_r_hamiltonian_{f.lower()}")(
                    geom, *args, cutoff=cutoff, **kwargs
                )
            self._set_file()
            if H is not None:
                return H

        return None

    def ArgumentParser(self, p=None, *args, **kwargs):
        """Returns the arguments that is available for this Sile"""
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geometry().ArgumentParser(p, *args, **newkw)


add_sile("win", winSileWannier90, gzip=True)
