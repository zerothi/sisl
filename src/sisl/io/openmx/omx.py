# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np

import sisl._array as _a
from sisl import Atom, Geometry, Lattice, SphericalOrbital
from sisl.messages import SislError, warn
from sisl.unit import units

from .._help import *
from ..sile import *
from .sile import SileOpenMX

__all__ = ["omxSileOpenMX"]

_LOGICAL_TRUE = ["on", "yes", "true", ".true.", "ok"]
_LOGICAL_FALSE = ["off", "no", "false", ".false.", "ng"]
_LOGICAL = _LOGICAL_FALSE + _LOGICAL_TRUE


class omxSileOpenMX(SileOpenMX):
    r"""OpenMX-input file

    By supplying base you can reference files in other directories.
    By default the ``base`` is the directory given in the file name.

    Parameters
    ----------
    filename: str
       input file
    mode : str, optional
       opening mode, default to read-only
    base : str, optional
       base-directory to read output files from.

    Examples
    --------
    >>> omx = omxSileOpenMX("tmp/input.dat") # reads output files in 'tmp/' folder
    >>> omx = omxSileOpenMX("tmp/input.dat", base=".") # reads output files in './' folder

    When using this file in conjunction with the `sgeom` script while your input data-files are
    named ``input.dat``, please do this:

    .. code:: bash

        sgeom input.dat{omx} output.xyz

    which forces the use of the omx file.
    """

    @property
    def file(self):
        """Return the current file name (without the directory prefix)"""
        return self._file

    def _setup(self, *args, **kwargs):
        """Setup the `omxSileOpenMX` after initialization"""
        super()._setup(*args, **kwargs)
        # These are the comments
        self._comment = ["#"]

        # List of parent file-handles used while reading
        self._parent_fh = []

    def _pushfile(self, f):
        if self.dir_file(f).is_file():
            self._parent_fh.append(self.fh)
            self.fh = self.dir_file(f).open(self._mode)
        else:
            warn(
                f"{self!s} is trying to include file: {f} but the file seems not to exist? Will disregard file!"
            )

    def _popfile(self):
        if len(self._parent_fh) > 0:
            self.fh.close()
            self.fh = self._parent_fh.pop()
            return True
        return False

    def _seek(self):
        """Closes all files, and starts over from beginning"""
        try:
            while self._popfile():
                pass
            self.fh.seek(0)
        except Exception:
            pass

    @sile_fh_open()
    def _r_key(self, key):
        """Try and read the first occurence of a key

        This will take care of blocks, labels and piped in labels

        Parameters
        ----------
        key : str
           key to find in the file
        """
        self._seek()

        def tokey(key):
            return key.lower()

        keyl = tokey(key)

        def valid_line(line):
            ls = line.strip()
            if len(ls) == 0:
                return False
            return not (ls[0] in self._comment)

        def process_line(line):
            # Split line by spaces
            ls = line.split()
            if len(ls) == 0:
                return None

            # Make a lower equivalent of ls
            lsl = list(map(tokey, ls))

            # The last case is if the key is the first word on the line
            # In that case we have found what we are looking for
            if lsl[0] == keyl:
                return (" ".join(ls[1:])).strip()

            elif lsl[0].startswith("<"):
                # Get key
                lsl_key = lsl[0][1:]
                lsl_end = lsl_key + ">"
                if lsl_key == keyl:
                    # Read in the block content
                    lines = []

                    # Now read lines
                    l = self.readline().strip()
                    while not tokey(l).endswith(lsl_end):
                        if len(l) > 0:
                            lines.append(l)
                        l = self.readline().strip()
                    return lines

            return None

        # Perform actual reading of line
        l = self.readline().split("#")[0]
        if len(l) == 0:
            return None
        l = process_line(l)
        while l is None:
            l = self.readline().split("#")[0]
            if len(l) == 0:
                if not self._popfile():
                    return None
            l = process_line(l)

        return l

    @classmethod
    def _type(cls, value):
        """Determine the type by the value

        Parameters
        ----------
        value : str or list or numpy.ndarray
            the value to check for input-type
        """
        if value is None:
            return None

        if isinstance(value, list):
            # A block, <[name]
            return "B"

        # Grab the entire line (beside the key)
        values = value.split()
        if len(values) == 1:
            val = values[0].lower()
            if val in _LOGICAL:
                # logical
                return "l"

            try:
                float(val)
                if "." in val:
                    # a real number (otherwise an integer)
                    return "r"
                return "i"
            except Exception:
                pass
            # fall-back to name with everything

        return "n"

    @sile_fh_open()
    def type(self, label):
        """Return the type of the fdf-keyword

        Parameters
        ----------
        label : str
            the label to look-up
        """
        self._seek()
        return self._type(self._r_key(label))

    @sile_fh_open()
    def get(self, key, default=None):
        """Retrieve keyword from the file

        Parameters
        ----------
        key : str
            the key to search for
        default : optional
            if the key is not found, this will be the returned value (default to ``None``)

        Returns
        -------
        value : the value of the key. If the key is a block, a `list` is returned, for
                a real value a `float` (or if the default is of `float`), for an integer, an
                `int` is returned.
        """
        # Try and read a line
        value = self._r_key(key)

        # Simply return the default value if not found
        if value is None:
            return default

        # Figure out what it is
        t = self._type(value)

        # We will only do something if it is a real, int, or physical.
        # Else we simply return, as-is
        if t == "r":
            if default is None:
                return float(value)
            t = type(default)
            return t(value)

        elif t == "i":
            if default is None:
                return int(value)
            t = type(default)
            return t(value)

        elif t == "l":
            return value.lower() in _LOGICAL_TRUE

        return value

    def read_basis(self, *args, **kwargs):
        """Reads basis

        Parameters
        ----------
        output: bool, optional
            whether to read lattice from output files (default to read from
            the input file).
        order: {'dat', 'omx'}
            the order of which to try and read the lattice
            If `order` is present `output` is disregarded.
        """
        order = parse_order(
            kwargs.pop("order", None), {True: ["dat", "omx"], False: "omx"}, output
        )
        for f in order:
            v = getattr(self, "_r_basis_{}".format(f.lower()))(*args, **kwargs)
            if v is not None:
                return v
        return None

    def _r_basis_omx(self):
        ns = self.get("Species.Number", 0)
        data = self.get("Definition.of.Atomic.Species")
        if data is None:
            return None

        if ns == 0:
            ns = len(data)
        data = data[:ns]

        def rf_func(R):
            if R > 0:
                r = np.linspace(0, R, 500)
                f = np.ones(500)
                f[r > R] = 0
                return r, f
            return np.linspace(0, 1.0, 10), np.zeros(10)

        def decompose_basis(l):
            # Only split once
            Zr, spec = l.split("-", 1)
            idx = 0
            for i, c in enumerate(Zr):
                if c.isdigit():
                    idx = i
                    break
            R = -1
            if idx == 0:
                Z = Zr
            else:
                Z = Zr[:idx]
                try:
                    R = float(Zr[idx:])
                except Exception:
                    pass

            # Now figure out the orbitals
            orbs = []
            m_order = {
                0: [0],
                1: [1, -1, 0],  # px, py, pz
                2: [0, 2, -2, 1, -1],  # d3z^2-r^2, dx^2-y^2, dxy, dxz, dyz
                3: [
                    0,
                    1,
                    -1,
                    2,
                    -2,
                    3,
                    -3,
                ],  # f5z^2-3r^2, f5xz^2-xr^2, f5yz^2-yr^2, fzx^2-zy^2, fxyz, fx^3-3*xy^2, f3yx^2-y^3
                4: [0, 1, -1, 2, -2, 3, -3, 4, -4],
                5: [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5],
            }
            for i, c in enumerate(spec):
                try:
                    l = "spdfgh".index(c)
                    try:
                        nZ = int(spec[i + 1])
                    except Exception:
                        nZ = 1
                    for z in range(nZ):
                        orbs.extend(
                            SphericalOrbital(l, rf_func(R)).toAtomicOrbital(
                                m=m_order[l], zeta=z + 1
                            )
                        )
                except Exception:
                    pass

            return Z, orbs

        # We are ready to parse
        atom = []
        for dat in data:
            d = dat.split()
            # Figure out the specie
            Z, orbs = decompose_basis(d[1])
            atom.append(Atom(Z, orbs, tag=d[0]))
        return atom

    def read_lattice(self, output: bool = False, *args, **kwargs) -> Lattice:
        """Reads lattice

        One can limit the tried files to only one file by passing
        only a single file ending.

        Parameters
        ----------
        output:
            whether to read lattice from output files (default to read from
            the input file).
        order: {'dat', 'omx'}
            the order of which to try and read the lattice.
            If `order` is present `output` is disregarded.
        """
        order = parse_order(
            kwargs.pop("order", None), {True: ["dat", "omx"], False: "omx"}, output
        )
        for f in order:
            v = getattr(self, "_r_lattice_{}".format(f.lower()))(*args, **kwargs)
            if v is not None:
                return v
        return None

    def _r_lattice_omx(self, *args, **kwargs):
        """Returns `Lattice` object from the omx file"""
        conv = self.get("Atoms.UnitVectors.Unit", default="Ang")
        if conv.upper() == "AU":
            conv = units("Bohr", "Ang")
        else:
            conv = 1.0

        # Read in cell
        cell = np.empty([3, 3], np.float64)

        lc = self.get("Atoms.UnitVectors")
        if not lc is None:
            for i in range(3):
                cell[i, :] = [float(k) for k in lc[i].split()[:3]]
        else:
            raise SileError("Could not find Atoms.UnitVectors in file")
        cell *= conv

        return Lattice(cell)

    _r_lattice_dat = _r_lattice_omx

    def read_geometry(self, output: bool = False, *args, **kwargs) -> Geometry:
        """Returns Geometry object

        One can limit the tried files to only one file by passing
        only a single file ending.

        Parameters
        ----------
        output:
            whether to read geometry from output files (default to read from
            the input file).
        order: {'dat', 'omx'}
            the order of which to try and read the geometry.
            If `order` is present `output` is disregarded.
        """
        order = parse_order(
            kwargs.pop("order", None), {True: ["dat", "omx"], False: "omx"}, output
        )
        for f in order:
            v = getattr(self, "_r_geometry_{}".format(f.lower()))(*args, **kwargs)
            if v is not None:
                return v
        return None

    def _r_geometry_omx(self, *args, **kwargs):
        """Returns `Geometry`"""
        lattice = self.read_lattice(order=["omx"])

        na = self.get("Atoms.Number", default=0)
        conv = self.get("Atoms.SpeciesAndCoordinates.Unit", default="Ang")
        data = self.get("Atoms.SpeciesAndCoordinates")
        if data is None:
            raise SislError("Cannot find key: Atoms.SpeciesAndCoordinates")

        if na == 0:
            # Default to the size of the labels
            na = len(data)

        # Reduce to the number of atoms.
        data = data[:na]

        atoms = self.read_basis(order=["omx"])

        def find_atom(tag):
            if atoms is None:
                return Atom(tag)
            for atom in atoms:
                if atom.tag == tag:
                    return atom
            raise SislError(f"Error when reading the basis for atomic tag: {tag}.")

        xyz = []
        atom = []
        for dat in data:
            d = dat.split()
            atom.append(find_atom(d[1]))
            xyz.append(list(map(float, dat.split()[2:5])))
        xyz = _a.arrayd(xyz)

        if conv == "AU":
            xyz *= units("Bohr", "Ang")
        elif conv == "FRAC":
            xyz = np.dot(xyz, lattice.cell)

        return Geometry(xyz, atoms=atom, lattice=lattice)

    _r_geometry_dat = _r_geometry_omx


add_sile("omx", omxSileOpenMX, case=False, gzip=True)
