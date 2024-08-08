# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import re
from collections.abc import Iterator

import numpy as np

import sisl._array as _a
from sisl import Geometry, Lattice
from sisl._internal import set_module
from sisl.messages import warn
from sisl.physics.brillouinzone import BrillouinZone
from sisl.physics.phonon import EigenmodePhonon
from sisl.unit.siesta import unit_convert

from ..sile import add_sile
from .sile import SileSiesta

__all__ = ["vectorsSileSiesta"]

_Bohr2Ang = unit_convert("Bohr", "Ang")
_cm1_eV = unit_convert("cm^-1", "eV")


@set_module("sisl.io.siesta")
class vectorsSileSiesta(SileSiesta):
    """Phonon eigenmode file

    Parameters
    ----------
    parent : obj, optional
        a parent may contain a DynamicalMatrix, Geometry or Lattice
    geometry : Geometry, optional
        a geometry contains a cell with corresponding lattice vectors
        used to convert k [1/Ang] -> [b], and the atomic masses
    """

    def _setup(self, *args, **kwargs):
        """Simple setup that needs to be overwritten

        All _r_next_* methods expect the file unit to be handled
        and that the position in the file is correct.
        """
        super()._setup(*args, **kwargs)

        # default lattice
        lattice = None

        parent = kwargs.get("parent")
        if parent is None:
            geometry = None
        elif isinstance(parent, Geometry):
            geometry = parent
        elif isinstance(parent, Lattice):
            lattice = parent
        else:
            geometry = parent.geometry

        geometry = kwargs.get("geometry", geometry)
        if geometry is not None and lattice is None:
            lattice = geometry.lattice

        lattice = kwargs.get("lattice", lattice)
        if lattice is None and geometry is not None:
            raise ValueError(
                f"{self.__class__.__name__}(geometry=Geometry, lattice=None) is not an allowed argument combination."
            )

        if parent is None:
            parent = geometry
        if parent is None:
            parent = lattice

        self._parent = parent
        self._geometry = geometry
        self._lattice = lattice
        if self._parent is None and self._geometry is None and self._lattice is None:

            def conv(k):
                if not np.allclose(k, 0.0):
                    warn(
                        f"{self.__class__.__name__} cannot convert stored k-points from 1/Ang to reduced coordinates. "
                        "Please ensure parent=DynamicalMatrix, geometry=Geometry, or lattice=Lattice to calculate reduced k."
                    )
                return k / _Bohr2Ang

        else:

            def conv(k):
                return (k @ lattice.cell.T) / (2 * np.pi * _Bohr2Ang)

        self._convert_k = conv

    def _open(self, rewind: bool = False):
        """Open the file

        Here we also initialize some variables to keep track of the state of the read.
        """
        super()._open()
        if rewind:
            self.seek(0)

        # Here we initialize the variables that will keep track of the state of the read.
        # The process for identification is done on this basis:
        #  _ik is the current (Python) index for the k-point to be read
        #  _state is:
        #        -1 : the file-descriptor has just been opened (i.e. in front of header)
        #         0 : it means that the file-descriptor is in front of k point information
        self._state = -1
        self._ik = 0

        # read in sizes
        self._sizes = self._r_next_sizes()
        self._state = 0

    def _r_next_sizes(self) -> None:
        """Determine the dimension of the data stored in the vectors file"""
        pos = self.fh.tell()

        # Skip past header
        self.readline()  # Empty line
        self.readline()  # K point
        self.readline()  # Eigenmode index
        self.readline()  # Frequency
        self.readline()  # Eigenmode header

        # Determine number of atoms (i.e. rows per mode)
        natoms = 0
        while True:
            line = self.readline()
            if line.startswith("Eigenmode"):
                break
            else:
                natoms += 1
        self._natoms = natoms

        # Determine number of modes
        nmodes = 1
        while True:
            line = self.readline()
            if re.match("k *=", line):
                break
            elif line.startswith("Eigenvector"):
                nmodes += 1
        self._nmodes = nmodes

        # Rewind
        self.seek(pos)

    def _r_next_eigenmode(self) -> EigenmodePhonon:
        """Reads the next phonon eigenmode in the vectors file.

        Returns
        --------
        EigenmodePhonon:
            The next eigenmode.
        """
        # Skip empty line at the head of the file
        self.readline()

        k = _a.fromiterd(map(float, self.readline().split()[2:]))
        if len(k) == 0:
            return None
        # Read first eigenvector index

        # Determine number of atoms (i.e. rows per mode)
        mode = _a.emptyz([self._nmodes, self._natoms, 3])
        c = _a.emptyd(self._nmodes)
        for imode in range(self._nmodes):
            self.readline()

            # Read frequency
            c[imode] = float(self.readline().split("=")[1])

            # Read real part of eigenmode
            # Skip eigenmode header
            self.readline()
            for iatom in range(self._natoms):
                line = self.readline()
                mode[imode, iatom].real = list(map(float, line.split()))

            # Read imaginary part of eigenmode
            # Skip eigenmode header
            self.readline()
            for iatom in range(self._natoms):
                line = self.readline()
                mode[imode, iatom].imag = list(map(float, line.split()))

        info = dict(k=self._convert_k(k), parent=self._parent, gauge="orbital")
        return EigenmodePhonon(mode.reshape(self._nmodes, -1), c * _cm1_eV, **info)

    def yield_eigenmode(self) -> Iterator[EigenmodePhonon]:
        """Iterates over the modes in the vectors file

        Yields
        ------
        EigenmodePhonon
        """
        # Open file and get parsing information
        self._open()

        # Iterate over all eigenmodes in the vectors file, yielding control to the caller at
        # each iteration.
        mode = self._r_next_eigenmode()
        while mode is not None:
            yield mode
            mode = self._r_next_eigenmode()

    def read_eigenmode(self, k=(0, 0, 0), atol: float = 1e-4) -> EigenmodePhonon:
        """Reads a specific eigenmode from the file.

        This method iterates over the modes until it finds a match. Do not call
        this method repeatedly. If you want to loop eigenmodes, use `yield_eigenmode`.

        Parameters
        ----------
        k: array-like of shape (3,), optional
            The k point of the mode you want to find.
        atol:
            The threshold value for considering two k-points the same (i.e. to match
            the query k point with the modes k point).

        See Also
        --------
        yield_eigenmode

        Returns
        -------
        EigenmodePhonon or None:
            If found, the mode that was queried.

        Raises
        ------
        LookupError :
            in case the requested k-point can not be found in the file.
        """
        # Iterate over all eigenmodes in the file
        for mode in self.yield_eigenmode():
            if np.allclose(mode.info["k"], k, atol=atol):
                # This is the mode that the user requested
                return mode
        raise LookupError(
            f"{self.__class__.__name__}.read_eigenmode could not find k-point: {k!s} eigenmode"
        )

    def read_brillouinzone(self) -> BrillouinZone:
        """Read the brillouin zone object (only containing the k-points)"""
        k = [mode.info["k"] for mode in self.yield_eigenmode()]
        return BrillouinZone(self._parent, k=k)


add_sile("vectors", vectorsSileSiesta, gzip=True)
