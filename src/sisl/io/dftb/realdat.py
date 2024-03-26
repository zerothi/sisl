# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import scipy.sparse as sps
from scipy.sparse import lil_matrix

import sisl._array as _a
from sisl._core import Atom, AtomicOrbital, Atoms, Geometry, Lattice
from sisl.messages import warn
from sisl.physics import Hamiltonian, Overlap

from ..sile import add_sile, sile_fh_open
from .sile import SileDFTB


def _lt2full(M):
    csr = M._csr
    diag = csr.diags(csr.diagonal())
    return M + M.transpose() - diag


class _realSileDFTB(SileDFTB):

    def _setup(self, *args, **kwargs):
        super()._setup(*args, **kwargs)
        self._comment = ["#"]

    def _r_geometry_info(self):
        """Parses the top content of the file for basic info"""

        # 1. Line will contain the number of atoms
        line = self.readline()
        na = int(line)

        # Create the list of atomic information
        nos = []
        atom_neighbors = []

        for ia in range(na):
            # 2. contains:
            line = self.readline()
            #   - the atomic index
            #   - number of neighbors
            #   - number of orbitals on atom
            _, nneighbor, no = map(int, line.split())

            nos.append(no)
            atom_neighbors.append(nneighbor)

        nos = _a.arrayi(nos)
        atom_neighbors = _a.arrayi(atom_neighbors)

        return na, nos, atom_neighbors

    def _r_matrix(self, na, nos, nneighs):

        # Create some minor variables that aids in the
        # Hamiltonian construction.
        no = nos.sum()
        firsto = np.insert(np.cumsum(nos), 0, 0)

        # Create a defaultdict to accommodate arbitrary number of supercells
        # Typically it doesn't get very big, but this is just to accommodate
        # different resulting supercells.
        # The final construction happens in the end.
        # Remember that DFTB+ only returns the lower triangle of the matrix.
        Hsc = defaultdict(lambda: lil_matrix((no, no), dtype=np.float64))

        isc = [0, 0, 0]
        for ia in range(na):
            for inneigh in range(nneighs[ia]):
                line = self.readline()
                ia1, ineigh, ja, isc[0], isc[1], isc[2] = map(int, line.split())
                assert ia1 - 1 == ia
                ja -= 1

                # Get the current coupling matrix
                # The defaultdict will ensure it gets created when needed
                H = Hsc[tuple(isc)]

                io = firsto[ia]
                jo = firsto[ja]
                joe = jo + nos[ja]
                for i in range(nos[ia]):
                    line = self.readline()
                    H[io + i, jo:joe] = list(map(float, line.split()))

        return Hsc

    def _get_atoms(self, na, nos):
        """Create a ficticious `Atoms` object containing orbitals using the default ordering"""

        def get_orbital(io):
            # This is taken directly from the documentation
            # The documentation lists only these shells to be functional
            name = [
                "s",
                "py",
                "pz",
                "px",
                "dxy",
                "dyz",
                "dz2",
                "dxz",
                "dx2-y2",
                "fy(3x2-y2)",
                "fxyz",
                "fz2y",
                "fz3",
                "fz2x",
                "fz(x2-y2)",
                "fx(x2-3y2)",
            ][io]

            return AtomicOrbital(name)

        nos_uniqs = np.unique(nos)

        def get_atom(ia):
            nonlocal nos, nos_uniqs
            no = nos[ia]
            Z = (nos_uniqs == no).nonzero()[0][0] + 1
            return Atom(Z, orbitals=[get_orbital(io) for io in range(no)])

        return Atoms([get_atom(ia) for ia in range(na)])

    @sile_fh_open(True)
    def _r_file(self, geometry: Optional[Geometry] = None):
        """Read content in the current file and return the actual matrices"""
        na, nos, atom_neighbors = self._r_geometry_info()
        Msc = self._r_matrix(na, nos, atom_neighbors)

        # Get supercell size
        nsc = _a.zerosi(3)
        for isc in Msc.keys():
            for i in (0, 1, 2):
                nsc[i] = max(abs(isc[i]), nsc[i])

        # Create the full supercell
        nsc = nsc * 2 + 1

        # Create the lattice vector
        if geometry is None:
            if np.any(nsc > 1):
                warn(
                    f"{self.__class__.__name__}.read_* found a supercell matrix. The returned matrix cannot be used in k-point format since the true atomic coordinates are incorrect, please pass a 'geometry' argument."
                )

            lattice = Lattice(na * 3 + 1, nsc=nsc)
            geometry = Geometry(
                np.arange(na * 3).reshape(na, 3), self._get_atoms(na, nos), lattice
            )
        else:
            geometry = geometry.copy()
            geometry.set_nsc(nsc)

        # Create the big matrix
        M = []
        for isc in geometry.lattice.sc_off:
            M.append(Msc[tuple(isc)].tocsr())

        return geometry, sps.hstack(M)


class overrealSileDFTB(_realSileDFTB):

    def read_overlap(self, geometry: Optional[Geometry] = None) -> Overlap:
        r"""Parse the output overlap matrix created by DFTB+"""
        geometry, S = self._r_file(geometry)
        S.eliminate_zeros()

        # Convert to class
        S = Overlap.fromsp(geometry, S)
        return _lt2full(S)


class hamrealSileDFTB(_realSileDFTB):

    def read_overlap(self, geometry: Optional[Geometry] = None) -> Overlap:
        """Parse the overlap matrix from the ``overreal.dat`` file

        Parameters
        ----------
        geometry:
            define the geometry of the Hamiltonian.
            The data files does *not* contain the geometry information.
            Hence it can be very useful to retrieve the geometry from
            somewhere else.
        """
        orig_file = self.file
        self._file = self.dir_file("overreal.dat")

        # Read in the overlap matrix
        geometry, S = self._r_file(geometry)
        S.eliminate_zeros()

        # Return the file-handle
        self._file = orig_file

        S = Overlap.fromsp(geometry, S)

        return _lt2full(S)

    def read_hamiltonian(self, geometry: Optional[Geometry] = None) -> Hamiltonian:
        r"""Parse the output Hamiltonian created by DFTB+

        This will automatically try to discover the ``hamreal[1-4].dat``
        and ``overreal.dat`` files in the current directory.
        As such the single file read is not really done.

        Parameters
        ----------
        geometry:
            define the geometry of the Hamiltonian.
            The data files does *not* contain the geometry information.
            Hence it can be very useful to retrieve the geometry from
            somewhere else.
        """
        orig_file = self.file

        Hs = []
        for i in range(1, 5):
            self._file = self.dir_file(f"hamreal{i}.dat")
            if not self._file.exists():
                continue

            geometry, H = self._r_file(geometry)
            Hs.append(H)

        # Read in the overlap as well
        self._file = self.dir_file("overreal.dat")
        geometry, S = self._r_file(geometry)

        # Reset the file
        self._file = orig_file

        H = Hamiltonian.fromsp(geometry, Hs, S)

        # Transform back from the charge, x, y, z
        # to the intrinsic way that sisl handles things
        if H.spin.is_noncolinear:
            mat = np.empty([4, 4])
            mat[0] = [0.5, 0, 0, 0.5]
            mat[1] = [0.5, 0, 0, -0.5]
            mat[2] = [0, 0.5, 0, 0]
            mat[3] = [0, 0, -0.5, 0]
            H = H.transform(mat)

        return _lt2full(H)


add_sile("hamreal1.dat", hamrealSileDFTB, gzip=True)
add_sile("hamreal.dat", hamrealSileDFTB, gzip=True)
add_sile("overreal.dat", overrealSileDFTB, gzip=True)
